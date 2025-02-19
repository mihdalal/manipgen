from collections import Counter, deque
import os
import shutil
import signal
import time
import cv2
import imageio
import wandb
from collections import OrderedDict
from isaacgym import gymapi, gymutil, gymtorch  # must import isaacgym before pytorch
import numpy as np
from manipgen.utils.rlgames_utils import load_model, get_actions

import torch
from torch.utils.data import DataLoader

from isaacgymenvs.utils.utils import set_seed
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from isaacgymenvs.utils.torch_jit_utils import quat_mul, quat_conjugate
from isaacgymenvs.tasks.factory.factory_control import axis_angle_from_quat

import hydra
from omegaconf import DictConfig

from manipgen_robomimic.utils.log_utils import custom_tqdm as tqdm  # use robomimic tqdm that prints to stdout

from manipgen.envs import environments
import manipgen.utils.robomimic_utils as RMUtils
from manipgen.utils.obs_utils import get_visual_obs_handler, get_seg_obs_handler
from manipgen.utils.media_utils import camera_shot, vis_depth, apply_mask
from manipgen.utils.dagger_utils import *

class Storage(object):
    def __init__(
            self, buffer_size, num_envs, 
            obs_shape, visual_obs_shape, actions_shape, 
            visual_obs_type="pcd", visual_obs_handler=None, use_seg_obs=False, seg_obs_handler=None,
            output_device="cuda:0", storage_devices=[0], traj_length=120, seq_length=1, frame_stack=0,
        ):
        """Storage for storing expert data on GPU.

        Args:
            buffer_size (int): How many trajectories to store per GPU, this will multiply num envs to get total num trajectories.
            num_envs (int): Number of environments.
            obs_shape (tuple): Shape of the state obs.
            visual_obs_shape (tuple): Shape of the visual obs.
            actions_shape (tuple): Shape of the actions.
            visual_obs_type (str, optional): Type of the visual obs. Defaults to "pcd". Can also be "depth".
            visual_obs_handler (ObsHandler, optional): This class handles pre-processing the obs and applying data augmentation. Defaults to None.
            use_seg_obs (bool, optional): Whether to use segmentation observations from the first frame. This enables selecting the object to manipulate from a clutter. Defaults to False.
            seg_obs_handler (SegmentationObsHandler, optional): This class handles pre-processing the segmentation obs and applying data augmentation. Defaults to None.
            device (str, optional): Device id to store the data on.. Defaults to "cuda:0".
            traj_length (int, optional): The max length of the trajectories, assumed to be the same for all envs for now. Defaults to 120.
            seq_length (int, optional): The length of the RNN horizon. Defaults to 1. If seq_length > 1, then we are using RNN, framestack should be 0. This naming convention is taken from robomimic.
            frame_stack (int, optional): The length of the transformer horizon. Defaults to 0. If frame_stack > 0, then we are using transformer, seq_length should be 1. This naming convention is taken from robomimic.
        """
        self.output_device = output_device
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.visual_obs_type = visual_obs_type
        self.visual_obs_handler = visual_obs_handler
        self.use_seg_obs = use_seg_obs
        self.seg_obs_handler = seg_obs_handler
        traj_length = traj_length
        self.traj_length = traj_length
        self.seq_length = seq_length

        buffer_size = buffer_size * num_envs

        # Core
        # The buffer is organized as full trajectories, with additional padding pre-defined by frame_stack (pre-padding) and seq_length (post-padding)
        # split buffer evenly across storage devices
        total_traj_length = frame_stack + traj_length+seq_length -1
        self.obs, self.visual_obs, self.seg_obs, self.dones, self.rewards, self.actions = [], [], [], [], [], []
        for device in storage_devices:
            device = f"cuda:{device}"
            self.obs.append(torch.zeros(buffer_size, total_traj_length, *obs_shape, device=device))
            self.visual_obs.append(torch.zeros(buffer_size, total_traj_length, *visual_obs_shape, device=device))
            self.dones.append(torch.zeros(buffer_size, total_traj_length, device=device, dtype=torch.bool))
            self.rewards.append(torch.zeros(buffer_size, total_traj_length, device=device))
            self.actions.append(torch.zeros(buffer_size, total_traj_length, *actions_shape, device=device))
            if self.use_seg_obs:
                self.seg_obs.append(torch.zeros(buffer_size, 1, *visual_obs_shape, device=device))              # only for the first frame
                        
        self.cur = 0
        self.step = 0
        self.ep_step = frame_stack
        self.frame_stack = frame_stack
        self.current_device_idx = 0
        self.storage_devices = storage_devices
        self.valid_buffers = 0

    def add_transitions(self, obs, visual_obs, actions, rewards, dones, seg_obs=None):
        """Add a transition to the storage across all environments.
        Note ep_step is the current step in the trajectory and then for the start and the end we 
        have particular logic for handling the padding.

        Args:
            obs (torch.Tensor): The state obs.
            visual_obs (torch.Tensor): The visual obs.
            actions (torch.Tensor): The actions.
            rewards (torch.Tensor): The rewards.
            dones (torch.Tensor): The dones.
        """
        start = self.cur * self.num_envs
        end = start + self.num_envs
        current_device = self.storage_devices[self.current_device_idx]
        obs = obs.to(f"cuda:{current_device}")
        visual_obs = visual_obs.to(f"cuda:{current_device}")
        actions = actions.to(f"cuda:{current_device}")
        rewards = rewards.to(f"cuda:{current_device}")
        dones = dones.to(f"cuda:{current_device}")
        
        if self.ep_step == self.frame_stack and self.frame_stack > 0:
            # pad the start of the trajectory according to frame_stack
            # padding is repeating the first element of the trajectory
            self.obs[current_device][start:end, :self.ep_step] = obs[:, :7].unsqueeze(1)
            self.visual_obs[current_device][start:end, :self.ep_step] = visual_obs.unsqueeze(1)
            self.rewards[current_device][start:end, :self.ep_step] = rewards.unsqueeze(1)
            self.dones[current_device][start:end, :self.ep_step] = dones.unsqueeze(1)
            self.actions[current_device][start:end, :self.ep_step] = actions.unsqueeze(1)
        self.obs[self.current_device_idx][start:end, self.ep_step].copy_(obs[:, :7])
        self.visual_obs[self.current_device_idx][start:end, self.ep_step].copy_(visual_obs)
        self.rewards[self.current_device_idx][start:end, self.ep_step].copy_(rewards)
        self.dones[self.current_device_idx][start:end, self.ep_step].copy_(dones)
        self.actions[self.current_device_idx][start:end, self.ep_step].copy_(actions)
        if self.ep_step == self.frame_stack and self.use_seg_obs:
            self.seg_obs[self.current_device_idx][start:end, 0].copy_(seg_obs)

        self.ep_step += 1
        if dones.any():
            # pad end of the trajectory with the last element all the way to the end
            if self.seq_length > 1:
                self.obs[self.current_device_idx][start:end, self.traj_length:] = self.obs[start:end, self.traj_length-1].unsqueeze(1)
                self.visual_obs[self.current_device_idx][start:end, self.traj_length:] = self.visual_obs[start:end, self.traj_length-1].unsqueeze(1)
                self.rewards[self.current_device_idx][start:end, self.traj_length:] = self.rewards[start:end, self.traj_length-1].unsqueeze(1)
                self.dones[self.current_device_idx][start:end, self.traj_length:] = self.dones[start:end, self.traj_length-1].unsqueeze(1)
                self.actions[self.current_device_idx][start:end, self.traj_length:] = self.actions[start:end, self.traj_length-1].unsqueeze(1)
            
            # note for our tasks everything has the same traj length
            self.ep_step = self.frame_stack
            self.step = self.step + 1
            if self.step % self.buffer_size == 0:
                # if we go beyond the buffer, then move on to the next buffer
                self.current_device_idx = (self.current_device_idx + 1) % len(self.storage_devices)
                self.valid_buffers += 1
                self.valid_buffers = min(self.valid_buffers, len(self.storage_devices))
            self.cur = self.step % self.buffer_size

    def mini_batch_generator(self, mini_batch_size):
        """Generate mini-batches of indices for training.
        Split the buffer into mini-batches of size mini_batch_size.

        Args:
            mini_batch_size (int): The size of the mini-batch.

        Returns:
            indices (torch.Tensor): A list of indices for the mini-batches.
        """
        indices = torch.randperm(self.buffer_size * self.num_envs * self.traj_length * self.valid_buffers, device=self.output_device)
        indices = indices.split(mini_batch_size)
        return indices
    
    def get_batch(self, indices):
        """ Get a batch of data from the storage. 
        Handle the sampling for RNN and transformer.
        RNN: we sample a sequence of length seq_length, starting from the sampled indices.
        Transformer: we sample a sequence of length frame_stack, starting from sampled indices - frame_stack.

        Args:
            indices (torch.Tensor): The indices to sample from the storage.
        
        Returns:
            obs (torch.Tensor): The state obs.
            visual_obs (torch.Tensor): The visual obs.
            actions (torch.Tensor): The actions.
        """
        # get a list of batch indices per buffer
        buffer_assignment = (indices // (self.traj_length * self.buffer_size * self.num_envs))
        # need to split the indices into list of indices per buffer
        buffer_indices = [indices[buffer_assignment == i] % (self.traj_length * self.buffer_size * self.num_envs) for i in range(self.valid_buffers)]
        obs_combined, visual_obs_combined, actions_combined, first_obs_combined, first_visual_obs_combined, prev_obs_combined, first_seg_obs_combined = [], [], [], [], [], [], []
        for buffer_idx in range(self.valid_buffers):
            current_device_id = self.storage_devices[buffer_idx]
            buffer_indices_ = buffer_indices[buffer_idx].to(f"cuda:{current_device_id}")
            batch_indices = buffer_indices_ // self.traj_length
            seq_indices = buffer_indices_ % self.traj_length + self.frame_stack
            if self.frame_stack == 0:
                # RNN: we count forwards from the sampled index
                obs = [self.obs[buffer_idx][batch_indices, seq_indices + i] for i in range(self.seq_length)]
                visual_obs = [self.visual_obs[buffer_idx][batch_indices, seq_indices + i] for i in range(self.seq_length)]
                actions = [self.actions[buffer_idx][batch_indices, seq_indices + i] for i in range(self.seq_length)]
                prev_obs = [self.obs[buffer_idx][batch_indices, torch.maximum(seq_indices + i - 1, torch.zeros_like(seq_indices))] for i in range(self.seq_length)]
                obs = torch.stack(obs, dim=1)
                visual_obs = torch.stack(visual_obs, dim=1)
                actions = torch.stack(actions, dim=1)
                prev_obs = torch.stack(prev_obs, dim=1)

                first_obs = self.obs[buffer_idx][batch_indices, 0:1].repeat((1, self.seq_length, *([1]*len(self.obs[buffer_idx].shape[2:]))))
                first_visual_obs = self.visual_obs[buffer_idx][batch_indices, 0:1].repeat((1, self.seq_length, *([1]*len(self.visual_obs[buffer_idx].shape[2:]))))
                if self.use_seg_obs:
                    first_seg_obs = self.seg_obs[buffer_idx][batch_indices, 0:1].repeat((1, self.seq_length, *([1]*len(self.seg_obs[buffer_idx].shape[2:]))))
            else:
                # for transformer: we count forwards from the sampled index - frame_stack
                obs = [self.obs[buffer_idx][batch_indices, seq_indices - i] for i in range(self.frame_stack - 1, -1, -1)]
                visual_obs = [self.visual_obs[buffer_idx][batch_indices, seq_indices - i] for i in range(self.frame_stack - 1, -1, -1)]
                actions = [self.actions[buffer_idx][batch_indices, seq_indices - i] for i in range(self.frame_stack - 1, -1, -1)]
                prev_obs = [self.obs[buffer_idx][batch_indices, torch.maximum(seq_indices - i - 1, torch.zeros_like(seq_indices))] for i in range(self.frame_stack - 1, -1, -1)]
                obs = torch.stack(obs, dim=1)
                visual_obs = torch.stack(visual_obs, dim=1)
                actions = torch.stack(actions, dim=1)
                prev_obs = torch.stack(prev_obs, dim=1)

                first_obs = self.obs[buffer_idx][batch_indices, 0:1].repeat((1, self.frame_stack, *([1]*len(self.obs[buffer_idx].shape[2:]))))
                first_visual_obs = self.visual_obs[buffer_idx][batch_indices, 0:1].repeat((1, self.frame_stack, *([1]*len(self.visual_obs[buffer_idx].shape[2:]))))
                if self.use_seg_obs:
                    first_seg_obs = self.seg_obs[buffer_idx][batch_indices, 0:1].repeat((1, self.frame_stack, *([1]*len(self.seg_obs[buffer_idx].shape[2:]))))

            obs_combined.append(obs.to(self.output_device))
            visual_obs_combined.append(visual_obs.to(self.output_device))
            actions_combined.append(actions.to(self.output_device))
            first_obs_combined.append(first_obs.to(self.output_device))
            first_visual_obs_combined.append(first_visual_obs.to(self.output_device))
            prev_obs_combined.append(prev_obs.to(self.output_device))
            if self.use_seg_obs:
                first_seg_obs_combined.append(first_seg_obs.to(self.output_device))
            
        obs = torch.cat(obs_combined, dim=0)
        visual_obs = torch.cat(visual_obs_combined, dim=0)
        actions = torch.cat(actions_combined, dim=0)
        first_obs = torch.cat(first_obs_combined, dim=0)
        first_visual_obs = torch.cat(first_visual_obs_combined, dim=0)
        prev_obs = torch.cat(prev_obs_combined, dim=0)
        if self.use_seg_obs:
            first_seg_obs = torch.cat(first_seg_obs_combined, dim=0)
        else:
            first_seg_obs = None
        
        # apply noise to visual obs
        visual_obs = self.visual_obs_handler.apply_noise(visual_obs)
        first_visual_obs = self.visual_obs_handler.apply_noise(first_visual_obs)
        first_seg_obs = self.seg_obs_handler.apply_noise(first_seg_obs)
        
        obs, visual_obs = get_student_obs(obs, visual_obs, first_obs, first_visual_obs, prev_obs, self.visual_obs_type, seg_frame0_obs=first_seg_obs)
        
        return obs.clone(), visual_obs.clone(), actions.clone()
    
    def save(self):
        # return a dict of all the data in the storage as cpu tensors
        d = {
            "obs": [o.cpu() for o in self.obs],
            "visual_obs": [v.cpu() for v in self.visual_obs],
            "actions": [a.cpu() for a in self.actions],
            "rewards": [r.cpu() for r in self.rewards],
            "dones": [d.cpu() for d in self.dones],
            "cur": self.cur,
            "step": self.step,
            "ep_step": self.ep_step,
            "valid_buffers": self.valid_buffers,
        }
        if self.use_seg_obs:
            d["seg_obs"] = [s.cpu() for s in self.seg_obs]
        return d
    
    def load(self, data):
        # load the data from the dict
        self.obs, self.visual_obs, self.seg_obs, self.dones, self.rewards, self.actions = [], [], [], [], [], []
        for device in self.storage_devices:
            cuda_device = f"cuda:{device}"
            self.obs.append(data["obs"][device].to(cuda_device))
            self.visual_obs.append(data["visual_obs"][device].to(cuda_device))
            self.dones.append(data["dones"][device].to(cuda_device))
            self.rewards.append(data["rewards"][device].to(cuda_device))
            self.actions.append(data["actions"][device].to(cuda_device))
            if self.use_seg_obs:
                self.seg_obs.append(data["seg_obs"][device].to(cuda_device))
        self.cur = data["cur"]
        self.step = data["step"]
        self.ep_step = data["ep_step"]
        self.valid_buffers = data["valid_buffers"]
        self.current_device_idx = 0
    
class Dagger(object):
    def __init__(self, config):
        # if config.object_list is a file
        if os.path.isfile(str(config.object_list)):
            object_list_file = config.object_list
            object_list = []
            with open(object_list_file, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        object_id, scale = parts
                        object_list.append((object_id, float(scale)))
                    else:
                        object_list.append(parts[0])
            config.object_list = object_list
        self.cfg = config
        self.cfg_dict = omegaconf_to_dict(config)
        self.device = self.cfg.device
        self.cfg.seed = set_seed(self.cfg.seed)

        self.visual_obs_handler = get_visual_obs_handler(self.cfg.dagger.visual_obs_type, self.cfg)
        self.seg_obs_handler = get_seg_obs_handler(self.cfg)

        # robomimic init
        robomimic_cfg = RMUtils.load_config(
            algo_cfg_path=config.dagger.student_cfg_path,
            override_cfg=config,
        )
        RMUtils.initialize(robomimic_cfg)
        self.seq_length = robomimic_cfg.train.seq_length
        self.frame_stack = robomimic_cfg.train.frame_stack

        # logging
        run_name = f"{self.cfg.wandb_run_name}"
        self.log_dir = os.path.join(self.cfg.train_dir, run_name)
        self.checkpoint_dir = os.path.join(self.log_dir, "nn")
        self.video_dir = os.path.join(self.log_dir, "videos")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # build environment & generate datasets
        if self.cfg.dagger.multitask:
            self.setup_rl_env_and_expert_multitask()
        else:
            self.setup_rl_env_and_expert()

        self.obs_handler = get_visual_obs_handler(self.cfg.dagger.visual_obs_type, self.cfg)
        self.setup_storage()
        # load obs-specific robomimic data
        obs_shape_meta = get_obs_shape_meta(self.cfg)

        # set up training
        self.setup_training(robomimic_cfg, obs_shape_meta)
        self.total_steps = 0
        self.total_episodes = 0
        self.total_epochs = 0
        self.log_history = {}

        # restore checkpoint
        if self.cfg.resume:
            self.load_checkpoint(self.cfg.resume)
        else:
            if not self.cfg.eval_mode:
                # have to fill the storage with data before training
                self.collect_data("eval")
                self.collect_data("train")

    def log(self, run, log_dict):
        if run is not None:
            expanded_log_dict = {}
            for k, v in log_dict.items():
                if k not in self.log_history:
                    self.log_history[k] = []
                self.log_history[k].append(v)
                expanded_log_dict.update({
                    k: v,
                    k + '/mean': np.mean(self.log_history[k]),
                    k + '/std': np.std(self.log_history[k]),
                    k + '/min': np.min(self.log_history[k]),
                    k + '/max': np.max(self.log_history[k]),
                })
            run.log(expanded_log_dict)

    def setup_rl_env_and_expert(self):
        """Set up the environment & expert model."""
        assert self.cfg.init_states != "", "Please specify path to initial states"
        init_states = torch.load(self.cfg.init_states)
        idx = torch.randperm(len(list(init_states.values())[0]))
        for key in init_states.keys():
            init_states[key] = init_states[key][idx]
        self.env = environments[self.cfg.task_name](
            self.cfg,
            init_states
        )
        self.env.disable_automatic_reset = True
        self.env.disable_hardcode_control = True

        if not self.cfg.eval_mode:
            self.expert_player = load_model(
                actions_num=self.env.num_actions,
                obs_shape=(self.env.num_obs,),
                device=self.cfg.device,
                checkpoint_path=self.cfg.checkpoint,
                rl_config=omegaconf_to_dict(self.cfg.train),
            )
            
    def setup_rl_env_and_expert_multitask(self):
        """Set up the environment & expert model."""
        assert self.cfg.init_states != "", "Please specify path to initial states"
        self.env = environments[self.cfg.task_name](
            self.cfg,
            self.cfg.init_states
        )
        self.env.disable_automatic_reset = True
        self.env.disable_hardcode_control = True
        self.setup_expert_multitask()
        
    def setup_expert_multitask(self):
        # assume self.cfg.checkpoint is now a directory of checkpoints
        # assume that the current env is the right multitask one
        start_time = time.time()
        for path in os.listdir(self.cfg.checkpoint):
            trimmed_path = path[:-len(".pth")]
            split_path = trimmed_path.split("_")
            if len(split_path) == 2:
                object_code, object_scale = split_path
                curr_scale = f"{int(100 * self.env.object_scale):03d}"
                curr_object_code = self.env.object_code.replace('/', '-')
                is_current_expert = object_code == curr_object_code and object_scale == curr_scale
            else:
                object_path = split_path[-1]
                object_code = object_path
                is_current_expert = object_code == self.env.object_code
            if is_current_expert:
                expert_player = load_model(
                    actions_num=self.env.num_actions,
                    obs_shape=(self.env.num_obs,),
                    device=self.cfg.device,
                    checkpoint_path=os.path.join(self.cfg.checkpoint, path),
                    rl_config=omegaconf_to_dict(self.cfg.train),
                )
                self.expert_player = expert_player
                break
        if not self.cfg.logging.suppress_timing:
            print(f"Loaded expert in {time.time() - start_time:.2f}s")
        
    def setup_storage(self):
        # get visual obs shape
        state_obs = self.env.compute_observations()
        visual_obs = self.get_visual_obs(prev_vis_obs=None)
        visual_obs_shape = visual_obs.shape[1:]

        # build eval storage: data collected with expert
        self.eval_storage = Storage(
            1, self.env.num_envs,
            obs_shape=(7,),
            visual_obs_shape=visual_obs_shape,
            visual_obs_handler=self.visual_obs_handler,
            actions_shape=self.env.action_space.shape, 
            visual_obs_type=self.cfg.dagger.visual_obs_type, 
            use_seg_obs=self.cfg.dagger.use_seg_obs,
            seg_obs_handler=self.seg_obs_handler,
            traj_length=self.env.max_episode_length,
            seq_length=self.seq_length,
            frame_stack=self.frame_stack,
            output_device=self.device,
        )
        # build training storage
        self.train_storage = Storage(
            self.cfg.dagger.buffer_size, self.env.num_envs, 
            obs_shape=(7,),
            visual_obs_shape=visual_obs_shape,
            visual_obs_handler=self.visual_obs_handler,
            actions_shape=self.env.action_space.shape, 
            visual_obs_type=self.cfg.dagger.visual_obs_type, 
            use_seg_obs=self.cfg.dagger.use_seg_obs,
            seg_obs_handler=self.seg_obs_handler,
            traj_length=self.env.max_episode_length,
            seq_length=self.seq_length,
            frame_stack=self.frame_stack,
            output_device=self.device,
            storage_devices=self.cfg.storage_devices,
        )
        
        
    def setup_training(self, robomimic_cfg, obs_shape_meta):
        """Set up student model and training parameters."""
        self.student_player = RMUtils.build_model(
            config=robomimic_cfg,
            shape_meta=obs_shape_meta,
            device=torch.device(self.cfg.rl_device),
        )

        # print student model details
        def format_parameters(num):
            if num < 1e6:
                return f"{num / 1e3:.2f}K"  # Thousands
            elif num < 1e9:
                return f"{num / 1e6:.2f}M"  # Millions
            elif num < 1e12:
                return f"{num / 1e9:.2f}G"  # Billions
            else:
                return f"{num / 1e12:.2f}T"  # Trillions

        # print model info
        print("\n============= Model Summary =============")
        print(self.student_player)  # print model summary
        num_policy_params = sum(p.numel() for p in self.student_player.nets['policy'].parameters())
        num_enc_params = sum(p.numel() for p in self.student_player.nets['policy'].nets['encoder'].parameters())
        print("Policy params:", format_parameters(num_policy_params))
        print("Encoder params:", format_parameters(num_enc_params))
        print("")

        # parameters
        self.num_learning_iterations = self.cfg.dagger.num_learning_iterations
        self.num_learning_epochs = self.cfg.dagger.num_learning_epochs
        self.num_transitions_per_iter = self.cfg.dagger.num_transitions_per_iter

    @property
    def storage(self):
        return self.train_storage

    def reset_envs(self):
        # TODO: support multitask dagger here (switch objects, reload policies, and recreate envs when necessary)
        env_ids = torch.arange(self.env.num_envs, device=self.env.device)
        if self.cfg.dagger.multitask: 
            self.env.reset_idx(env_ids, switch_object=True, init_states=self.cfg.init_states)
            self.setup_expert_multitask()
        self.env.reset_idx(env_ids)

    def get_visual_obs(self, prev_vis_obs, **kwargs):
        return self.visual_obs_handler.get_visual_obs(
            envs=self.env,
            prev_vis_obs=prev_vis_obs,
            device=self.device,
            **kwargs,
        )
    
    def get_seg_obs(self, **kwargs):
        if self.cfg.dagger.use_seg_obs:
            return self.env.get_segmentation_observations()
        else:
            return None
    
    @torch.inference_mode()
    def collect_data(self, split="train"):
        """Collect trajectories for evaluation with the expert."""
        state_obs = self.env.compute_observations()
        visual_obs = self.get_visual_obs(prev_vis_obs=None)
        seg_frame0_obs = self.get_seg_obs()

        if split == "train":
            storage = self.storage
        elif split == "eval":
            storage = self.eval_storage

        for iter_id in tqdm(range(storage.buffer_size*self.env.max_episode_length), desc=f"{split} data collection"):
            actions_expert = get_actions(
                {"obs": state_obs}, self.expert_player, 
                is_deterministic=self.cfg.dagger.deterministic_expert,
            )

            # take a step
            obs_dict, rews, dones, infos = self.env.step(actions_expert, get_camera_images=False)
            if (iter_id + 1) % self.env.max_episode_length == 0:
                dones[:] = True

            # update storage
            storage.add_transitions(state_obs, visual_obs, actions_expert, rews, dones, seg_obs=seg_frame0_obs)

            # update new obs
            state_obs = obs_dict["obs"]
            if dones.any():
                self.reset_envs()
                state_obs = self.env.compute_observations()
                visual_obs = None  # reset prev visual obs
                seg_frame0_obs = self.get_seg_obs()
            visual_obs = self.get_visual_obs(prev_vis_obs=visual_obs)

        self.reset_envs()

    def train(self):
        """Train the student policy using DAgger."""

        if self.cfg.wandb_activate:
            run = wandb.init(
                project=self.cfg.wandb_project,
                config=self.cfg_dict,
                sync_tensorboard=True,
                name=self.cfg.wandb_run_name,
                resume=True,
                dir=self.log_dir,
            )
        else:
            run = None

        print("Training DAgger...")
        state_obs = self.env.compute_observations()
        
        visual_obs = self.get_visual_obs(prev_vis_obs=None)
        seg_frame0_obs = self.get_seg_obs()
        state_frame0_obs, visual_frame0_obs = state_obs.clone(), visual_obs.clone()
        prev_state_obs = state_obs.clone()

        with tqdm(range(self.total_episodes, self.total_episodes + self.num_learning_iterations), desc='DAgger Training') as pbar:
            test_success = self.test(num_test_iterations=1) # just to prime the dict
            pbar.set_postfix(
                ep=self.total_episodes,
                mse=f"{0.0:.4f}",
                l1=f"{0.0:.4f}",
                gmm=f"{0.0:.4f}",
                test_success=f"{test_success['success']:.4f}",
            )
            for iter_id in pbar:
                wandb_log_dict = {}
                # rollout student
                t1 = time.time()
                with torch.inference_mode():
                    self.student_player.set_eval()
                    for _ in range(self.num_transitions_per_iter):
                        actions_expert = get_actions(
                            {"obs": state_obs}, self.expert_player, 
                            is_deterministic=self.cfg.dagger.deterministic_expert,
                        )

                        concat_state_obs, concat_visual_obs = get_student_obs(
                            state_obs[..., :7], 
                            self.visual_obs_handler.apply_noise(visual_obs), 
                            state_frame0_obs[..., :7], 
                            self.visual_obs_handler.apply_noise(visual_frame0_obs), 
                            prev_state_obs[..., :7],
                            self.cfg.dagger.visual_obs_type,
                            seg_frame0_obs=self.seg_obs_handler.apply_noise(seg_frame0_obs),
                        )
                        if self.frame_stack > 0:
                            if self.storage.ep_step == self.frame_stack:
                                state_obs_history = deque([concat_state_obs.clone().unsqueeze(1) for _ in range(self.frame_stack)], maxlen=self.frame_stack)
                                visual_obs_history = deque([concat_visual_obs.clone().unsqueeze(1) for _ in range(self.frame_stack)], maxlen=self.frame_stack)
                            else:
                                state_obs_history.append(concat_state_obs.clone().unsqueeze(1))
                                visual_obs_history.append(concat_visual_obs.clone().unsqueeze(1))
                            concat_state_obs = torch.cat(tuple(state_obs_history), dim=1)
                            concat_visual_obs = torch.cat(tuple(visual_obs_history), dim=1)
                        actions = get_rollout_action(
                            model=self.student_player,
                            state_obs=concat_state_obs.clone(),
                            visual_obs=concat_visual_obs.clone(),
                        )

                        # take a step
                        obs_dict, rews, dones, infos = self.env.step(actions, get_camera_images=False)
                        if (self.total_steps + 1) % self.env.max_episode_length == 0:
                            dones[:] = True

                        # update storage
                        self.storage.add_transitions(state_obs, visual_obs, actions_expert, rews, dones, seg_obs=seg_frame0_obs)

                        # update new obs
                        prev_state_obs = state_obs.clone()
                        state_obs = obs_dict["obs"].clone()
                        if dones.any():
                            self.total_episodes += 1
                            avg_mse_loss, avg_l1_loss, avg_gmm_loss = self.eval()
                            wandb_log_dict.update({
                                "distillation/eval_mse": avg_mse_loss,
                                "distillation/eval_l1": avg_l1_loss,
                                "distillation/eval_gmm": avg_gmm_loss,
                            })

                            if self.total_episodes % self.cfg.test_frequency == 0:
                                test_success = self.test(num_test_iterations=self.cfg.test_episodes, run=run)
                                self.save_checkpoint(f"checkpoint_{self.total_episodes}_success_{test_success['success']:.4f}.pth", save_storage=False)
                                for k in test_success:
                                    wandb_log_dict[f"distillation/test_{k}"] = test_success[k]

                            pbar.set_postfix(
                                ep=self.total_episodes,
                                mse=f"{avg_mse_loss:.4f}",
                                l1=f"{avg_l1_loss:.4f}",
                                gmm=f"{avg_gmm_loss:.4f}",
                                test_success=f"{test_success['success']:.4f}",
                            )

                            self.reset_envs()
                            self.student_player.reset()
                            state_obs = self.env.compute_observations()
                            prev_state_obs = state_obs.clone()
                            visual_obs = self.get_visual_obs(prev_vis_obs=None)
                            seg_frame0_obs = self.get_seg_obs()
                            state_frame0_obs, visual_frame0_obs = state_obs.clone(), visual_obs.clone()
                        else:
                            visual_obs = self.get_visual_obs(prev_vis_obs=visual_obs)

                        self.total_steps += 1
                t2 = time.time()
                # learning step
                t2 = time.time()
                hidden_state = RMUtils.get_hidden_state(self.student_player)
                avg_loss = self.update()
                t3 = time.time()
                wandb_log_dict["distillation/train_avg_loss"] = avg_loss
                RMUtils.set_hidden_state(self.student_player, hidden_state)

                # log to wandb
                if self.cfg.wandb_activate:
                    self.log(run, wandb_log_dict)

                if not self.cfg.logging.suppress_timing:
                    print(f"Running time: rollout: {t2 - t1:.2f}s, update: {t3 - t2:.2f}s")
                
                if iter_id % 10 == 0:
                    # save checkpoint every 10 epochs, its expensive to save the buffer (30s)
                    self.save_checkpoint(prefix='checkpoint_latest')
        self.save_checkpoint(prefix='checkpoint_latest')

    def update(self):
        model = self.student_player
        model.set_train()

        batch_indices = self.storage.mini_batch_generator(self.cfg.dagger.batch_size)
        num_batches = len(batch_indices)
        tot_loss = 0.0

        for epoch in range(self.num_learning_epochs):
            for indices in batch_indices:
                obs_batch, visual_batch, actions_expert_batch = self.storage.get_batch(indices)
                batch = {
                    "actions": actions_expert_batch,
                    "obs": {
                        "state": obs_batch,
                        "visual": self.visual_obs_handler.format_for_robomimic(visual_batch),
                    }
                }

                # process batch for training
                input_batch = model.process_batch_for_training(batch)
                input_batch = model.postprocess_batch_for_training(input_batch, obs_normalization_stats=None)

                # forward pass
                predictons = model._forward_training(input_batch)
                losses = model._compute_losses(predictons, input_batch)

                # backward pass
                model.optimizers["policy"].zero_grad()
                losses["action_loss"].backward()
                model.optimizers["policy"].step()

                tot_loss += losses["action_loss"].detach().item()

            model.on_epoch_end(self.total_epochs)
            self.total_epochs += 1

        tot_loss /= num_batches * self.num_learning_epochs
        return tot_loss

    @torch.inference_mode()
    def eval(self):
        model = self.student_player
        model.set_train()

        batch_indices = self.eval_storage.mini_batch_generator(self.cfg.dagger.batch_size)
        num_batches = len(batch_indices)
        mse, l1, gmm = 0.0, 0.0, 0.0
        for indices in batch_indices:
            obs_batch, visual_batch, actions_expert_batch = self.eval_storage.get_batch(indices)
            batch = {
                "actions": actions_expert_batch,
                "obs": {
                    "state": obs_batch,
                    "visual": self.visual_obs_handler.format_for_robomimic(visual_batch),
                }
            }

            # process batch for training
            input_batch = model.process_batch_for_training(batch)
            input_batch = model.postprocess_batch_for_training(input_batch, obs_normalization_stats=None)

            # forward and backward pass
            with torch.no_grad():
                predictions = model._forward_training(input_batch)
                losses = model._compute_losses(predictions, input_batch)

                if "l2_loss" in losses:             # not using GMM
                    mse += losses["l2_loss"].detach().item()
                    l1 += losses["l1_loss"].detach().item()
                else:                               # using GMM
                    gmm += losses["action_loss"].detach().item()

        mse /= num_batches
        l1 /= num_batches
        gmm /= num_batches

        return mse, l1, gmm

    @torch.inference_mode()
    def test(self, num_test_iterations=5, run=None):
        """Test the student policy."""
        self.env.disable_hardcode_control = False
        self.env.render_hardcode_control = True
        self.student_player.set_eval()

        num_success = Counter()
        total_runs_per_key = Counter()
        tik = time.time()

        total_runs = 0
        video_ims = []
        local_obs_ims = []
        with tqdm(total=num_test_iterations * self.env.max_episode_length, desc='Testing student policy') as pbar:
            for iter_id in range(num_test_iterations):
                self.reset_envs()
                self.student_player.reset()
                
                state_obs = self.env.compute_observations()
                visual_obs = self.get_visual_obs(prev_vis_obs=None)
                seg_frame0_obs = self.get_seg_obs()
                state_frame0_obs, visual_frame0_obs = state_obs.clone(), visual_obs.clone()
                prev_state_obs = state_obs.clone()

                for test_step in range(self.env.max_episode_length - 1):
                    concat_state_obs, concat_visual_obs = get_student_obs(
                        state_obs[..., :7], 
                        self.visual_obs_handler.apply_noise(visual_obs), 
                        state_frame0_obs[..., :7], 
                        self.visual_obs_handler.apply_noise(visual_frame0_obs), 
                        prev_state_obs[..., :7],
                        self.cfg.dagger.visual_obs_type,
                        seg_frame0_obs=self.seg_obs_handler.apply_noise(seg_frame0_obs),
                    )
                    if self.frame_stack > 0:
                        if test_step == 0:
                            state_obs_history = deque([concat_state_obs.clone().unsqueeze(1) for _ in range(self.frame_stack)], maxlen=self.frame_stack)
                            visual_obs_history = deque([concat_visual_obs.clone().unsqueeze(1) for _ in range(self.frame_stack)], maxlen=self.frame_stack)
                        else:
                            state_obs_history.append(concat_state_obs.clone().unsqueeze(1))
                            visual_obs_history.append(concat_visual_obs.clone().unsqueeze(1))
                        concat_state_obs = torch.cat(tuple(state_obs_history), dim=1)
                        concat_visual_obs = torch.cat(tuple(visual_obs_history), dim=1)
                    actions = get_rollout_action(
                        model=self.student_player,
                        state_obs=concat_state_obs.clone(),
                        visual_obs=concat_visual_obs.clone(),
                    )

                    obs_dict, rews, dones, infos = self.env.step(actions, get_camera_images=False)
                    prev_state_obs = state_obs.clone()
                    state_obs = obs_dict["obs"].clone()
                    visual_obs = self.get_visual_obs(prev_vis_obs=visual_obs)

                    if self.env.capture_video:
                        if "hardcode_images" not in infos or len(infos["hardcode_images"]) == 0:
                            ims = np.array(camera_shot(
                                self.env, env_ids=range(self.env.capture_envs), camera_ids=[0]
                            )[0])[:, 0, :, :, :3]
                            video_ims.append(ims)
                        else:
                            ims = np.array(infos["hardcode_images"])[:, :, 0, :, :, :3]
                            ims = [ims[i] for i in range(ims.shape[0]) if i % 3 == 0]
                            video_ims.extend(ims)

                    if self.cfg.capture_local_obs:
                        if self.cfg.dagger.visual_obs_type == "depth":
                            depth = self.visual_obs_handler.apply_noise(visual_obs)[0, 0].cpu().numpy()
                            depth *= -1.0       # flip depth for visualization
                            depth = vis_depth(depth)
                            if self.cfg.dagger.use_seg_obs and test_step < 10:   # include segmentation mask in the first few video frames
                                mask = self.seg_obs_handler.apply_noise(seg_frame0_obs)[0, 0].cpu().numpy()
                                depth = apply_mask(depth, mask)
                            local_obs_ims.append(depth)
                    
                    pbar.update()

                for k in infos:
                    if k.endswith('success'):
                        num_success[k] += infos[k].sum().item()
                        total_runs_per_key[k] += self.env.num_envs
                total_runs += self.env.num_envs

        self.env.disable_hardcode_control = True
        self.env.render_hardcode_control = False

        if self.env.capture_video:
            ims = []
            for env_idx in range(self.env.capture_envs):
                for im in video_ims:
                    ims.append(im[env_idx])
            make_video(ims, self.video_dir, epoch=self.total_epochs)
            # log video to wandb:
            if self.cfg.wandb_activate and run is not None:
                run.log({"visualization/video": wandb.Video(os.path.join(self.video_dir, f"viz_{self.total_epochs}.mp4"))}, commit=False)

        if self.cfg.capture_local_obs:
            make_video(local_obs_ims, self.video_dir, epoch=self.total_epochs, name=f"viz_local_obs_{self.total_epochs}.mp4")
            # log video to wandb:
            if self.cfg.wandb_activate and run is not None:
                run.log({"visualization/local_obs": wandb.Video(os.path.join(self.video_dir, f"viz_local_obs_{self.total_epochs}.mp4"))}, commit=False)

        if not self.cfg.logging.suppress_timing:
            print(f"Finished testing in {time.time() - tik:.2f}s")

        return {k: v / total_runs_per_key[k] for k, v in num_success.items()}


    def save_checkpoint(self, prefix='checkpoint_latest', save_storage=True):
        start_time = time.time()
        distillation_ckpt_path = os.path.join(self.checkpoint_dir, f"{prefix}.pth")
        checkpoint = {
            "student_state_dict": self.student_player.serialize(),
            "total_steps": self.total_steps,
            "total_episodes": self.total_episodes,
            "total_epochs": self.total_epochs,
        }
        if save_storage:
            checkpoint["train_storage"] = self.train_storage.save()
            checkpoint["eval_storage"] = self.eval_storage.save()
        torch.save(
            checkpoint,
            distillation_ckpt_path,
        )
        if not self.cfg.logging.suppress_timing:
            print("Time to save checkpoint:", time.time() - start_time, "s")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.student_player.deserialize(checkpoint["student_state_dict"])
        if 'train_storage' in checkpoint:
            self.train_storage.load(checkpoint["train_storage"])
            print("Loaded storage of size:", self.train_storage.step, self.train_storage.valid_buffers, self.train_storage.cur)
        if 'eval_storage' in checkpoint:
            self.eval_storage.load(checkpoint["eval_storage"])
        self.total_steps = checkpoint["total_steps"]
        self.total_episodes = checkpoint["total_episodes"]
        self.total_epochs = checkpoint["total_epochs"]


@hydra.main(version_base="1.1", config_path="./config", config_name="config_dagger")
def main(cfg: DictConfig):
    import torch
    
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("medium")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    agent = Dagger(cfg)
    def handle(signum, frame):
        print("Signal handler called with signal", signum)
        print("Saving checkpoint before exiting...")
        agent.save_checkpoint()
        exit()
    signal.signal(signal.SIGUSR1, handle)
    if cfg.eval_mode:
        if cfg.export_rigid_body_poses_dir:             # export rigid body poses for rendering
            if os.path.exists(cfg.export_rigid_body_poses_dir):
                shutil.rmtree(cfg.export_rigid_body_poses_dir)
            agent.env.export_rigid_body_poses_dir = cfg.export_rigid_body_poses_dir
        test_success = agent.test(num_test_iterations=cfg.test_episodes)
        print(f"Test success: {test_success['success']:.4f}")

        if cfg.export_rigid_body_poses_dir:
            meta_data = {"task": cfg.task_name}
            if cfg.task_name in ("pick", "place"):
                meta_data.update({
                    "object_code": agent.env.object_code,
                    "object_scale": agent.env.object_scale,
                    "clutter": agent.env.clutter_object_codes_and_scales,
                })
            elif cfg.task_name in ("grasp_handle", "open", "close", "open_nograsp", "close_nograsp"):
                meta_data.update({
                    "object_code": agent.env.object_code,
                })
            np.save(os.path.join(cfg.export_rigid_body_poses_dir, "meta_data.npy"), meta_data)
    else:
        agent.train()


if __name__ == "__main__":
    import torch._dynamo
    torch._dynamo.config.disable = True

    main()
