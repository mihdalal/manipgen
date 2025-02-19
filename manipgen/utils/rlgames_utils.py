import os
import gym
import time
import yaml
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import wandb

from rl_games.common.algo_observer import AlgoObserver
from rl_games.algos_torch.a2c_continuous import A2CAgent
from rl_games.common.a2c_common import swap_and_flatten01, print_statistics
from rl_games.algos_torch import model_builder, torch_ext
from isaacgymenvs.utils.utils import flatten_dict, retry
from isaacgymenvs.utils.rlgames_utils import RLGPUAlgoObserver
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from manipgen.utils.media_utils import make_mp4_from_numpy

class PSLAlgoObserver(RLGPUAlgoObserver):
    def __init__(self):
        super().__init__()
        self.consecutive_successes = -1.0

    def process_infos(self, infos, done_indices):
        super().process_infos(infos, done_indices)

        if "consecutive_successes" in infos:
            self.consecutive_successes = infos["consecutive_successes"].mean().cpu().numpy()

    def after_print_stats(self, frame, epoch_num, total_time):
        super().after_print_stats(frame, epoch_num, total_time)

        if self.consecutive_successes != -1.0:
            self.writer.add_scalar("episode_success/consecutive_successes", self.consecutive_successes, frame)
            self.consecutive_successes = -1.0


class WandbAlgoObserver(AlgoObserver):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.wandb_project = cfg.wandb_project
        self.wandb_entity = None
        self.wandb_group = None
        self.wandb_tags = []
        self.wandb_logcode_dir = None

        # visualization
        self.images = []
        self.video_processes = []

    def before_init(self, base_name, config, experiment_name):
        """
        Must call initialization of Wandb before RL-games summary writer is initialized, otherwise
        sync_tensorboard does not work.
        """
        self.experiment_name = experiment_name
        datetime_str = datetime.now().strftime("%Y%m%d-%H-%M-%S")
        wandb_unique_id = f"uid_{experiment_name}_{datetime_str}"
        print(f"Wandb using unique id {wandb_unique_id}")

        # this can fail occasionally, so we try a couple more times
        @retry(3, exceptions=(Exception,))
        def init_wandb():
            wandb.init(
                project=self.wandb_project,
                entity=self.wandb_entity,
                group=self.wandb_group,
                tags=self.wandb_tags,
                sync_tensorboard=True,
                id=wandb_unique_id,
                name=experiment_name,
                resume=True,
                settings=wandb.Settings(start_method="fork"),
            )

            if self.wandb_logcode_dir:
                wandb.run.log_code(root=self.wandb_logcode_dir)
                print("wandb running directory........", wandb.run.dir)

        print("Initializing WandB...")
        try:
            init_wandb()
        except Exception as exc:
            print(f"Could not initialize WandB! {exc}")

        if isinstance(self.cfg, dict):
            wandb.config.update(self.cfg, allow_val_change=True)
        else:
            wandb.config.update(omegaconf_to_dict(self.cfg), allow_val_change=True)

    def process_infos(self, infos, done_indices):
        total_steps = infos["total_steps"]
        if (
            not self.cfg.capture_video
            or total_steps % self.cfg.capture_interval >= self.cfg.capture_length
        ):
            return

        images = infos["camera_images"]
        assert images is not None
        self.images.append(images)
        num_videos = len(self.images[0])

        if (
            not self.cfg.capture_video
            or total_steps % self.cfg.capture_interval == self.cfg.capture_length - 1
        ):
            self.images = [
                [
                    env_images[0][100:700, 320:1120, :].transpose(2, 1, 0)
                    for env_images in step_images
                ]
                for step_images in self.images
            ]
            self.images = [
                [step_images[env_id] for step_images in self.images]
                for env_id in range(num_videos)
            ]

            # for env_id in range(num_videos):
            #     frames = np.array(self.images[env_id])
            #     frames = frames[::6]
            #     wandb.log(
            #         {
            #             f"{self.experiment_name}_step{total_steps + 1}": wandb.Video(
            #                 frames, fps=10, format="gif"
            #             )
            #         }
            #     )
            # combine into one video in sequence of environments
            for env_id in range(num_videos):
                frames = []
                for im in self.images[env_id]:
                    frames.append(im.transpose(2, 1, 0)[:, :, :3])
                make_mp4_from_numpy(
                    "/tmp",
                    frames,
                    f"{self.experiment_name}_step{total_steps + 1}",
                )
                wandb.log({
                    f"{self.experiment_name}_step{total_steps + 1}": wandb.Video(
                        f"/tmp/{self.experiment_name}_step{total_steps + 1}.mp4", fps=20
                    )
                })

            self.images = []


# ===============================================================================


class A2CAgent_ManipGen(A2CAgent):
    """Original rlgames implementation select the best checkpoint based on reward,
    but we want to select based on success rate.
    """

    def __init__(self, base_name, params):
        super().__init__(base_name, params)
        self.best_mean_successes = -1.0
        self.best_epoch = -1
        self.game_successes = None
        self.early_stop = self.config.get("early_stop", -1)

    def play_steps(self):
        update_list = self.update_list

        step_time = 0.0

        for n in range(self.horizon_length):
            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)
            self.experience_buffer.update_data("obses", n, self.obs["obs"])
            self.experience_buffer.update_data("dones", n, self.dones)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k])
            if self.has_central_value:
                self.experience_buffer.update_data("states", n, self.obs["states"])

            step_time_start = time.time()
            self.obs, rewards, self.dones, infos = self.env_step(res_dict["actions"])
            step_time_end = time.time()

            step_time += step_time_end - step_time_start

            shaped_rewards = self.rewards_shaper(rewards)
            if self.value_bootstrap and "time_outs" in infos:
                shaped_rewards += (
                    self.gamma
                    * res_dict["values"]
                    * self.cast_obs(infos["time_outs"]).unsqueeze(1).float()
                )

            self.experience_buffer.update_data("rewards", n, shaped_rewards)

            # add the following 2 lines to update the success rate
            if "consecutive_successes" in infos:
                self.game_successes = infos["consecutive_successes"]

            self.current_rewards += rewards
            self.current_shaped_rewards += shaped_rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            env_done_indices = all_done_indices[:: self.num_agents]

            self.game_rewards.update(self.current_rewards[env_done_indices])
            self.game_shaped_rewards.update(
                self.current_shaped_rewards[env_done_indices]
            )
            self.game_lengths.update(self.current_lengths[env_done_indices])
            self.algo_observer.process_infos(infos, env_done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_shaped_rewards = (
                self.current_shaped_rewards * not_dones.unsqueeze(1)
            )
            self.current_lengths = self.current_lengths * not_dones

        last_values = self.get_values(self.obs)

        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict["dones"].float()
        mb_values = self.experience_buffer.tensor_dict["values"]
        mb_rewards = self.experience_buffer.tensor_dict["rewards"]
        mb_advs = self.discount_values(
            fdones, last_values, mb_fdones, mb_values, mb_rewards
        )
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(
            swap_and_flatten01, self.tensor_list
        )
        batch_dict["returns"] = swap_and_flatten01(mb_returns)
        batch_dict["played_frames"] = self.batch_size
        batch_dict["step_time"] = step_time

        return batch_dict

    def train(self):
        self.init_tensors()
        self.last_mean_rewards = -100500
        start_time = time.time()
        total_time = 0
        rep_count = 0
        self.obs = self.env_reset()
        self.curr_frames = self.batch_size_envs

        if self.multi_gpu:
            print("====================broadcasting parameters")
            model_params = [self.model.state_dict()]
            dist.broadcast_object_list(model_params, 0)
            self.model.load_state_dict(model_params[0])

        while True:
            epoch_num = self.update_epoch()
            (
                step_time,
                play_time,
                update_time,
                sum_time,
                a_losses,
                c_losses,
                b_losses,
                entropies,
                kls,
                last_lr,
                lr_mul,
            ) = self.train_epoch()
            total_time += sum_time
            frame = self.frame // self.num_agents

            # cleaning memory to optimize space
            self.dataset.update_values_dict(None)
            should_exit = False

            if self.global_rank == 0:
                self.diagnostics.epoch(self, current_epoch=epoch_num)
                # do we need scaled_time?
                scaled_time = self.num_agents * sum_time
                scaled_play_time = self.num_agents * play_time
                curr_frames = (
                    self.curr_frames * self.world_size
                    if self.multi_gpu
                    else self.curr_frames
                )
                self.frame += curr_frames

                print_statistics(
                    self.print_stats,
                    curr_frames,
                    step_time,
                    scaled_play_time,
                    scaled_time,
                    epoch_num,
                    self.max_epochs,
                    frame,
                    self.max_frames,
                )

                self.write_stats(
                    total_time,
                    epoch_num,
                    step_time,
                    play_time,
                    update_time,
                    a_losses,
                    c_losses,
                    entropies,
                    kls,
                    last_lr,
                    lr_mul,
                    frame,
                    scaled_time,
                    scaled_play_time,
                    curr_frames,
                )

                if len(b_losses) > 0:
                    self.writer.add_scalar(
                        "losses/bounds_loss",
                        torch_ext.mean_list(b_losses).item(),
                        frame,
                    )

                if self.has_soft_aug:
                    self.writer.add_scalar(
                        "losses/aug_loss", np.mean(aug_losses), frame
                    )

                if self.game_rewards.current_size > 0:
                    mean_rewards = self.game_rewards.get_mean()
                    mean_shaped_rewards = self.game_shaped_rewards.get_mean()
                    mean_lengths = self.game_lengths.get_mean()
                    self.mean_rewards = mean_rewards[0]
                    mean_successes = None
                    try:
                        mean_successes = self.game_successes.mean().cpu().numpy()
                    except:
                        pass

                    for i in range(self.value_size):
                        rewards_name = "rewards" if i == 0 else "rewards{0}".format(i)
                        self.writer.add_scalar(
                            rewards_name + "/step".format(i), mean_rewards[i], frame
                        )
                        self.writer.add_scalar(
                            rewards_name + "/iter".format(i), mean_rewards[i], epoch_num
                        )
                        self.writer.add_scalar(
                            rewards_name + "/time".format(i),
                            mean_rewards[i],
                            total_time,
                        )
                        self.writer.add_scalar(
                            "shaped_" + rewards_name + "/step".format(i),
                            mean_shaped_rewards[i],
                            frame,
                        )
                        self.writer.add_scalar(
                            "shaped_" + rewards_name + "/iter".format(i),
                            mean_shaped_rewards[i],
                            epoch_num,
                        )
                        self.writer.add_scalar(
                            "shaped_" + rewards_name + "/time".format(i),
                            mean_shaped_rewards[i],
                            total_time,
                        )

                    self.writer.add_scalar("episode_lengths/step", mean_lengths, frame)
                    self.writer.add_scalar(
                        "episode_lengths/iter", mean_lengths, epoch_num
                    )
                    self.writer.add_scalar(
                        "episode_lengths/time", mean_lengths, total_time
                    )

                    if self.has_self_play_config:
                        self.self_play_manager.update(self)

                    if mean_successes is not None:
                        if (
                            mean_successes > self.best_mean_successes
                            and epoch_num >= self.save_best_after
                        ):
                            print("saving next best successes: ", mean_successes)
                            self.best_mean_successes = mean_successes
                            self.best_epoch = epoch_num
                            self.save(os.path.join(self.nn_dir, self.config["name"]))

                    # if mean_rewards[0] > self.last_mean_rewards and epoch_num >= self.save_best_after:
                    #     print('saving next best rewards: ', mean_rewards)
                    #     self.last_mean_rewards = mean_rewards[0]
                    #     self.save(os.path.join(self.nn_dir, self.config['name']))

                    #     if 'score_to_win' in self.config:
                    #         if self.last_mean_rewards > self.config['score_to_win']:
                    #             print('Maximum reward achieved. Network won!')
                    #             self.save(os.path.join(self.nn_dir, checkpoint_name))
                    #             should_exit = True

                if epoch_num >= self.max_epochs and self.max_epochs != -1:
                    if self.game_rewards.current_size == 0:
                        print(
                            "WARNING: Max epochs reached before any env terminated at least once"
                        )
                        mean_rewards = -np.inf

                    self.save(
                        os.path.join(
                            self.nn_dir,
                            "last_checkpoint",
                        )
                    )
                    # self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_ep_' + str(epoch_num) \
                    #     + '_rew_' + str(mean_rewards).replace('[', '_').replace(']', '_')))
                    print("MAX EPOCHS NUM!")
                    should_exit = True

                if (
                    self.best_epoch != -1
                    and self.early_stop != -1
                    and epoch_num - self.best_epoch > self.early_stop
                ):
                    self.save(
                        os.path.join(
                            self.nn_dir,
                            "last_" + self.config["name"] + "_ep_" + str(epoch_num),
                        )
                    )
                    print("EARLY STOP!")
                    should_exit = True

                if self.frame >= self.max_frames and self.max_frames != -1:
                    if self.game_rewards.current_size == 0:
                        print(
                            "WARNING: Max frames reached before any env terminated at least once"
                        )
                        mean_rewards = -np.inf

                    self.save(
                        os.path.join(
                            self.nn_dir,
                            "last_"
                            + self.config["name"]
                            + "_frame_"
                            + str(self.frame)
                            + "_rew_"
                            + str(mean_rewards).replace("[", "_").replace("]", "_"),
                        )
                    )
                    print("MAX FRAMES NUM!")
                    should_exit = True

                update_time = 0

            if self.multi_gpu:
                should_exit_t = torch.tensor(should_exit, device=self.device).float()
                dist.broadcast(should_exit_t, 0)
                should_exit = should_exit_t.float().item()
            if should_exit:
                return self.last_mean_rewards, epoch_num

    def set_full_state_weights(self, weights, set_epoch=True):
        super().set_full_state_weights(weights, set_epoch)
        self.best_mean_successes = weights["best_mean_successes"]
        self.best_epoch = weights["best_epoch"]

    def get_full_state_weights(self):
        state = super().get_full_state_weights()
        state["best_mean_successes"] = self.best_mean_successes
        state["best_epoch"] = self.best_epoch
        return state

# ====================== External use of RL Games Models ======================


def load_model(
    actions_num,
    obs_shape,
    device,
    checkpoint_path,
    config_path="",
    rl_config=None,
):
    """
    Args:
        actions_num: number of actions, int
        obs_shape: shape of the observation, tuple
        device: device to run the model
        config_path: path to the config file
        checkpoint_path: path to the checkpoint
        rl_config: config of the model
    Returns:
        model: policy model
    """

    if rl_config is None:
        with open(config_path, "r") as file:
            rl_config = yaml.safe_load(file)

    builder = model_builder.ModelBuilder()
    config_network = builder.load(rl_config["params"])
    network = config_network
    config = {
        "actions_num": actions_num,
        "input_shape": obs_shape,
        "num_seqs": 1,
        "value_size": 1,
        "normalize_value": rl_config["params"]["config"]["normalize_value"],
        "normalize_input": rl_config["params"]["config"]["normalize_input"],
    }

    model = network.build(config)
    model.to(device)
    model.eval()

    assert checkpoint_path != ""
    # checkpoint = torch_ext.load_checkpoint(checkpoint_path)
    print("=> loading checkpoint '{}'".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    if "running_mean_std" in checkpoint:
        model.running_mean_std.load_state_dict(checkpoint["running_mean_std"])

    return model


def get_actions(obs, model, is_deterministic=True):
    obs = obs["obs"]
    input_dict = {
        "is_train": False,
        "prev_actions": None,
        "obs": obs,
        "rnn_states": None,
    }

    with torch.no_grad():
        res_dict = model(input_dict)

    mu = res_dict["mus"]
    action = res_dict["actions"]
    if is_deterministic:
        current_action = mu
    else:
        current_action = action

    return torch.clamp(current_action, -1.0, 1.0)
