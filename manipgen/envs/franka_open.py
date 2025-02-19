from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgymenvs.utils.torch_jit_utils import (
    to_torch,
    quat_from_euler_xyz,
    quat_mul,
    quat_rotate,
    quat_conjugate,
    quat_from_angle_axis
)

from gym import spaces
import numpy as np
import torch
import trimesh as tm

from typing import Dict, Any, Tuple, List, Set
import math
import random
import os
import yaml

from manipgen.envs.franka_articulated_env import FrankaArticulatedEnv
from manipgen.envs.franka_grasp_handle import FrankaGraspHandle
import isaacgymenvs.tasks.factory.factory_control as fc

class FrankaOpen(FrankaArticulatedEnv):
    def __init__(
        self,
        cfg,
        init_states=None,
        **kwargs,
    ):
        """
        Args:
            cfg: config dictionary for the environment.
            init_states: pre-sampled initial states for reset.
        """

        self._get_env_yaml_params(cfg)
        self._get_task_yaml_params(cfg)

        super().__init__(cfg, **kwargs)
        # max episode length in policy learning
        self.max_episode_length = self.cfg["env"]["episode_length"]
     
        # prepare domain randomization
        self.randomize = False
        if self.randomize:
            self.prepare_randomization()
        self.init_states_cache = {}
        
        # prepare tensors
        self.states = {}
        self.obs_dict = {}
        self._acquire_env_tensors()
        self.refresh_env_tensors()
        self._acquire_task_tensors()
        self.parse_controller_spec()

        # set up viewer and camera sensors
        self.create_viewer_and_camera()

        # reward settings
        self.reward_settings = {
            "dist_object_gripper_reward_temp": self.cfg["rl"]["dist_object_gripper_reward_temp"],
            "dist_object_dof_pos_reward_temp": self.cfg["rl"]["dist_object_dof_pos_reward_temp"],
            "grasp_pos_offset_reward_temp": self.cfg["rl"]["grasp_pos_offset_reward_temp"],
            "gripper_direction_reward_temp": self.cfg["rl"]["gripper_direction_reward_temp"],
            "gripper_direction_threshold": self.cfg["rl"]["gripper_direction_threshold"],
            "action_penalty_temp": self.cfg["rl"]["action_penalty_temp"],
            "action_penalty_threshold": self.cfg["rl"]["action_penalty_threshold"],
            "success_threshold": self.cfg["rl"]["success_threshold"],
        }

        # Reset all environments
        self.object_id = -1
        self.reset_idx(
            torch.arange(self.num_envs, device=self.device),
            switch_object=True,
            init_states=init_states,
        )

    create_envs = FrankaGraspHandle.create_envs
    _create_actors = FrankaGraspHandle._create_actors
    _compute_nearest_grasp_pose_keypoint_distance = FrankaGraspHandle._compute_nearest_grasp_pose_keypoint_distance
    _compute_keypoints_pos = FrankaGraspHandle._compute_keypoints_pos
    switch_object = FrankaGraspHandle.switch_object
    _sample_object_pose = FrankaGraspHandle._sample_object_pose
    _reset_franka_and_object = FrankaGraspHandle._reset_franka_and_object

    def _acquire_task_tensors(self):
        """Acquire and wrap tensors. Create views."""

        # object state tensors are created in switch_object
        # because we need to know object id in multitask mode

        # fingertip pos
        self.left_fingertip_pos = self.body_pos[:, self.left_fingertip_body_id_env, 0:3]
        self.right_fingertip_pos = self.body_pos[:, self.right_fingertip_body_id_env, 0:3]

        # reward
        self.num_object_keypoints = self.cfg_env.rl.num_object_keypoints
        self.object_keypoints = torch.zeros(self.num_envs, self.num_object_keypoints * 3, device=self.device)
        self.dist_object_dof_pos = torch.zeros(self.num_envs, device=self.device)
        self.dist_object_gripper = torch.zeros(self.num_envs, device=self.device)
        self.dist_init_grasp_pos = torch.zeros(self.num_envs, device=self.device)
        self.up_v = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)
        self.gripper_direction_local = torch.zeros(self.num_envs, 3, device=self.device)
        self.prev_object_dof_pos = torch.zeros(self.num_envs, self.object_num_dofs, device=self.device)
        self.object_dof_diff = torch.zeros(self.num_envs, device=self.device)
        self.prev_eef_height = torch.zeros(self.num_envs, device=self.device)
        self.eef_height = torch.zeros(self.num_envs, device=self.device)
        self.eef_height_diff = torch.zeros(self.num_envs, device=self.device)
        self.action_handle_frame = torch.zeros(self.num_envs, 3, device=self.device)
        self.actions = torch.zeros((self.num_envs, self.cfg_task.env.numActions), device=self.device)

        # metric for open / close
        self.relative_dof_completeness = torch.zeros(self.num_envs, device=self.device)

        # override in reset
        self.init_object_dof_pos = torch.zeros(self.num_envs, self.object_num_dofs, device=self.device)
        self.init_fingertip_centered_pos_local = torch.zeros(self.num_envs, 3, device=self.device)
        self.init_eef_height = torch.zeros(self.num_envs, device=self.device)

        self.prev_fingertip_centered_pos_local = torch.zeros(self.num_envs, 3, device=self.device)

    def _refresh_task_tensors(self):
        """Refresh tensors."""
        self.object_keypoints = self._compute_keypoints_pos()

        if self.sample_mode:
            self.dist_object_dof_pos = torch.abs(self.object_dof_pos - self.init_object_dof_pos).mean(dim=-1)
        else:
            self.dist_object_dof_pos = (self.object_dof_pos - self.target_object_dof_pos.reshape(-1, self.object_num_dofs)).mean(dim=-1)
        self.dist_object_gripper = self._compute_nearest_grasp_pose_keypoint_distance()
        # convert current fingertip center pos to handle frame
        self.fingertip_centered_pos_local = quat_rotate(
            quat_conjugate(self.handle_quat), self.fingertip_centered_pos - self.handle_pos
        )
        self.grasp_pos_offset = self.fingertip_centered_pos_local - self.prev_fingertip_centered_pos_local
        self.dist_init_grasp_pos = torch.norm(self.fingertip_centered_pos_local - self.init_fingertip_centered_pos_local, dim=-1)
        self.prev_fingertip_centered_pos_local = self.fingertip_centered_pos_local.clone()

        gripper_direction = quat_rotate(self.fingertip_centered_quat, self.up_v)
        self.gripper_direction_local = quat_rotate(quat_conjugate(self.handle_quat), gripper_direction)

        self.eef_height = self.fingertip_centered_pos[:, 2]
        self.eef_height_diff = (self.eef_height - self.prev_eef_height).abs()
        self.prev_eef_height[:] = self.eef_height.clone()

        self.object_dof_diff = (self.object_dof_pos - self.prev_object_dof_pos).abs().mean(dim=-1) \
            / torch.abs(self.target_object_dof_pos - self.rest_object_dof_pos).mean(dim=-1)
        self.prev_object_dof_pos[:] = self.object_dof_pos.clone()

        self.relative_dof_completeness = ((self.object_dof_pos - self.init_object_dof_pos) \
            / (self.target_object_dof_pos.reshape(-1, self.object_num_dofs) - self.init_object_dof_pos)).mean(dim=-1)
        
        self.action_handle_frame = quat_rotate(quat_conjugate(self.handle_quat), self.actions[:, :3])

    def compute_reward(self):
        """Compute rewards."""
        dist_object_gripper_reward_temp = self.reward_settings["dist_object_gripper_reward_temp"]
        dist_object_dof_pos_reward_temp = self.reward_settings["dist_object_dof_pos_reward_temp"]
        grasp_pos_offset_reward_temp = self.reward_settings["grasp_pos_offset_reward_temp"]
        gripper_direction_reward_temp = self.reward_settings["gripper_direction_reward_temp"]
        gripper_direction_threshold = self.reward_settings["gripper_direction_threshold"]
        action_penalty_temp = self.reward_settings["action_penalty_temp"]
        action_penalty_threshold = self.reward_settings["action_penalty_threshold"]
        success_threshold = self.reward_settings["success_threshold"]

        # component 1: keypoint distance between current gripper pose and nearest pre-sampled grasp pose
        # encourage the gripper to be close to the pre-sampled grasp pose
        dist_gripper_reward = (
            torch.exp(dist_object_gripper_reward_temp * self.dist_object_gripper)
            + torch.exp(dist_object_gripper_reward_temp * self.dist_object_gripper * 10)
            + torch.exp(dist_object_gripper_reward_temp * self.dist_object_gripper * 100)
            + torch.exp(dist_object_gripper_reward_temp * self.dist_object_gripper * 1000)
        ) / 4

        # component 2: change in eef pos in handle frame
        # we want the gripper to keep the relative positioning with the handle
        grasp_pos_offset_reward = grasp_pos_offset_reward_temp * self.grasp_pos_offset.norm(dim=-1)

        # component 3: how close the object dof pos is to the target object dof pos
        # encourage opening the door
        dof_ratio = self.dist_object_dof_pos.abs() / torch.abs(self.target_object_dof_pos - self.rest_object_dof_pos).mean(dim=-1)
        dist_dof_reward = torch.exp(dist_object_dof_pos_reward_temp * dof_ratio)

        # component 4: angle between gripper and door
        # encourage the gripper to be perpendicular to the door (with some threshold)
        gripper_direction_reward = torch.exp(
            gripper_direction_reward_temp * torch.clamp(-self.gripper_direction_local[:, 0] - gripper_direction_threshold, max=0.0)
        )

        # component 5: penalize action in unnecessary directions (y and z in handle frame) for better sim2real transfer
        action_mag_yz = torch.norm(self.action_handle_frame[:, 1:], dim=-1)
        action_penalty = torch.clamp(action_mag_yz - action_penalty_threshold, min=0.0) * action_penalty_temp

        total_reward = dist_gripper_reward * gripper_direction_reward * dist_dof_reward + grasp_pos_offset_reward + action_penalty

        self.rew_buf[:] = total_reward
        reset = (self.progress_buf == self.max_episode_length - 1)

        is_last_step = (self.progress_buf[0] == self.max_episode_length - 1)
        if is_last_step:
            success = dof_ratio < success_threshold
            grasping = (self.gripper_dof_pos[:, :].sum(dim=-1) > 0.001).float()
            self.extras["dof_completeness"] = 1.0 - dof_ratio.mean()
            self.extras["relative_dof_completeness"] = self.relative_dof_completeness.mean()
            self.extras["keypoint_dist"] = self.dist_object_gripper.mean()
            self.extras["grasping_ratio"] = grasping.mean()
            self.extras["grasping_dist"] = self.dist_init_grasp_pos.mean()
            self.extras["action_mag_yz"] = action_mag_yz.mean()

            self.success_buf[:] = success
            self.consecutive_successes[:] = torch.where(
                reset > 0, torch.clamp(self.relative_dof_completeness, max=1.0) * reset, self.consecutive_successes
            ).mean()

        if not self.disable_automatic_reset:
            self.reset_buf[:] = reset

    def compute_observations(self):
        """Compute observations."""

        self.global_eef_pos = self.fingertip_centered_pos       # position of the gripper
        self.global_eef_quat = self.fingertip_centered_quat     # orientation of the gripper

        local_eef_pos, local_eef_quat = self.pose_world_to_robot_base(
            self.global_eef_pos,    
            self.global_eef_quat, 
        )
        local_eef_quat = self._format_quaternion(local_eef_quat)

        obs_tensors = [
            local_eef_pos,
            local_eef_quat,
            self.fingertip_centered_linvel,                 # linear velocity of the gripper
            self.fingertip_centered_angvel,                 # angular velocity of the gripper
            self.handle_pos,                                # position of the handle
            self.handle_quat,                               # orientation of the handle
            self.object_dof_pos,                            # dof pos of the object
            self.object_dof_vel,                            # dof vel of the object
            self.object_keypoints,                          # position of keypoints of the object
            self.dist_object_dof_pos.unsqueeze(-1),         # distance between current object dof pos and target object dof pos
            self.dist_object_gripper.unsqueeze(-1),         # distance between object and gripper
            self.grasp_pos_offset,                          # distance between current and previous gripper position
            self.action_handle_frame,                       # action in handle frame
        ]

        self.obs_buf = torch.cat(obs_tensors, dim=-1)

        return self.obs_buf
    
    def _set_object_init_and_target_dof_pos(self):
        """ Set initial and target object dof positions for open task """
        self.rest_object_dof_pos = self.object_dof_lower_limits.clone()
        self.target_object_dof_pos = self.object_dof_upper_limits.clone()

    def prepare_init_states(self, init_states):
        """ Prepare initial states for the environment.
            Open / close tasks use the same set of initial states, 
            but filter them according to different object_dof_pos constraints.
        """
        if init_states is not None and type(init_states) == str:
            assert os.path.isdir(init_states), "Invalid path to initial states."
            if not self.object_id in self.init_states_cache:
                for path in os.listdir(init_states):
                    object_code = path[:-len('.pt')]
                    curr_object_code = self.object_code
                    if object_code == curr_object_code:
                        init_states = torch.load(os.path.join(init_states, path))
                        break
                idx = torch.randperm(len(list(init_states.values())[0]))
                for key in init_states.keys():
                    init_states[key] = init_states[key][idx]
                self.init_states_cache[self.object_id] = init_states
            else:
                init_states = self.init_states_cache[self.object_id]
                
        self.init_states = init_states
        if init_states is None:
            return
        
        # move to device
        first_key = list(init_states.keys())[0]
        for key in self.init_states.keys():
            self.init_states[key] = self.init_states[key].to(self.device)

        # filter
        dof_ratio = (self.init_states["init_object_dof_pos"] - self.rest_object_dof_pos.unsqueeze(0)) \
            / (self.target_object_dof_pos.reshape(-1, self.object_num_dofs) - self.rest_object_dof_pos.unsqueeze(0))
        filter = dof_ratio.mean(dim=-1) < self.cfg_task.randomize.max_object_init_dof_ratio
        for key in self.init_states.keys():
            self.init_states[key] = self.init_states[key][filter]
        print(f"Filtered {len(filter) - filter.sum()} initial states")

        # set number of initial states
        self.num_init_states = len(self.init_states[first_key])
        self.episode_cur = 0
        self.val_num = int(self.num_init_states * self.cfg["env"].get("val_ratio", 0.1))
        self.episode_cur_val = self.num_init_states - self.val_num
        print("Number of initial states:", self.num_init_states)
        print("Validation set size:", self.val_num)

    def reset_idx(self, env_ids: torch.Tensor, validation_set: bool = False, switch_object: bool = False, init_states = None):
        FrankaGraspHandle.reset_idx(self, env_ids, validation_set, switch_object, init_states)

        # record position of fingertip center in the handle frame
        self.init_fingertip_centered_pos_local[:] = quat_rotate(
            quat_conjugate(self.handle_quat), self.fingertip_centered_pos - self.handle_pos
        )
        self.prev_fingertip_centered_pos_local = quat_rotate(
            quat_conjugate(self.handle_quat), self.fingertip_centered_pos - self.handle_pos
        )

        if not self.sample_mode and self.object_meta["type"] == "door" and self.cfg_env.rl.partial_open_for_door:
            self.target_object_dof_pos = self.init_object_dof_pos + np.deg2rad(self.cfg_env.rl.partial_open_degree)
            self.target_object_dof_pos = torch.clamp(self.target_object_dof_pos, self.object_dof_lower_limits, self.object_dof_upper_limits)
