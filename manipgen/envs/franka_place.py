from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgymenvs.utils.torch_jit_utils import (
    quat_from_euler_xyz,
    quat_mul,
    quat_rotate,
    quat_conjugate,
)

from gym import spaces
import numpy as np
import torch

from typing import Dict, Any, Tuple, List, Set
import math

from manipgen.envs.franka_env import FrankaEnv
from manipgen.envs.franka_pick import (
    FrankaPick, 
    transform_keypoints_6d,
    compute_nearest_keypoint_6d_distance,
)
from manipgen.utils.geometry_utils import get_keypoint_offsets_6d
import isaacgymenvs.tasks.factory.factory_control as fc
import os

class FrankaPlace(FrankaEnv):
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
            "target_xy_noise": self.cfg["rl"]["target_xy_noise"],
            "dist_object_keypoints_reward_temp": self.cfg["rl"]["dist_object_keypoints_reward_temp"],
            "dist_object_gripper_reward_temp": self.cfg["rl"]["dist_object_gripper_reward_temp"],
            "eef_pose_consistency_reward_temp": self.cfg["rl"]["eef_pose_consistency_reward_temp"],
            "finger_contact_force_threshold": self.cfg["rl"]["finger_contact_force_threshold"],
            "finger_contact_reward_temp": self.cfg["rl"]["finger_contact_reward_temp"],
            "success_bonus": self.cfg["rl"]["success_bonus"],
            "success_threshold": self.cfg["rl"]["success_threshold"],
        }

        # initial state sampling & reset
        self.object_pos_noise = self.cfg_task.sampler.object_pos_noise
        self.object_pos_center = [
            self.table_pose.p.x,
            self.table_pose.p.y,
            self.cfg_base.env.table_height + 0.2,
        ]

        # Reset all environments
        self.object_id = -1
        self.reset_idx(
            torch.arange(self.num_envs, device=self.device),
            switch_object=True,
            init_states=init_states,
        )

    _compute_nearest_grasp_pose_keypoint_distance = FrankaPick._compute_nearest_grasp_pose_keypoint_distance
    _compute_keypoints_pos = FrankaPick._compute_keypoints_pos
    _compute_lift_height = FrankaPick._compute_lift_height
    create_envs = FrankaPick.create_envs
    _create_actors = FrankaPick._create_actors
    sample_unidexgrasp_clutter_asset = FrankaPick.sample_unidexgrasp_clutter_asset
    import_clutter_and_obstacle_assets = FrankaPick.import_clutter_and_obstacle_assets
    _check_obstacle_pose = FrankaPick._check_obstacle_pose
    _get_random_rotation = FrankaPick._get_random_rotation
    _clear_clutter_and_obstacles = FrankaPick._clear_clutter_and_obstacles
    _reset_clutter_and_obstacles = FrankaPick._reset_clutter_and_obstacles

    def _acquire_task_tensors(self):
        """Acquire and wrap tensors. Create views."""

        # object state tensors are created in switch_object
        # because we need to know object id in multitask mode

        # fingertip pos
        self.left_fingertip_pos = self.body_pos[:, self.left_fingertip_body_id_env, 0:3]
        self.right_fingertip_pos = self.body_pos[:, self.right_fingertip_body_id_env, 0:3]

        # contact force on fingers
        self.lf_contact_force = self.contact_force.view(-1, 3)[self.franka_left_finger_ids_sim, :]
        self.rf_contact_force = self.contact_force.view(-1, 3)[self.franka_right_finger_ids_sim, :]

        # gripper direction
        self.up_v = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)
        self.gripper_direction = torch.zeros(self.num_envs, 3, device=self.device)
        self.gripper_direction_prev = torch.zeros(self.num_envs, 3, device=self.device)
        self.gripper_direction_init = torch.zeros(self.num_envs, 3, device=self.device)

        # reward
        self.num_object_keypoints = 16 if self.sample_mode else self.cfg_env.rl.num_object_keypoints    # set to 16 in sample mode - match pick env
        self.object_keypoints = torch.zeros(self.num_envs, self.num_object_keypoints * 3, device=self.device)
        self.dist_object_keypoints = torch.zeros(self.num_envs, device=self.device)
        self.dist_object_gripper = torch.zeros(self.num_envs, device=self.device)  # distance from object to gripper
        self.dist_xy = torch.zeros(self.num_envs, device=self.device)  # distance between current object xy and target object xy
        self.object_height = torch.zeros(self.num_envs, device=self.device)  # object height

        # override in reset
        self.target_object_xy = torch.zeros(self.num_envs, 2, device=self.device)

        # clutter and obstacle
        if self.cfg_env.env.enable_clutter_and_obstacle:
            self.clutter_pos = self.root_pos[:, self.clutter_actor_id_env[0]:self.clutter_actor_id_env[-1]+1, 0:3]                              # (n_envs, n_clutter_objects, 3)
            self.clutter_quat = self.root_quat[:, self.clutter_actor_id_env[0]:self.clutter_actor_id_env[-1]+1, :]
            self.clutter_contact_force = self.contact_force.view(-1, 3)[self.clutter_rigid_body_ids_sim, :].transpose(0, 1).norm(dim=-1)        # (n_envs, n_clutter_objects)
            self.obstacle_pos = self.root_pos[:, self.obstacle_actor_id_env[0]:self.obstacle_actor_id_env[-1]+1, 0:3]                           # (n_envs, n_obstacles, 3)
            self.obstacle_quat = self.root_quat[:, self.obstacle_actor_id_env[0]:self.obstacle_actor_id_env[-1]+1, :]
            self.obstacle_contact_force = self.contact_force.view(-1, 3)[self.obstacle_rigid_body_ids_sim, :].transpose(0, 1).norm(dim=-1)      # (n_envs, n_obstacles)

            self.clutter_pos_xy = self.clutter_pos[:, :, :2].reshape(self.num_envs, -1)                                              # (n_envs, n_clutter_objects * 2)
        else:
            self.clutter_pos_xy = torch.zeros((self.num_envs, 0), device=self.device)

    def _refresh_task_tensors(self):
        """Refresh tensors."""
        # contact force on fingers
        self.lf_contact_force = self.contact_force.view(-1, 3)[self.franka_left_finger_ids_sim, :]
        self.rf_contact_force = self.contact_force.view(-1, 3)[self.franka_right_finger_ids_sim, :]

        self.object_keypoints = self._compute_keypoints_pos()
        if self.cfg_task.rl.use_object_keypoint_6d:
            self.dist_object_keypoints = self._compute_nearest_object_keypoint_6d_distance()
        else:
            current_object_keypoints = self.object_keypoints.reshape(-1, self.num_object_keypoints, 3).clone()
            target_object_keypoints = self.target_object_keypoints.clone().reshape(-1, self.num_object_keypoints, 3)
            current_object_keypoints[:, :, :2] -= self.target_object_xy.unsqueeze(1)    # apply target xy offset to current keypoints as they have the same dimension
            self.dist_object_keypoints = compute_object_keypoint_distance(
                current_object_keypoints, 
                target_object_keypoints
            )
        self.dist_object_gripper = self._compute_object_gripper_distance(type="finger")
        self.dist_xy = torch.norm(self.object_pos[:, :2] - self.target_object_xy, dim=-1)
        self.object_height = self._compute_lift_height()
        self.gripper_direction = quat_rotate(self.fingertip_centered_quat, self.up_v)
        self.dist_gripper_direction = torch.arccos(
            torch.clamp((self.gripper_direction_prev * self.gripper_direction).sum(dim=-1), min=-1.0, max=1.0)
        )
        self.gripper_direction_prev = self.gripper_direction.clone()

        if self.sample_mode:
            self.dist_grasp_keypoints = self._compute_nearest_grasp_pose_keypoint_distance()
        
        # clutter and obstacle
        if self.cfg_env.env.enable_clutter_and_obstacle:
            self.clutter_contact_force = self.contact_force.view(-1, 3)[self.clutter_rigid_body_ids_sim, :].transpose(0, 1).norm(dim=-1)        # (n_envs, n_clutter_objects)
            self.obstacle_contact_force = self.contact_force.view(-1, 3)[self.obstacle_rigid_body_ids_sim, :].transpose(0, 1).norm(dim=-1)      # (n_envs, n_obstacles)
            self.clutter_pos_xy = self.clutter_pos[:, :, :2].reshape(self.num_envs, -1)                                                         # (n_envs, n_clutter_objects * 2)
        else:
            self.clutter_pos_xy = torch.zeros((self.num_envs, 0), device=self.device)

    def _compute_nearest_object_keypoint_6d_distance(self):
        """Based on current object keypoints (6d), compute distance to nearest rest object pose keypoints (6d)."""
        object_pos = self.object_pos.clone()
        object_pos[:, :2] -= self.target_object_xy
        object_pose = torch.cat([object_pos, self.object_quat], dim=-1)
        currect_object_keypoints_6d = transform_keypoints_6d(object_pose, self.object_keypoint_6d_offsets)
        # compute distance to nearest rest pose based on keypoints
        dist_object_keypoint_6d = compute_nearest_keypoint_6d_distance(currect_object_keypoints_6d, self.target_object_keypoints_6d)

        return dist_object_keypoint_6d

    def _compute_object_gripper_distance(self, object_pose=None, type="finger"):
        """Compute distance between object and gripper."""
        if object_pose is None:
            object_pos = self.object_pos
            object_rot = self.object_quat
        else:
            object_pos = object_pose[:, :3]
            object_rot = object_pose[:, 3:]
        grip_site_pos = self.fingertip_centered_pos
        lf_pos = self.left_fingertip_pos
        rf_pos = self.right_fingertip_pos
        if type == "grasp":
            # compute distance to pre-sampled grasp points
            grip_site_pos_local = quat_rotate(
                quat_conjugate(object_rot), grip_site_pos - object_pos
            )
            lf_pos_local = quat_rotate(quat_conjugate(object_rot), lf_pos - object_pos)
            rf_pos_local = quat_rotate(quat_conjugate(object_rot), rf_pos - object_pos)
            object_gripper_dist, object_lf_dist, object_rf_dist = (
                distance_to_grasp_points(
                    self.grasp_pose[:, :3], grip_site_pos_local, lf_pos_local, rf_pos_local
                )
            )
            distance = (object_gripper_dist + torch.maximum(object_lf_dist, object_rf_dist)) / 2
        elif type == "finger":
            # compute distance to pre-sampled finger positions
            lf_pos_local = quat_rotate(quat_conjugate(object_rot), lf_pos - object_pos)
            rf_pos_local = quat_rotate(quat_conjugate(object_rot), rf_pos - object_pos)
            lf_points = self.finger_pos[:, :3]
            rf_points = self.finger_pos[:, 3:]
            distance = distance_to_target_finger_pos(
                lf_points, rf_points, lf_pos_local, rf_pos_local
            )
        elif type == "center":
            # compute distance between object and gripper center
            object_gripper_dist = distance_to_object_center(object_pos, grip_site_pos)
            object_lf_dist = distance_to_object_center(object_pos, lf_pos)
            object_rf_dist = distance_to_object_center(object_pos, rf_pos)
            distance = (object_gripper_dist + torch.maximum(object_lf_dist, object_rf_dist)) / 2
        else:
            raise ValueError("Invalid type.")

        return distance
    
    def _check_place_success(self):
        place_success = (self.dist_xy < self.reward_settings["success_threshold"])
        return place_success
    
    def hardcode_control(self, get_camera_images=False):
        camera_images = []

        # teleportation: open gripper
        self.ctrl_target_fingertip_midpoint_pos[:] = self.fingertip_centered_pos
        self.ctrl_target_fingertip_midpoint_quat[:] = self.fingertip_centered_quat
        target_gripper_dof_pos = self.gripper_dof_pos[:] + 0.02
        images = self.move_gripper_to_target_pose(gripper_dof_pos=target_gripper_dof_pos, sim_steps=40, get_camera_images=get_camera_images)
        camera_images.extend(images)

        # teleportation: pull out the arm
        self.ctrl_target_fingertip_midpoint_pos[:] -= 0.15 * self.gripper_direction
        images = self.move_gripper_to_target_pose(gripper_dof_pos=target_gripper_dof_pos, sim_steps=30, get_camera_images=get_camera_images)
        camera_images.extend(images)

        return camera_images

    def compute_reward(self):
        """Compute rewards."""
        dist_object_keypoints_reward_temp = self.reward_settings["dist_object_keypoints_reward_temp"]
        dist_object_gripper_reward_temp = self.reward_settings["dist_object_gripper_reward_temp"]
        eef_pose_consistency_reward_temp = self.reward_settings["eef_pose_consistency_reward_temp"]
        finger_contact_reward_temp = self.reward_settings["finger_contact_reward_temp"]
        finger_contact_force_threshold = self.reward_settings["finger_contact_force_threshold"]
        success_bonus = self.reward_settings["success_bonus"]

        # component 1: keypoint distance between current object pose and nearest rest object pose
        # encourage moving the object to a pre-sampled rest pose
        if self.cfg_task.rl.use_object_keypoint_6d:
            dist_object_keypoints_reward = (
                torch.exp(-1 * self.dist_object_keypoints)
                + torch.exp(-1 * self.dist_object_keypoints * 10)
                + torch.exp(-1 * self.dist_object_keypoints * 100)
                + torch.exp(-1 * self.dist_object_keypoints * 1000)
            ) / 4
        else:
            dist_object_keypoints_reward = torch.exp(dist_object_keypoints_reward_temp * self.dist_object_keypoints)

        # component 2: keypoint distance between current gripper pose and nearest pre-sampled grasp pose
        # encourage the gripper to move to the nearest grasp pose
        dist_object_gripper_reward = torch.exp(dist_object_gripper_reward_temp * self.dist_object_gripper)

        # component 3: the angle between the current and previous gripper pose along the gripper's central axis
        # we want the gripper to minimize its change in orientation to avoid collision with invisible obstacles
        eef_pose_consistency_reward = eef_pose_consistency_reward_temp * self.dist_gripper_direction

        # component 4: contact force on the gripper fingers
        # penalize the fingers from contacting the obstacles
        finger_contact_reward = finger_contact_reward_temp * (
            torch.clamp(torch.max(self.lf_contact_force[:, 2], self.rf_contact_force[:, 2]), min=finger_contact_force_threshold) \
            - finger_contact_force_threshold
        )

        total_reward = dist_object_keypoints_reward * dist_object_gripper_reward + eef_pose_consistency_reward + finger_contact_reward

        self.rew_buf[:] = total_reward
        reset = (self.progress_buf == self.max_episode_length - 1)

        if self.render_hardcode_control:
            self.extras['hardcode_images'] = []

        is_last_step = (self.progress_buf[0] == self.max_episode_length - 1)
        if is_last_step and not self.disable_hardcode_control:
            # log keypoint distance
            keypoint_dist = self.dist_object_keypoints.mean()
            self.extras['keypoint_dist'] = keypoint_dist
            self.extras['dist_xy'] = self.dist_xy.mean()
            self.extras['gripper_direction_dist'] = torch.arccos(
                torch.clamp((self.gripper_direction_init * self.gripper_direction).sum(dim=-1), min=-1.0, max=1.0)
            ).mean()
            self.extras['contact_force_z'] = torch.max(self.lf_contact_force[:, 2], self.rf_contact_force[:, 2]).mean()
            # hardcode control
            hardcode_images = self.hardcode_control(get_camera_images=self.render_hardcode_control)
            if self.render_hardcode_control:
                self.extras['hardcode_images'] = hardcode_images
            # check place success and apply success bonus
            place_success = self._check_place_success()
            self.rew_buf[:] += place_success * success_bonus
            self.success_buf[:] = place_success
            self.consecutive_successes[:] = torch.where(
                reset > 0, self.success_buf * reset, self.consecutive_successes
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

        open_gripper = (self.progress_buf == self.max_episode_length - 1).float()
        if self.disable_hardcode_control:
            open_gripper[:] = 0.0
        obs_tensors = [
            local_eef_pos,
            local_eef_quat,
            self.fingertip_centered_linvel,                 # linear velocity of the gripper
            self.fingertip_centered_angvel,                 # angular velocity of the gripper
            self.object_pos,                                # position of the object
            self.object_quat,                               # orientation of the object
            self.object_keypoints,                          # position of keypoints of the object
            self.target_object_xy,                          # target xy position of the object
            self.dist_gripper_direction.unsqueeze(-1),      # distance between current and previous gripper direction
            self.dist_object_keypoints.unsqueeze(-1),       # distance between current and target object keypoints
            self.dist_object_gripper.unsqueeze(-1),         # distance between object and gripper
            self.lf_contact_force[:, -1:],                  # contact force on the left finger along z-axis
            self.rf_contact_force[:, -1:],                  # contact force on the right finger along z-axis
            self.clutter_pos_xy,                            # xy position of clutter objects
            self.obstacle_pos.reshape(self.num_envs, -1),   # position of obstacles
            self.obstacle_quat.reshape(self.num_envs, -1),  # orientation of obstacles
            open_gripper.unsqueeze(-1),                     # whether to open the gripper and lift up the arm
        ]

        self.obs_buf = torch.cat(obs_tensors, dim=-1)

        return self.obs_buf

    def prepare_init_states(self, init_states):
        """ Prepare initial states for the environment """
        if init_states is not None and type(init_states) == str:
            assert os.path.isdir(init_states), "Invalid path to initial states."
            if not self.object_id in self.init_states_cache:
                for path in os.listdir(init_states):
                    object_code, object_scale = path[:-len('.pt')].split("_")
                    curr_scale = f"{int(100 * self.object_scale):03d}"
                    curr_object_code = self.object_code.replace('/', '-')
                    if object_code == curr_object_code and object_scale == curr_scale:
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

        # set number of initial states
        self.num_init_states = len(self.init_states[first_key])
        self.episode_cur = 0
        self.val_num = int(self.num_init_states * self.cfg["env"].get("val_ratio", 0.1))
        self.episode_cur_val = self.num_init_states - self.val_num
        print("Number of initial states:", self.num_init_states)
        print("Validation set size:", self.val_num)

    def switch_object(self, init_states):
        if self.object_id != -1:    
            # if not the first switch, reset the current object to the initial state
            # move current object out of frame, before switching to new object
            self.object_pos[:, 0] = 1.0
            self.object_pos[:, 1] = 0.0
            self.object_pos[:, 2] = 0.15
            self.object_quat[:, :] = 0.0
            self.object_quat[:, 3] = 1.0
            self.object_linvel[:, :] = 0.0
            self.object_angvel[:, :] = 0.0

            object_actor_ids_sim = self.object_actor_ids_sim.to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.root_state),
                gymtorch.unwrap_tensor(object_actor_ids_sim),
                len(object_actor_ids_sim),
            )

            self.simulate_and_refresh()

            # backup the current episode count
            self.episode_cur_multitask[self.object_id] = self.episode_cur
            
            # clear success in extras:
            if f'{self.object_code}_{self.object_scale}_success' in self.extras:
                 del self.extras[f'{self.object_code}_{self.object_scale}_success']
        else:
            self.episode_cur_multitask = [0 for _ in range(self.num_objects)]

        self.object_id = (self.object_id + 1) % self.num_objects
        self.object_code = self.object_codes[self.object_id]
        self.object_scale = self.object_scales[self.object_id]

        self.object_handles = self.object_handles_multitask[self.object_id]
        self.object_actor_ids_sim = self.object_actor_ids_sim_multitask[self.object_id]
        self.object_rigid_body_ids_sim = self.object_rigid_body_ids_sim_multitask[self.object_id]
        self.object_actor_id_env = self.object_actor_id_env_multitask[self.object_id]

        # object pose and velocity
        self.object_pos = self.root_pos[:, self.object_actor_id_env, 0:3]
        self.object_quat = self.root_quat[:, self.object_actor_id_env, 0:4]
        self.object_linvel = self.root_linvel[:, self.object_actor_id_env, 0:3]
        self.object_angvel = self.root_angvel[:, self.object_actor_id_env, 0:3]

        # load grasp data
        self.load_unidexgrasp_data(
            self.object_codes[self.object_id],
            self.object_scales[self.object_id],
            load_grasp_poses=self.sample_mode,
            load_finger_pos=True,                   # comptue object-gripper distance
            load_convex_hull_points=True,           # compute object height
            load_object_keypoints=True,             # compute object keypoints
            filter_threshold=self.cfg_task.rl.filter_pose_threshold if self.sample_mode else 0.0,
        )

        if self.sample_mode:
            # process grasp data: keypoints corresponding to pre-sampled grasp poses in the object frame
            # compute obs for pick policy in init state sampler
            gripper_keypoint_scale = self.cfg_task.rl.gripper_keypoint_scale
            self.gripper_keypoint_offsets = get_keypoint_offsets_6d(self.device) * gripper_keypoint_scale
            
            assert self.cfg_task.rl.gripper_keypoint_dof in (5, 6), "Invalid gripper keypoint DOF."
            if self.cfg_task.rl.gripper_keypoint_dof == 5:
                select = torch.tensor([True, False, True, False, False, True, False]).to(self.device)
                self.gripper_keypoint_offsets = self.gripper_keypoint_offsets[select]
            self.target_grasp_keypoints_local = transform_keypoints_6d(self.grasp_pose, self.gripper_keypoint_offsets)

        # compute target object keypoints
        object_rest_pose_list = []
        object_euler_xy = self.grasp_data[self.object_code][self.object_scale]["object_euler_xy"]
        object_z = self.grasp_data[self.object_code][self.object_scale]["object_init_z"]
        num_rest_samples = len(object_euler_xy)
        object_rest_pos = torch.zeros((num_rest_samples, 3), device=self.device)
        object_rest_pos[:, 2] = object_z + self.cfg_base.env.table_height
        num_rotation_divisions = 8
        for i in range(num_rotation_divisions):
            theta = 2 * math.pi * i / num_rotation_divisions
            object_rest_rot = quat_from_euler_xyz(
                object_euler_xy[:, 0],
                object_euler_xy[:, 1],
                torch.ones(num_rest_samples, device=self.device) * theta,
            )
            object_rest_pose = torch.cat([object_rest_pos, object_rest_rot], dim=-1)
            object_rest_pose_list.append(object_rest_pose)
        object_rest_pose = torch.cat(object_rest_pose_list, dim=0)
        if self.cfg_task.rl.use_object_keypoint_6d:
            self.object_keypoint_6d_offsets = get_keypoint_offsets_6d(self.device) * self.cfg_task.rl.object_keypoint_6d_scale
            self.target_object_keypoints_6d = transform_keypoints_6d(object_rest_pose, self.object_keypoint_6d_offsets)
        else:
            self.target_object_keypoints = self._compute_keypoints_pos(object_rest_pose)

        # load initial states
        self.prepare_init_states(init_states)
        self.episode_cur = self.episode_cur_multitask[self.object_id]
    
    def update_extras(self, camera_images):
        super().update_extras(camera_images)
        self.extras[f'{self.object_code}_{self.object_scale}_success'] = self.success_buf

    def _reset_object(self, env_ids, object_init_pose=None):
        """Reset pose of the object.
        Args:
            env_ids: environment ids to reset
            object_init_pose: presampled initial pose of the object. sample if None
        """
        if object_init_pose is not None:
            init_pos = object_init_pose[:, 0:3]
            init_rot = object_init_pose[:, 3:7]
        else:
            num_resets = len(env_ids)
            object_euler_xy = self.grasp_data[self.object_code][self.object_scale][
                "object_euler_xy"
            ]
            object_z = self.grasp_data[self.object_code][self.object_scale][
                "object_init_z"
            ]
            num_rest_samples = len(object_euler_xy)
            object_rest_pos = torch.zeros((num_rest_samples, 3), device=self.device)
            object_rest_pos[:, 2] = (
                object_z + self.cfg_base.env.table_height
            )
            object_rest_rot = quat_from_euler_xyz(
                object_euler_xy[:, 0],
                object_euler_xy[:, 1],
                torch.zeros(num_rest_samples, device=self.device),
            )

            select_rest_pose = torch.randint(
                num_rest_samples, (num_resets,), device=self.device
            )
            init_pos = (
                2 * torch.rand((num_resets, 3), device=self.device) - 1.0
            )
            init_pos[:, 0] = (
                self.object_pos_center[0] + init_pos[:, 0] * self.object_pos_noise
            )
            init_pos[:, 1] = (
                self.object_pos_center[1] + init_pos[:, 1] * self.object_pos_noise
            )
            init_pos[:, 2] = object_rest_pos[select_rest_pose, 2] + 0.001
            rest_rot = object_rest_rot[select_rest_pose]
            theta_half = (
                torch.rand(num_resets, device=self.device) * math.pi
            )
            rot_z = torch.zeros((num_resets, 4), device=self.device)
            rot_z[:, 2] = torch.sin(theta_half)
            rot_z[:, 3] = torch.cos(theta_half)
            init_rot = quat_mul(rot_z, rest_rot)
        
        self.object_pos[env_ids] = init_pos
        self.object_quat[env_ids] = init_rot
        self.object_linvel[env_ids, :] = 0.0
        self.object_angvel[env_ids, :] = 0.0

        object_actor_ids_sim = self.object_actor_ids_sim[env_ids].to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state),
            gymtorch.unwrap_tensor(object_actor_ids_sim),
            len(object_actor_ids_sim),
        )

        # Update target xy
        p = torch.randn((len(env_ids), 2), device=self.device)
        p = p / p.norm(dim=-1, keepdim=True)
        u = torch.rand(len(env_ids), device=self.device).unsqueeze(-1) ** (1.0 / 2)
        self.target_object_xy[env_ids] = init_pos[:, :2] + self.reward_settings["target_xy_noise"] * u * p

    def reset_idx(self, env_ids: torch.Tensor, validation_set: bool = False, switch_object: bool = False, init_states = None):
        """Reset environments having the provided indices.
        Args:
            env_ids: environments to reset
            validation_set: if True, load initial states from the validation set
            switch_object: if True, switch to the next object (only applicable in multitask setting)
            init_states: pre-sampled initial states for the next object (should be passed if switch_object is True)
        """
        if switch_object:
            self.switch_object(init_states)

        if self.init_states is not None:
            if validation_set:
                start = self.episode_cur_val
                end = self.episode_cur_val = self.episode_cur_val + len(env_ids)
                assert end < self.num_init_states, "Out of validation set."
            else:
                start = self.episode_cur
                end = self.episode_cur = self.episode_cur + len(env_ids)
                assert end < self.num_init_states - self.val_num, "Out of training set."
            franka_init_dof_pos = self.init_states["franka_dof_pos"][start:end]
            object_init_pose = self.init_states["object_pose"][start:end]
        else:
            franka_init_dof_pos = None
            object_init_pose = None

        self._clear_clutter_and_obstacles(env_ids)

        self._reset_franka(env_ids, franka_init_dof_pos)
        self._reset_object(env_ids, object_init_pose)
        self.disable_gravity()
        self.simulate_and_refresh()
        self.enable_gravity()

        self._reset_clutter_and_obstacles(env_ids)

        if self.cfg_env.env.enable_clutter_and_obstacle:
            # If clutter and obstacles are enabled, we check collision in _reset_clutter_and_obstacles.
            # After the collision check, we need to reset the object and franka again.
            self._reset_franka(env_ids, franka_init_dof_pos)
            self._reset_object(env_ids, object_init_pose)
            self.simulate_and_refresh()

        self.gripper_direction_init = quat_rotate(self.fingertip_centered_quat, self.up_v)
        self.gripper_direction_prev = quat_rotate(self.fingertip_centered_quat, self.up_v)

        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        self.prev_visual_obs = None
        self.prev_seg_obs = None
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def pre_steps(self, obs, keep_runs):
        """The function generates easier initial states to test the policy.
        Args:
            obs: observations before the pre-step
            keep_runs: whether a run is valid or not (not used here)
            num_pre_steps: number of steps to run in the pre-step
        Returns:
            obs: observations after the pre-step
            keep_runs: whether a run is valid or not after the pre-step
        """
        capture_video = self.capture_video
        self.capture_video = False

        # lift up
        self.ctrl_target_fingertip_midpoint_pos[:] = self.fingertip_centered_pos.clone()
        self.ctrl_target_fingertip_midpoint_quat[:] = self.fingertip_centered_quat.clone()
        self.ctrl_target_fingertip_midpoint_pos[:, 2] += 0.3
        self.move_gripper_to_target_pose(gripper_dof_pos=-0.04, sim_steps=30)

        # move to target rot
        target_pos = self.fingertip_centered_pos.clone()
        # rotation noise: along any axis by an angle < 30 degree
        down_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        rot_noise = torch.zeros((self.num_envs, 4), device=self.device)
        p = torch.randn((self.num_envs, 3), device=self.device)
        p = p / p.norm(dim=-1, keepdim=True)
        theta_half = (2 * torch.rand(self.num_envs, device=self.device) - 1) * math.pi / 6 * 0
        rot_noise[:, :3] = p * torch.sin(theta_half).unsqueeze(-1)
        rot_noise[:, 3] = torch.cos(theta_half)
        target_rot = quat_mul(rot_noise, down_q)
        self.ctrl_target_fingertip_midpoint_pos[:] = target_pos
        self.ctrl_target_fingertip_midpoint_quat[:] = target_rot
        self.move_gripper_to_target_pose(gripper_dof_pos=-0.04, sim_steps=60)

        # lower 
        height_offset = self.fingertip_centered_pos[:, 2] - self.object_height[:]
        target_height = torch.rand(self.num_envs, device=self.device) * 0.08 + 0.05 + height_offset
        self.ctrl_target_fingertip_midpoint_pos[:, 2] = target_height
        self.move_gripper_to_target_pose(gripper_dof_pos=-0.04, sim_steps=60)

        self.target_object_xy[:] = self.object_pos[:, :2].clone()

        # check validity
        # gripper not fully closed
        caging = self.gripper_dof_pos[:, -2:].sum(dim=-1) > 0.002

        # object is at least 1cm above the table
        lifted = self.object_height > 0.01

        keep_runs &= (caging & lifted)

        self.capture_video = capture_video
        return obs, keep_runs


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def distance_to_target_finger_pos(lf_points, rf_points, lf_local, rf_local):
    # type: (Tensor, Tensor, Tensor, Tensor) -> Tensor

    num_envs = lf_local.shape[0]
    num_points = lf_points.shape[0]

    lf_dist = torch.norm(lf_points.expand(num_envs, num_points, 3) - lf_local.unsqueeze(1), dim=-1)
    rf_dist = torch.norm(rf_points.expand(num_envs, num_points, 3) - rf_local.unsqueeze(1), dim=-1)
    min_dist = torch.min(lf_dist + rf_dist, dim=-1)[0]

    # symmetric
    lf_dist = torch.norm(rf_points.expand(num_envs, num_points, 3) - lf_local.unsqueeze(1), dim=-1)
    rf_dist = torch.norm(lf_points.expand(num_envs, num_points, 3) - rf_local.unsqueeze(1), dim=-1)
    min_dist_sym = torch.min(lf_dist + rf_dist, dim=-1)[0]

    dist = torch.minimum(min_dist, min_dist_sym) / 2
    return dist

@torch.jit.script
def distance_to_grasp_points(grasp_points, position_local, lf_local, rf_local):
    # type: (Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor]

    num_envs = position_local.shape[0]
    num_points = grasp_points.shape[0]
    grasp_points_dist = torch.norm(
        grasp_points.expand(num_envs, num_points, 3) - position_local.unsqueeze(1),
        dim=-1,
    )
    min_dist, min_idx = torch.min(grasp_points_dist, dim=-1)

    lf_dist = torch.norm(grasp_points[min_idx] - lf_local, dim=-1)
    rf_dist = torch.norm(grasp_points[min_idx] - rf_local, dim=-1)

    return min_dist, lf_dist, rf_dist

@torch.jit.script
def distance_to_object_center(object_pos, position):
    # type: (Tensor, Tensor) -> Tensor

    dist = torch.norm(object_pos - position, dim=-1)
    return dist

@torch.jit.script
def compute_object_keypoint_distance(current_keypoints, target_keypoints):
    # type: (Tensor, Tensor) -> Tensor
    """Compute distance between current and target keypoints sampled on the object mesh
    """

    # current_keypoints: num_envs x num_keypoints x 3
    # target_keypoints: num_rest_poses x num_keypoints x 3

    num_grasp_poses = target_keypoints.shape[0]
    num_envs = current_keypoints.shape[0]

    target_keypoints = target_keypoints.unsqueeze(0).expand(num_envs, -1, -1, -1)
    current_keypoints = current_keypoints.unsqueeze(1).expand(-1, num_grasp_poses, -1, -1)

    dist = torch.norm(current_keypoints - target_keypoints, dim=-1).mean(dim=-1)
    min_dist, min_idx = dist.min(dim=-1)
    return min_dist
