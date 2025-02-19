from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch

from gym import spaces
import numpy as np
import torch
import trimesh as tm

from typing import Dict, Any, Tuple, List, Set
import math

from manipgen.envs.franka_env import FrankaEnv
from manipgen.envs.franka_pick import compute_axis_object_alignment
from manipgen.utils.geometry_utils import get_keypoint_offsets_6d
import isaacgymenvs.tasks.factory.factory_control as fc
from isaacgymenvs.utils.torch_jit_utils import (
    tf_combine,
    quat_mul,
    quat_rotate,
    quat_conjugate,
)

class FrankaPickCube(FrankaEnv):
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
            "dist_gripper_reward_temp": self.cfg["rl"]["dist_gripper_reward_temp"],
            "dist_xy_reward_temp": self.cfg["rl"]["dist_xy_reward_temp"],
            "object_in_view_reward_temp": self.cfg["rl"]["object_in_view_reward_temp"],
            "object_in_view_reward_threshold": self.cfg["rl"]["object_in_view_reward_threshold"],
            "success_bonus": self.cfg["rl"]["success_bonus"],
        }

        # initial states loading
        self.prepare_init_states(init_states)

        # initial state sampling & reset
        self.object_pos_noise = self.cfg_task.sampler.object_pos_noise
        self.object_pos_center = [
            self.table_pose.p.x,
            self.table_pose.p.y,
            self.cfg_base.env.table_height + 0.5 * self.box_size,
        ]

        # Reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def _import_cube_asset(self):
        """Import cube asset."""
        asset_options = gymapi.AssetOptions()
        self.box_size = self.cfg_task.env.cube_size
        # name it `object_asset` to be consistent with general pick policy
        object_asset = self.gym.create_box(
            self.sim, self.box_size, self.box_size, self.box_size, asset_options
        )
        return object_asset

    def create_envs(self):
        """Set env options. Import assets. Create actors."""

        lower = gymapi.Vec3(-self.cfg_base.env.env_spacing, -self.cfg_base.env.env_spacing, 0.0)
        upper = gymapi.Vec3(self.cfg_base.env.env_spacing, self.cfg_base.env.env_spacing, self.cfg_base.env.env_spacing)
        num_per_row = int(np.sqrt(self.num_envs))

        # import assets
        self.print_sdf_warning()
        franka_asset, table_asset = self.import_franka_assets()
        object_asset = self._import_cube_asset()
        
        # create actors
        self._create_actors(lower, upper, num_per_row, franka_asset, table_asset, object_asset)

    def _create_actors(self, lower, upper, num_per_row, franka_asset, table_asset, object_asset):
        """Set initial actor poses. Create actors. Set shape and DOF properties."""

        franka_pose = gymapi.Transform()
        franka_pose.p.x = -self.cfg_base.env.franka_depth
        franka_pose.p.y = 0.0
        franka_pose.p.z = self.cfg_base.env.table_height
        franka_pose.r = gymapi.Quat(
            0.0, 0.0, 0.0, 1.0
        )

        table_pose = gymapi.Transform()
        table_pose.p.x = 0.0
        table_pose.p.y = 0.0
        table_pose.p.z = self.cfg_base.env.table_height * 0.5
        table_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self.table_pose = table_pose

        object_pose = gymapi.Transform()
        object_pose.p.x = 0.0
        object_pose.p.y = 0.0
        object_pose.p.z = self.cfg_base.env.table_height + 0.5 * self.box_size
        object_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.env_ptrs = []
        self.franka_handles = []
        self.table_handles = []
        self.object_handles = []
        self.shape_ids = []
        self.franka_actor_ids_sim = []  # within-sim indices
        self.table_actor_ids_sim = []  # within-sim indices
        self.object_actor_ids_sim = []  # within-sim indices
        self.franka_rigid_body_ids_sim = [] # within-sim indices
        actor_count = 0

        for i in range(self.num_envs):
            # create env
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # create actors
            if self.cfg_env.sim.disable_franka_collisions:
                franka_handle = self.gym.create_actor(
                    env_ptr, franka_asset, franka_pose, 'franka', i + self.num_envs, 0, 1
                )
            else:
                franka_handle = self.gym.create_actor(
                    env_ptr, franka_asset, franka_pose, 'franka', i, 0, 1
                )
            self.franka_actor_ids_sim.append(actor_count)
            actor_count += 1

            table_handle = self.gym.create_actor(
                env_ptr, table_asset, table_pose, "table", i, 1, 2
            )
            self.table_actor_ids_sim.append(actor_count)
            actor_count += 1

            object_handle = self.gym.create_actor(
                env_ptr, object_asset, object_pose, "object", i, 2, 3
            )
            self.object_actor_ids_sim.append(actor_count)
            actor_count += 1

            # set shape properties
            link7_id = self.gym.find_actor_rigid_body_index(
                env_ptr, franka_handle, "panda_link7", gymapi.DOMAIN_ACTOR
            )
            hand_id = self.gym.find_actor_rigid_body_index(
                env_ptr, franka_handle, "panda_hand", gymapi.DOMAIN_ACTOR
            )
            left_finger_id = self.gym.find_actor_rigid_body_index(
                env_ptr, franka_handle, "panda_leftfinger", gymapi.DOMAIN_ACTOR
            )
            right_finger_id = self.gym.find_actor_rigid_body_index(
                env_ptr, franka_handle, "panda_rightfinger", gymapi.DOMAIN_ACTOR
            )
            self.shape_ids = [link7_id, hand_id, left_finger_id, right_finger_id]

            franka_shape_props = self.gym.get_actor_rigid_shape_properties(
                env_ptr, franka_handle
            )
            for shape_id in range(len(franka_shape_props)):
                franka_shape_props[
                    shape_id
                ].friction = self.cfg_base.env.franka_friction
                franka_shape_props[shape_id].rolling_friction = 0.0  # default = 0.0
                franka_shape_props[shape_id].torsion_friction = 0.0  # default = 0.0
                franka_shape_props[shape_id].restitution = 0.0  # default = 0.0
                franka_shape_props[shape_id].compliance = 0.0  # default = 0.0
                franka_shape_props[shape_id].thickness = 0.0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(
                env_ptr, franka_handle, franka_shape_props
            )

            franka_link_names = self.gym.get_actor_rigid_body_names(env_ptr, franka_handle)
            for link_name in franka_link_names:
                link_idx = self.gym.find_actor_rigid_body_index(
                    env_ptr, franka_handle, link_name, gymapi.DOMAIN_SIM
                )
                self.franka_rigid_body_ids_sim.append(link_idx)

            table_shape_props = self.gym.get_actor_rigid_shape_properties(
                env_ptr, table_handle
            )
            table_shape_props[0].friction = self.cfg_base.env.table_friction
            table_shape_props[0].rolling_friction = 0.0  # default = 0.0
            table_shape_props[0].torsion_friction = 0.0  # default = 0.0
            table_shape_props[0].restitution = 0.0  # default = 0.0
            table_shape_props[0].compliance = 0.0  # default = 0.0
            table_shape_props[0].thickness = 0.0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(
                env_ptr, table_handle, table_shape_props
            )

            object_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle)
            object_shape_props[0].friction = self.cfg_env.env.object_friction
            object_shape_props[0].rolling_friction = 0.0  # default = 0.0
            object_shape_props[0].torsion_friction = 0.0  # default = 0.0
            object_shape_props[0].restitution = 0.0  # default = 0.0
            object_shape_props[0].compliance = 0.0  # default = 0.0
            object_shape_props[0].thickness = 0.0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, object_shape_props)

            object_color = gymapi.Vec3(0.6, 0.1, 0.0)
            self.gym.set_rigid_body_color(
                env_ptr, object_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, object_color
            )

            self.franka_num_dofs = self.gym.get_actor_dof_count(env_ptr, franka_handle)

            self.gym.enable_actor_dof_force_sensors(env_ptr, franka_handle)

            self.env_ptrs.append(env_ptr)
            self.franka_handles.append(franka_handle)
            self.table_handles.append(table_handle)
            self.object_handles.append(object_handle)

        self.num_actors = int(actor_count / self.num_envs)  # per env
        self.num_bodies = self.gym.get_env_rigid_body_count(env_ptr)  # per env
        self.num_dofs = self.gym.get_env_dof_count(env_ptr)  # per env

        # For setting targets
        self.franka_actor_ids_sim = torch.tensor(self.franka_actor_ids_sim, dtype=torch.int32, device=self.device)
        self.table_actor_ids_sim = torch.tensor(self.table_actor_ids_sim, dtype=torch.int32, device=self.device)
        self.object_actor_ids_sim = torch.tensor(self.object_actor_ids_sim, dtype=torch.int32, device=self.device)

        # For extracting root pos/quat
        self.franka_actor_id_env = self.gym.find_actor_index(env_ptr, 'franka', gymapi.DOMAIN_ENV)
        self.object_actor_id_env = self.gym.find_actor_index(env_ptr, 'object', gymapi.DOMAIN_ENV)

        # For extracting body pos/quat, force, and Jacobian
        self.robot_base_body_id_env = self.gym.find_actor_rigid_body_index(
            env_ptr, franka_handle, "panda_link0", gymapi.DOMAIN_ENV
        )

        self.hand_body_id_env = self.gym.find_actor_rigid_body_index(
            env_ptr, franka_handle, "panda_hand", gymapi.DOMAIN_ENV
        )

        self.left_finger_body_id_env = self.gym.find_actor_rigid_body_index(
            env_ptr, franka_handle, "panda_leftfinger", gymapi.DOMAIN_ENV
        )
        self.right_finger_body_id_env = self.gym.find_actor_rigid_body_index(
            env_ptr, franka_handle, "panda_rightfinger", gymapi.DOMAIN_ENV
        )
        self.fingertip_centered_body_id_env = self.gym.find_actor_rigid_body_index(
            env_ptr, franka_handle, "panda_fingertip_centered", gymapi.DOMAIN_ENV
        )
        self.hand_body_id_env_actor = self.gym.find_actor_rigid_body_index(
            env_ptr, franka_handle, "panda_hand", gymapi.DOMAIN_ACTOR
        )

        self.left_finger_body_id_env_actor = self.gym.find_actor_rigid_body_index(
            env_ptr, franka_handle, "panda_leftfinger", gymapi.DOMAIN_ACTOR
        )
        self.right_finger_body_id_env_actor = self.gym.find_actor_rigid_body_index(
            env_ptr, franka_handle, "panda_rightfinger", gymapi.DOMAIN_ACTOR
        )
        self.fingertip_centered_body_id_env_actor = (
            self.gym.find_actor_rigid_body_index(
                env_ptr, franka_handle, "panda_fingertip_centered", gymapi.DOMAIN_ACTOR
            )
        )

    def _acquire_task_tensors(self):
        """Acquire and wrap tensors. Create views."""

        self.object_pos = self.root_pos[:, self.object_actor_id_env, 0:3]
        self.object_quat = self.root_quat[:, self.object_actor_id_env, 0:4]
        self.object_linvel = self.root_linvel[:, self.object_actor_id_env, 0:3]
        self.object_angvel = self.root_angvel[:, self.object_actor_id_env, 0:3]

        self.object_rest_height = torch.zeros((self.num_envs,), device=self.device) + self.cfg_base.env.table_height + self.box_size / 2
        self.init_object_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self.dist_xy = torch.zeros(self.num_envs, device=self.device)  # distance between current object xy and initial object xy

        # gripper keypoints
        self.keypoint_offsets = get_keypoint_offsets_6d(self.device) * self.cfg_task.rl.gripper_keypoint_scale
        self.keypoints_gripper = torch.zeros((self.num_envs, self.keypoint_offsets.shape[0], 3),
                                            dtype=torch.float32,
                                            device=self.device)
        self.keypoints_target = torch.zeros_like(self.keypoints_gripper, device=self.device)
        self.identity_quat = (
            torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )

        # encourage including the object in wrist camera view
        if self.cfg_task.rl.enable_object_in_view_reward:
            # sample points on the cube
            cube_mesh = tm.creation.box(extents=[self.box_size,] * 3).subdivide()
            self.object_mesh_points = torch.from_numpy(cube_mesh.vertices).to(self.device)      # (n_points, 3)
            
            # wrist camera position and direction in the frame of Franka hand
            self.wrist_camera_pos_hand_frame = torch.tensor(
                self.cfg_task.env.local_obs.camera_offset, device=self.device
            ).float().repeat(self.num_envs, 1)
            self.wrist_camera_direction_hand_frame = torch.tensor(
                [-np.sin(self.cfg_task.env.local_obs.camera_angle), 0.0, np.cos(self.cfg_task.env.local_obs.camera_angle)], device=self.device
            ).float().repeat(self.num_envs, 1)

            # evaluate how aligned the object is with the central axis of the wrist camera
            self.axis_object_alignment = torch.zeros((self.num_envs,), device=self.device)

    def _refresh_task_tensors(self):
        """Refresh tensors."""
        self.dist_xy = torch.norm(self.object_pos[:, :2] - self.init_object_xy, dim=-1)

        target_pos = self.object_pos
        down_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        target_rot = quat_mul(down_q, quat_conjugate(self.object_quat))

        for idx, keypoint_offset in enumerate(self.keypoint_offsets):
            _, self.keypoints_gripper[:, idx] = tf_combine(self.fingertip_centered_quat,
                                                            self.fingertip_centered_pos,
                                                            self.identity_quat,
                                                            keypoint_offset.repeat(self.num_envs, 1))
            _, self.keypoints_target[:, idx] = tf_combine(target_rot,
                                                            target_pos,
                                                            self.identity_quat,
                                                            keypoint_offset.repeat(self.num_envs, 1))

        if self.cfg_task.rl.enable_object_in_view_reward:
            wrist_camera_pos = self.hand_pos + quat_rotate(self.hand_quat, self.wrist_camera_pos_hand_frame)
            wrist_camera_dir = quat_rotate(self.hand_quat, self.wrist_camera_direction_hand_frame)

            wrist_camera_pos_object_frame = quat_rotate(quat_conjugate(self.object_quat), wrist_camera_pos - self.object_pos)   # (n_envs, 3)
            wrist_camera_dir_object_frame = quat_rotate(quat_conjugate(self.object_quat), wrist_camera_dir)                     # (n_envs, 3)

            self.axis_object_alignment[:] = compute_axis_object_alignment(
                wrist_camera_pos_object_frame, wrist_camera_dir_object_frame, self.object_mesh_points
            )

    def _check_lift_success(self, lift_height=0.10):
        lift_success = torch.where(
            self.object_pos[:, 2] > lift_height + self.object_rest_height,
            torch.ones((self.num_envs,), device=self.device),
            torch.zeros((self.num_envs,), device=self.device))

        return lift_success

    def hardcode_control(self, get_camera_images=False):
        camera_images = []

        # close gripper
        self.ctrl_target_fingertip_midpoint_pos[:] = self.fingertip_centered_pos
        self.ctrl_target_fingertip_midpoint_quat[:] = self.fingertip_centered_quat
        images = self.move_gripper_to_target_pose(gripper_dof_pos=-0.04, sim_steps=40, get_camera_images=get_camera_images)
        camera_images.extend(images)

        # lift up
        self.ctrl_target_fingertip_midpoint_pos[:, -1] += 0.3
        images = self.move_gripper_to_target_pose(gripper_dof_pos=-0.04, sim_steps=60, get_camera_images=get_camera_images)
        camera_images.extend(images)

        return camera_images

    def compute_reward(self):
        """Compute rewards."""
        dist_gripper_reward_temp = self.reward_settings["dist_gripper_reward_temp"]
        dist_xy_reward_temp = self.reward_settings["dist_xy_reward_temp"]
        object_in_view_reward_temp = self.reward_settings["object_in_view_reward_temp"]
        object_in_view_reward_threshold = self.reward_settings["object_in_view_reward_threshold"]
        success_bonus = self.reward_settings["success_bonus"]

        # encourage the gripper to reach a valid grasp pose
        keypoint_dist = torch.mean(torch.norm(self.keypoints_gripper - self.keypoints_target, p=2, dim=-1), dim=-1)
        dist_gripper_reward = (
            torch.exp(dist_gripper_reward_temp * keypoint_dist) 
            + torch.exp(10 * dist_gripper_reward_temp * keypoint_dist) 
            + torch.exp(100 * dist_gripper_reward_temp * keypoint_dist) 
            + torch.exp(1000 * dist_gripper_reward_temp * keypoint_dist)
        ) / 4.0

        # penalize the gripper for moving away from the object
        dist_xy_reward = torch.exp(dist_xy_reward_temp * self.dist_xy)

        # encourage keeping the object in wrist camera view
        if self.cfg_task.rl.enable_object_in_view_reward:
            alignment = torch.clamp(self.axis_object_alignment - object_in_view_reward_threshold, min=0.0)
            object_in_view_reward = torch.exp(object_in_view_reward_temp * alignment)
        else:
            object_in_view_reward = torch.ones_like(dist_gripper_reward)

        total_reward = dist_gripper_reward * dist_xy_reward * object_in_view_reward

        self.rew_buf[:] = total_reward
        reset = (self.progress_buf >= self.max_episode_length - 1)

        if self.render_hardcode_control:
            self.extras['hardcode_images'] = []

        # In this policy, episode length is constant across all envs
        is_last_step = (self.progress_buf[0] == self.max_episode_length - 1)
        if is_last_step and not self.disable_hardcode_control:
            # log distance
            dist = torch.norm(self.object_pos - self.fingertip_centered_pos, p=2, dim=-1)
            self.extras['dist'] = dist
            self.extras['keypoint_dist'] = keypoint_dist.mean()
            # teleportation: close gripper and lift up
            hardcode_images = self.hardcode_control(get_camera_images=self.render_hardcode_control)
            if self.render_hardcode_control:
                self.extras['hardcode_images'] = hardcode_images
            # check lift success
            lift_success = self._check_lift_success(lift_height=0.10)
            self.rew_buf[:] += lift_success * success_bonus
            self.success_buf[:] = lift_success
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

        obs_tensors = [
            local_eef_pos, 
            local_eef_quat,
            self.fingertip_centered_linvel,
            self.fingertip_centered_angvel,
            self.object_pos,
            self.object_quat,
            self.init_object_xy,
        ]

        self.obs_buf = torch.cat(obs_tensors, dim=-1)

        return self.obs_buf
    
    def _reset_object(self, env_ids, object_init_pose=None):
        """Reset pose of the object.
        Args:
            env_ids: environment ids to reset
            object_init_pose: presampled initial pose of the object. sample if None
        """
        if object_init_pose is not None:
            init_pos = object_init_pose[:, :3]
            init_rot = object_init_pose[:, 3:]
        else:
            num_resets = len(env_ids)
            init_pos = (
                2 * torch.rand((num_resets, 3), device=self.device) - 1.0
            )  # position
            init_pos[:, 0] = (
                self.object_pos_center[0] + init_pos[:, 0] * self.object_pos_noise
            )
            init_pos[:, 1] = (
                self.object_pos_center[1] + init_pos[:, 1] * self.object_pos_noise
            )
            init_pos[:, 2] = self.object_pos_center[2]

            theta_half = (
                (2 * torch.rand(num_resets, device=self.device) - 1) * math.pi / 8
            )  # rotation along z-axis
            init_rot = torch.zeros((num_resets, 4), device=self.device)
            init_rot[:, 2] = torch.sin(theta_half)
            init_rot[:, 3] = torch.cos(theta_half)

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

        self.init_object_xy[env_ids] = init_pos[:, :2]

    def reset_idx(self, env_ids: torch.Tensor, validation_set: bool = False):
        """Reset environments having the provided indices.
        Args:
            env_ids: environments to reset
        """
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

        self._reset_franka(env_ids, franka_init_dof_pos)
        self._reset_object(env_ids, object_init_pose)
        self.simulate_and_refresh()

        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        self.prev_visual_obs = None
        self.prev_seg_obs = None
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def pre_steps(self, obs, keep_runs, num_pre_steps=120):
        """Teleport the gripper to somewhere above the cube. 
        The function generates easier initial states to test the policy.
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

        xy_offset = (2 * torch.rand_like(self.fingertip_centered_pos[:, :2]) - 1) * 0.00
        for step in range(num_pre_steps):
            target_pos = self.object_pos + torch.tensor([0.0, 0.0, 0.07], device=self.device)
            target_pos[:, :2] += xy_offset
            target_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(
                self.num_envs, 1
            )

            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_centered_pos,
                fingertip_midpoint_quat=self.fingertip_centered_quat,
                ctrl_target_fingertip_midpoint_pos=target_pos,
                ctrl_target_fingertip_midpoint_quat=target_rot,
                jacobian_type=self.cfg_ctrl['jacobian_type'],
                rot_error_type='axis_angle')

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)
            actions = delta_hand_pose * 5.0

            obs, reward, done, info = self.step(actions)
            self.progress_buf[:] = 0

        self.capture_video = capture_video
        return obs, keep_runs
