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
from manipgen.envs.franka_pick import (
    transform_keypoints_6d, 
    compute_axis_object_alignment,
    compute_nearest_keypoint_6d_distance,
)
from manipgen.utils.geometry_utils import get_keypoint_offsets_6d
import isaacgymenvs.tasks.factory.factory_control as fc

class FrankaGraspHandle(FrankaArticulatedEnv):
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
            "gripper_direction_reward_temp": self.cfg["rl"]["gripper_direction_reward_temp"],
            "gripper_direction_threshold": self.cfg["rl"]["gripper_direction_threshold"],
            "eef_height_consistency_reward_temp": self.cfg["rl"]["eef_height_consistency_reward_temp"],
            "object_in_view_reward_temp": self.cfg["rl"]["object_in_view_reward_temp"],
            "object_in_view_reward_threshold": self.cfg["rl"]["object_in_view_reward_threshold"],
            "success_bonus": self.cfg["rl"]["success_bonus"],
            "success_threshold": self.cfg["rl"]["success_threshold"],
        }

        # Reset all environments
        self.object_id = -1
        self.reset_idx(
            torch.arange(self.num_envs, device=self.device),
            switch_object=True,
            init_states=init_states,
        )

    def create_envs(self):
        """Set env options. Import assets. Create actors."""

        lower = gymapi.Vec3(-self.cfg_base.env.env_spacing, -self.cfg_base.env.env_spacing, 0.0)
        upper = gymapi.Vec3(self.cfg_base.env.env_spacing, self.cfg_base.env.env_spacing, self.cfg_base.env.env_spacing)
        num_per_row = int(np.sqrt(self.num_envs,))

        # import assets
        self.print_sdf_warning()
        franka_asset, table_asset = self.import_franka_assets()
        object_assets = self.import_partnet_assets()
        
        # create actors
        self._create_actors(lower, upper, num_per_row, franka_asset, table_asset, object_assets)

    def _create_actors(self, lower, upper, num_per_row, franka_asset, table_asset, object_assets):
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
        object_pose.p.z = 0.15
        object_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.env_ptrs = []
        self.franka_handles = []
        self.table_handles = []
        self.object_handles_multitask = [[] for _ in range(self.num_objects)]
        self.shape_ids = []
        self.franka_actor_ids_sim = []  # within-sim indices
        self.table_actor_ids_sim = []  # within-sim indices
        self.object_actor_ids_sim_multitask = [[] for _ in range(self.num_objects)]  # within-sim indices
        self.franka_rigid_body_ids_sim = [] # within-sim indices
        self.franka_left_finger_ids_sim = [] # within-sim indices
        self.franka_right_finger_ids_sim = [] # within-sim indices
        self.handle_rigid_body_ids_sim_multitask = [[] for _ in range(self.num_objects)] # within-sim indices
        actor_count = 0

        for i in range(self.num_envs):
            # create env
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # create actors
            if self.cfg_env.sim.disable_franka_collisions:
                franka_handle = self.gym.create_actor(
                    env_ptr, franka_asset, franka_pose, 'franka', i + self.num_envs, 1, 1
                )
            else:
                franka_handle = self.gym.create_actor(
                    env_ptr, franka_asset, franka_pose, 'franka', i, 1, 1
                )
            self.franka_actor_ids_sim.append(actor_count)
            self.franka_handles.append(franka_handle)
            actor_count += 1

            table_handle = self.gym.create_actor(
                env_ptr, table_asset, table_pose, "table", i, 2, 2
            )
            self.table_actor_ids_sim.append(actor_count)
            self.table_handles.append(table_handle)
            actor_count += 1

            for object_id, object_asset in enumerate(object_assets):
                # disable object-object and object-table collision
                object_handle = self.gym.create_actor(
                    env_ptr, object_asset, object_pose, f"object_{object_id}", i, 2, 4
                )
                self.object_actor_ids_sim_multitask[object_id].append(actor_count)
                self.object_handles_multitask[object_id].append(object_handle)
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

            for object_id in range(self.num_objects):
                object_handle = self.object_handles_multitask[object_id][-1]
                object_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle)
                object_shape_props[0].friction = self.cfg_env.env.object_friction
                object_shape_props[0].rolling_friction = 0.0  # default = 0.0
                object_shape_props[0].torsion_friction = 0.0  # default = 0.0
                object_shape_props[0].restitution = 0.0  # default = 0.0
                object_shape_props[0].compliance = 0.0  # default = 0.0
                object_shape_props[0].thickness = 0.0  # default = 0.0
                self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, object_shape_props)

                # set handle color
                handle_color = gymapi.Vec3(0.6, 0.1, 0.0)
                handle_idx_actor_domain = self.gym.find_actor_rigid_body_index(
                    env_ptr, object_handle, "handle_link", gymapi.DOMAIN_ACTOR
                )
                self.gym.set_rigid_body_color(
                    env_ptr, object_handle, handle_idx_actor_domain, gymapi.MESH_VISUAL_AND_COLLISION, handle_color
                )
                # set skeleton color
                skeleton_color = gymapi.Vec3(0.57, 0.71, 0.81)
                skeleton_idx_actor_domain = self.gym.find_actor_rigid_body_index(
                    env_ptr, object_handle, "skeleton", gymapi.DOMAIN_ACTOR
                )
                self.gym.set_rigid_body_color(
                    env_ptr, object_handle, skeleton_idx_actor_domain, gymapi.MESH_VISUAL_AND_COLLISION, skeleton_color
                )

            # rigid body indices for contact checking
            franka_link_names = self.gym.get_actor_rigid_body_names(env_ptr, franka_handle)
            for link_name in franka_link_names:
                link_idx = self.gym.find_actor_rigid_body_index(
                    env_ptr, franka_handle, link_name, gymapi.DOMAIN_SIM
                )
                self.franka_rigid_body_ids_sim.append(link_idx)
                if link_name == "panda_leftfinger":
                    self.franka_left_finger_ids_sim.append(link_idx)
                elif link_name == "panda_rightfinger":
                    self.franka_right_finger_ids_sim.append(link_idx)
            for object_id in range(self.num_objects):
                object_handle = self.object_handles_multitask[object_id][-1]
                handle_idx_actor_domain = self.gym.find_actor_rigid_body_index(
                    env_ptr, object_handle, "handle_link", gymapi.DOMAIN_ACTOR
                )
                handle_rigid_body_id_sim = self.gym.get_actor_rigid_body_index(
                    env_ptr, object_handle, handle_idx_actor_domain, gymapi.DOMAIN_SIM
                )
                self.handle_rigid_body_ids_sim_multitask[object_id].append(handle_rigid_body_id_sim)

            self.franka_num_dofs = self.gym.get_actor_dof_count(env_ptr, franka_handle)
            self.object_num_dofs = self.gym.get_actor_dof_count(env_ptr, object_handle)
            assert self.object_num_dofs == 1, "Only one DOF is supported for the object."

            self.gym.enable_actor_dof_force_sensors(env_ptr, franka_handle)

            self.env_ptrs.append(env_ptr)

        self.num_actors = int(actor_count / self.num_envs)  # per env
        self.num_bodies = self.gym.get_env_rigid_body_count(env_ptr)  # per env
        self.num_dofs = self.gym.get_env_dof_count(env_ptr)  # per env

        # For setting targets
        self.franka_actor_ids_sim = torch.tensor(self.franka_actor_ids_sim, dtype=torch.int32, device=self.device)
        self.table_actor_ids_sim = torch.tensor(self.table_actor_ids_sim, dtype=torch.int32, device=self.device)
        self.object_actor_ids_sim_multitask = [
            torch.tensor(object_actor_ids_sim, dtype=torch.int32, device=self.device)
            for object_actor_ids_sim in self.object_actor_ids_sim_multitask
        ]

        # For extracting root pos/quat
        self.franka_actor_id_env = self.gym.find_actor_index(env_ptr, 'franka', gymapi.DOMAIN_ENV)
        self.object_actor_id_env_multitask = [
            self.gym.find_actor_index(env_ptr, f"object_{object_id}", gymapi.DOMAIN_ENV)
            for object_id in range(self.num_objects)
        ]

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
        self.left_fingertip_body_id_env = self.gym.find_actor_rigid_body_index(
            env_ptr, franka_handle, "panda_leftfinger_tip", gymapi.DOMAIN_ENV
        )
        self.right_fingertip_body_id_env = self.gym.find_actor_rigid_body_index(
            env_ptr, franka_handle, "panda_rightfinger_tip", gymapi.DOMAIN_ENV
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
        self.up_v = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)
        self.gripper_direction_local = torch.zeros(self.num_envs, 3, device=self.device)
        self.prev_object_dof_pos = torch.zeros(self.num_envs, self.object_num_dofs, device=self.device)
        self.object_dof_diff = torch.zeros(self.num_envs, device=self.device)
        self.prev_eef_height = torch.zeros(self.num_envs, device=self.device)
        self.eef_height = torch.zeros(self.num_envs, device=self.device)
        self.eef_height_diff = torch.zeros(self.num_envs, device=self.device)

        # override in reset
        self.init_object_dof_pos = torch.zeros(self.num_envs, self.object_num_dofs, device=self.device)
        self.init_eef_height = torch.zeros(self.num_envs, device=self.device)

        # encourage including the handle in wrist camera view
        if self.cfg_task.rl.get("enable_object_in_view_reward", False):
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
        self.object_keypoints = self._compute_keypoints_pos()

        self.dist_object_dof_pos = torch.abs(self.object_dof_pos - self.init_object_dof_pos).mean(dim=-1)
        self.dist_object_gripper = self._compute_nearest_grasp_pose_keypoint_distance()

        gripper_direction = quat_rotate(self.fingertip_centered_quat, self.up_v)
        self.gripper_direction_local = quat_rotate(quat_conjugate(self.handle_quat), gripper_direction)

        self.eef_height = self.fingertip_centered_pos[:, 2]
        self.eef_height_diff = (self.eef_height - self.prev_eef_height).abs()
        self.prev_eef_height[:] = self.eef_height.clone()

        self.object_dof_diff = (self.object_dof_pos - self.prev_object_dof_pos).abs().mean(dim=-1) \
            / torch.abs(self.target_object_dof_pos - self.rest_object_dof_pos)
        self.prev_object_dof_pos[:] = self.object_dof_pos.clone()
        
        if self.cfg_task.rl.get("enable_object_in_view_reward", False):
            wrist_camera_pos = self.hand_pos + quat_rotate(self.hand_quat, self.wrist_camera_pos_hand_frame)
            wrist_camera_dir = quat_rotate(self.hand_quat, self.wrist_camera_direction_hand_frame)

            wrist_camera_pos_object_frame = quat_rotate(quat_conjugate(self.handle_quat), wrist_camera_pos - self.handle_pos)   # (n_envs, 3)
            wrist_camera_dir_object_frame = quat_rotate(quat_conjugate(self.handle_quat), wrist_camera_dir)                     # (n_envs, 3)

            self.axis_object_alignment[:] = compute_axis_object_alignment(
                wrist_camera_pos_object_frame, wrist_camera_dir_object_frame, self.object_mesh_points
            )

    def _compute_nearest_grasp_pose_keypoint_distance(self):
        """Based on current gripper keypoints, compute distance to nearest valid grasp pose keypoints."""
        # convert gripper pose to object local frame
        current_grasp_pos_local = quat_rotate(quat_conjugate(self.handle_quat), self.fingertip_centered_pos - self.handle_pos)
        current_grasp_rot_local = quat_mul(quat_conjugate(self.handle_quat), self.fingertip_centered_quat)
        current_grasp_pose_local = torch.cat([current_grasp_pos_local, current_grasp_rot_local], dim=-1)
        current_grasp_keypoints_local = transform_keypoints_6d(current_grasp_pose_local, self.gripper_keypoint_offsets)
        # compute distance to nearest grasp pose based on keypoints
        dist_grasp_keypoints = compute_nearest_keypoint_6d_distance(current_grasp_keypoints_local, self.target_grasp_keypoints_local)
        
        return dist_grasp_keypoints
    
    def _compute_keypoints_pos(self):
        num_object_poses = self.handle_pos.shape[0]
        key_points_world = quat_rotate(
            self.handle_quat.unsqueeze(1).expand(-1, self.num_object_keypoints, -1).reshape(-1, 4),
            self.object_keypoints_local.expand(num_object_poses, -1, -1).reshape(-1, 3),
        ).reshape(num_object_poses, self.num_object_keypoints, 3) + self.handle_pos.unsqueeze(1)
        key_points_world = key_points_world.reshape(num_object_poses, -1)
        return key_points_world
    
    def _check_grasp_success(self):
        grasp_success = self.gripper_dof_pos[:, :].sum(dim=-1) > self.reward_settings["success_threshold"]
        return grasp_success

    def hardcode_control(self, get_camera_images=False):
        camera_images = []

        # teleportation: close gripper
        self.ctrl_target_fingertip_midpoint_pos[:] = self.fingertip_centered_pos
        self.ctrl_target_fingertip_midpoint_quat[:] = self.fingertip_centered_quat
        images = self.move_gripper_to_target_pose(gripper_dof_pos=-0.04, sim_steps=60, get_camera_images=get_camera_images)
        camera_images.extend(images)
        
        # teloportation: pull the door
        self.enable_gravity((5.0, 0.0, 0.0))    # change direction of gravity
        front_v = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        direction = quat_rotate(self.handle_quat, front_v)
        distance = 0.03
        self.ctrl_target_fingertip_midpoint_pos[:] = self.fingertip_centered_pos + distance * direction
        self.ctrl_target_fingertip_midpoint_quat[:] = self.fingertip_centered_quat
        images = self.move_gripper_to_target_pose(gripper_dof_pos=-0.04, sim_steps=60, get_camera_images=get_camera_images)
        camera_images.extend(images)
        self.enable_gravity()                   # change gravity back to default

        return camera_images

    def compute_reward(self):
        """Compute rewards."""
        dist_object_gripper_reward_temp = self.reward_settings["dist_object_gripper_reward_temp"]
        dist_object_dof_pos_reward_temp = self.reward_settings["dist_object_dof_pos_reward_temp"]
        gripper_direction_reward_temp = self.reward_settings["gripper_direction_reward_temp"]
        gripper_direction_threshold = self.reward_settings["gripper_direction_threshold"]
        eef_height_consistency_reward_temp = self.reward_settings["eef_height_consistency_reward_temp"]
        object_in_view_reward_temp = self.reward_settings["object_in_view_reward_temp"]
        object_in_view_reward_threshold = self.reward_settings["object_in_view_reward_threshold"]
        success_bonus = self.reward_settings["success_bonus"]

        # component 1: keypoint distance between current gripper pose and nearest pre-sampled grasp pose
        # encourage the gripper to move to the nearest grasp pose
        dist_gripper_reward = (
            torch.exp(dist_object_gripper_reward_temp * self.dist_object_gripper)
            + torch.exp(dist_object_gripper_reward_temp * self.dist_object_gripper * 10)
            + torch.exp(dist_object_gripper_reward_temp * self.dist_object_gripper * 100)
            + torch.exp(dist_object_gripper_reward_temp * self.dist_object_gripper * 1000)
        ) / 4

        # component 2: change in object dof pos
        # grasping the handle should not move the door
        dist_dof_reward = dist_object_dof_pos_reward_temp * self.object_dof_diff
        
        # component 3: angle between gripper and door
        # encourage the gripper to be perpendicular to the door (with some threshold)
        gripper_direction_reward = torch.exp(
            gripper_direction_reward_temp * torch.clamp(-self.gripper_direction_local[:, 0] - gripper_direction_threshold, max=0.0)
        )

        # component 4: change in eef height
        # this term is to penalize the gripper from moving up and down
        # we find without this term, the gripper tends to move up even if current grasp is successful
        eef_height_consistency_reward = eef_height_consistency_reward_temp * self.eef_height_diff

        # (deprecated) encourage including the object in wrist camera view
        if self.cfg_task.rl.get("enable_object_in_view_reward", False):
            alignment = torch.clamp(self.axis_object_alignment - object_in_view_reward_threshold, min=0.0)
            object_in_view_reward = torch.exp(object_in_view_reward_temp * alignment)
        else:
            object_in_view_reward = torch.ones_like(dist_gripper_reward)

        total_reward = dist_gripper_reward * gripper_direction_reward * object_in_view_reward + eef_height_consistency_reward + dist_dof_reward

        self.rew_buf[:] = total_reward
        reset = (self.progress_buf == self.max_episode_length - 1)

        if self.render_hardcode_control:
            self.extras['hardcode_images'] = []

        is_last_step = (self.progress_buf[0] == self.max_episode_length - 1)
        if is_last_step and not self.disable_hardcode_control:
            self.extras["keypoint_dist"] = self.dist_object_gripper.mean()
            self.extras["eef_height_diff"] = (self.eef_height - self.init_eef_height).abs().mean()
            self.extras["dof_offset_ratio"] = (self.dist_object_dof_pos / torch.abs(self.target_object_dof_pos - self.rest_object_dof_pos)).mean()
            hardcode_images = self.hardcode_control(get_camera_images=self.render_hardcode_control)
            if self.render_hardcode_control:
                self.extras['hardcode_images'] = hardcode_images
            # check grasp success and add success bonus
            grasp_success = self._check_grasp_success()
            self.rew_buf[:] += grasp_success * success_bonus
            self.success_buf[:] = grasp_success
            self.consecutive_successes[:] = torch.where(
                reset > 0, self.success_buf * reset, self.consecutive_successes
            ).mean()

            # log success rate for different init dof ratio
            init_dof_ratio = (torch.abs(self.init_object_dof_pos - self.rest_object_dof_pos) \
                / torch.abs(self.target_object_dof_pos - self.rest_object_dof_pos)).mean(dim=-1)
            for idx, (lower, upper) in enumerate(((0.0, 0.3), (0.3, 0.6), (0.6, 1.0))):
                select = (lower <= init_dof_ratio) & (init_dof_ratio < upper)
                num_select = select.sum()
                success = 0.0
                if num_select > 0:
                    success = grasp_success[select].sum() / num_select
                self.extras[f"success_init_dof_ratio_{idx}"] = success

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

        grasp = (self.progress_buf == self.max_episode_length - 1).float()
        if self.disable_hardcode_control:
            grasp[:] = 0.0
        obs_tensors = [
            local_eef_pos, 
            local_eef_quat,
            self.fingertip_centered_linvel,                 # linear velocity of the gripper
            self.fingertip_centered_angvel,                 # angular velocity of the gripper
            self.handle_pos,                                # position of the handle
            self.handle_quat,                               # orientation of the handle
            self.object_dof_pos,                            # dof pos of the object
            self.object_keypoints,                          # position of keypoints of the object
            self.object_dof_diff.unsqueeze(-1),             # change in object dof pos
            self.dist_object_gripper.unsqueeze(-1),         # distance between object and gripper
            self.eef_height_diff.unsqueeze(-1),             # change in eef height
            grasp.unsqueeze(-1),                            # whether to close the gripper with hard-code control
        ]

        self.obs_buf = torch.cat(obs_tensors, dim=-1)
    
        return self.obs_buf

    def _set_object_init_and_target_dof_pos(self):
        """ Set initial and target object dof positions """
        self.rest_object_dof_pos = self.object_dof_lower_limits.clone()
        self.target_object_dof_pos = self.object_dof_upper_limits.clone()

    def switch_object(self, init_states):
        if self.object_id != -1:    
            # if not the first switch, reset the current object to the initial state
            self.object_pos[:, 0] = 0.0
            self.object_pos[:, 1] = 0.0
            self.object_pos[:, 2] = 0.15
            self.object_quat[:, :] = 0.0
            self.object_quat[:, 3] = 1.0
            self.object_linvel[:, :] = 0.0
            self.object_angvel[:, :] = 0.0
            self.object_dof_pos[:, :] = 0.0
            self.object_dof_vel[:, :] = 0.0

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
            if f'{self.object_code}_success' in self.extras:
                del self.extras[f'{self.object_code}_success']
        else:
            self.episode_cur_multitask = [0 for _ in range(self.num_objects)]

        self.object_id = (self.object_id + 1) % self.num_objects
        self.object_code = self.object_codes[self.object_id]

        self.object_handles = self.object_handles_multitask[self.object_id]
        self.object_actor_ids_sim = self.object_actor_ids_sim_multitask[self.object_id]
        self.handle_rigid_body_ids_sim = self.handle_rigid_body_ids_sim_multitask[self.object_id]
        self.object_actor_id_env = self.object_actor_id_env_multitask[self.object_id]

        # load grasp data
        self.load_partnet_data(
            self.object_codes[self.object_id],
            load_grasp_poses=True,
            load_object_keypoints=True,
            load_mesh_points=self.cfg_task.rl.get("enable_object_in_view_reward", False),
        )

        # object dof
        self.object_dof_lower_limits = self.object_dof_lower_limits_multitask[self.object_id]
        self.object_dof_upper_limits = self.object_dof_upper_limits_multitask[self.object_id]
        self._set_object_init_and_target_dof_pos()
        self._randomize_object_dof_properties()

        # object pose and velocity
        self.object_pos = self.root_pos[:, self.object_actor_id_env, 0:3]
        self.object_quat = self.root_quat[:, self.object_actor_id_env, 0:4]
        self.object_linvel = self.root_linvel[:, self.object_actor_id_env, 0:3]
        self.object_angvel = self.root_angvel[:, self.object_actor_id_env, 0:3]
        dof_start_idx = self.franka_num_dofs + self.object_id * self.object_num_dofs
        dof_end_idx = dof_start_idx + self.object_num_dofs
        self.object_dof_pos = self.dof_pos[:, dof_start_idx:dof_end_idx]
        self.object_dof_vel = self.dof_vel[:, dof_start_idx:dof_end_idx]
        self.handle_pos = self.body_pos[:, self.handle_rigid_body_ids_sim[0], 0:3]
        self.handle_quat = self.body_quat[:, self.handle_rigid_body_ids_sim[0], 0:4]
        self.handle_linvel = self.body_linvel[:, self.handle_rigid_body_ids_sim[0], 0:3]
        self.handle_angvel = self.body_angvel[:, self.handle_rigid_body_ids_sim[0], 0:3]

        # process grasp data: keypoints corresponding to pre-sampled grasp poses in the object frame
        gripper_keypoint_scale = self.cfg_task.rl.gripper_keypoint_scale
        self.gripper_keypoint_offsets = get_keypoint_offsets_6d(self.device) * gripper_keypoint_scale
        
        assert self.cfg_task.rl.gripper_keypoint_dof in (5, 6), "Invalid gripper keypoint DOF."
        if self.cfg_task.rl.gripper_keypoint_dof == 5:
            # allow rotation around y-axis in eef frame: only keep the keypoints on the y-axis
            select = torch.tensor([True, False, True, False, False, True, False]).to(self.device)
            self.gripper_keypoint_offsets = self.gripper_keypoint_offsets[select]
        
        self.target_grasp_keypoints_local = transform_keypoints_6d(self.grasp_pose, self.gripper_keypoint_offsets)

        # load initial states
        self.prepare_init_states(init_states)
        self.episode_cur = self.episode_cur_multitask[self.object_id]
    
    def _sample_object_pose(self, num_sample):
        # sample position:
        # objects are placed on the half circle centered at the robot base (with some noise to the radius)
        init_pos = torch.zeros((num_sample, 3), device=self.device)
        object_width, object_height = self.object_meta["primitive_width"], self.object_meta["primitive_height"]
        object_pos_radius = self.cfg_task.randomize.object_pos_radius
        if self.object_meta["type"] == "drawer":
            object_pos_radius += self.cfg_task.randomize.door_pos_radius_offset       # drawer is placed further away from the robot
        radius = object_pos_radius + (torch.rand((num_sample,), device=self.device) * 2 - 1) * self.cfg_task.randomize.object_pos_radius_noise
        angle = (torch.rand((num_sample,), device=self.device) * 2 - 1) * self.cfg_task.randomize.object_pos_angle_noise
        height = torch.rand((num_sample,), device=self.device) * max(0.0, self.cfg_task.randomize.object_z_max - object_height)  # prevent the object from being too high
        mesh_center_height = self.object_mesh.vertices.mean(axis=0)[2]
        height = torch.clamp(height, min=self.cfg_task.randomize.handle_z_min - mesh_center_height)     # prevent the handle from being too low

        init_pos[:, 0] = radius * torch.cos(angle) - self.cfg_base.env.franka_depth
        init_pos[:, 1] = radius * torch.sin(angle)
        init_pos[:, 2] = self.cfg_base.env.table_height + height

        # sample rotation:
        # once the object position is fixed, `angle` is the oirientation facing straight to the robot 
        # we add some noise to the orientation
        raw = torch.zeros((num_sample,), device=self.device)
        pitch = torch.zeros((num_sample,), device=self.device)
        if self.object_meta["type"] == "drawer":
            object_relative_rotation = self.cfg_task.randomize.drawer_rotation_noise
        else:
            # apply smaller rotation to door because it can rotate around z-axis
            # large rotation noise makes the door hard to open
            object_relative_rotation = self.cfg_task.randomize.door_rotation_noise
        yaw = angle + math.pi + (torch.rand((num_sample,), device=self.device) * 2 - 1) * object_relative_rotation
        init_rot = quat_from_euler_xyz(raw, pitch, yaw)

        # offset for door:
        # origin is placed on the joint axis of the door
        # we apply offset here to ensure the position sampled above is the center of the door
        if self.object_meta["type"] == "door":
            v = to_torch([1., 0., 0.], device=self.device).repeat(num_sample, 1)
            rot = quat_from_euler_xyz(raw, pitch, yaw + self.object_meta["joint_val"] * math.pi / 2)
            offset = quat_rotate(rot, v) * object_width / 2
            init_pos += offset

        return init_pos, init_rot

    def _reset_franka_and_object(
        self, 
        env_ids, 
        franka_init_dof_pos=None, 
        object_init_pose=None, 
        object_dof_pos=None
    ):
        """Reset DOF states, DOF torques, and DOF targets of Franka and articulated object.
        Args:
            env_ids: environment ids to reset
            franka_init_dof_pos: presampled DOF positions of Franka.
                use default arm and gripper pos if None
            object_init_pose: presampled object poses.
            object_dof_pos: presampled object DOF positions.
        """
        # Randomize Franka DOF pos
        if franka_init_dof_pos is not None:
            self.dof_pos[env_ids, :self.franka_num_dofs] = franka_init_dof_pos
        else:
            if self.cfg_task.randomize.franka_gripper_initial_state == 1:
                gripper_init_pos = self.gripper_dof_upper_limits
            else:
                gripper_init_pos = self.gripper_dof_lower_limits

            self.dof_pos[env_ids, :self.franka_num_dofs] = torch.cat(
                (
                    torch.tensor(
                        self.cfg_task.randomize.franka_arm_initial_dof_pos,
                        device=self.device,
                    ),
                    gripper_init_pos,
                ),
                dim=-1,
            ).unsqueeze(
                0
            )  # shape = (num_envs, num_dofs)

        # Stabilize Franka
        self.dof_vel[env_ids] = 0.0  # shape = (num_envs, num_dofs)
        self.dof_torque[env_ids] = 0.0

        self.ctrl_target_dof_pos[env_ids, :self.franka_num_dofs] = self.dof_pos[env_ids, :self.franka_num_dofs].clone()
        self.ctrl_target_fingertip_centered_pos[env_ids] = self.fingertip_centered_pos[env_ids].clone()
        self.ctrl_target_fingertip_centered_quat[env_ids] = self.fingertip_centered_quat[env_ids].clone()

        if object_init_pose is not None:
            self.object_pos[env_ids] = object_init_pose[:, 0:3]
            self.object_quat[env_ids] = object_init_pose[:, 3:7]
            self.object_dof_pos[env_ids] = object_dof_pos
        else:
            num_resets = len(env_ids)
            init_pos, init_rot = self._sample_object_pose(num_resets)
            self.object_pos[env_ids] = init_pos
            self.object_quat[env_ids] = init_rot
            self.object_dof_pos[env_ids] = self.rest_object_dof_pos + self.cfg_task.randomize.max_object_init_dof_ratio * (
                self.target_object_dof_pos - self.rest_object_dof_pos
            ) * torch.rand((len(env_ids), self.object_num_dofs), device=self.device)
        self.object_linvel[env_ids, :] = 0.0
        self.object_angvel[env_ids, :] = 0.0

        # Set DOF state
        franka_actor_ids_sim = self.franka_actor_ids_sim[env_ids].clone().to(dtype=torch.int32)
        object_actor_ids_sim = self.object_actor_ids_sim[env_ids].to(torch.int32)
        merged_actor_ids_sim = torch.cat((franka_actor_ids_sim, object_actor_ids_sim), dim=0)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(merged_actor_ids_sim),
            len(merged_actor_ids_sim),
        )

        # Set DOF torque
        self.gym.set_dof_actuation_force_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(torch.zeros_like(self.dof_torque)),
            gymtorch.unwrap_tensor(merged_actor_ids_sim),
            len(merged_actor_ids_sim),
        )

        # Set object state
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state),
            gymtorch.unwrap_tensor(object_actor_ids_sim),
            len(object_actor_ids_sim),
        )

        self.init_object_dof_pos[env_ids] = self.object_dof_pos[env_ids].clone()
        self.prev_object_dof_pos[env_ids] = self.object_dof_pos[env_ids].clone()
        self.init_eef_height[env_ids] = self.fingertip_centered_pos[env_ids, 2].clone()
        self.prev_eef_height[env_ids] = self.fingertip_centered_pos[env_ids, 2].clone()

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
                if end >= self.num_init_states - self.val_num:
                    start = 0
                    end = self.episode_cur = len(env_ids)
                    print("Warning: Out of training set. Restarting from the beginning.")
            franka_init_dof_pos = self.init_states["franka_dof_pos"][start:end]
            object_init_pose = self.init_states["object_pose"][start:end]
            object_dof_pos = self.init_states["init_object_dof_pos"][start:end]
        else:
            franka_init_dof_pos = None
            object_init_pose = None
            object_dof_pos = None

        self._reset_franka_and_object(env_ids, franka_init_dof_pos, object_init_pose, object_dof_pos)
        self.simulate_and_refresh()

        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        self.prev_visual_obs = None
        self.prev_seg_obs = None
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def pre_steps(self, obs, keep_runs, num_pre_steps=100):
        """Teleport the gripper to somewhere close to the handle. 
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

        # sample target eef pose in handle frame: point towards to board
        handle_mesh_center = (self.object_mesh.vertices.max(0) + self.object_mesh.vertices.min(0)) / 2
        yz_offset = (2 * torch.rand((self.num_envs, 2), device=self.device) - 1) * 0.00
        target_pos_local = torch.zeros(self.num_envs, 3, device=self.device)
        target_pos_local[:, 0] = 0.08
        target_pos_local[:, 1] = handle_mesh_center[1] + yz_offset[:, 0]
        target_pos_local[:, 2] = handle_mesh_center[2] + yz_offset[:, 1]

        raw = torch.zeros((self.num_envs,), device=self.device)
        pitch = torch.zeros((self.num_envs,), device=self.device) - math.pi / 2
        yaw = torch.zeros((self.num_envs,), device=self.device)
        target_pose_local = quat_from_euler_xyz(raw, pitch, yaw)

        # apply rotation
        raw = torch.zeros((self.num_envs,), device=self.device) + math.pi / 2
        pitch = torch.zeros((self.num_envs,), device=self.device)
        yaw = torch.zeros((self.num_envs,), device=self.device)
        rot = quat_from_euler_xyz(raw, pitch, yaw)
        target_pose_local = quat_mul(rot, target_pose_local)

        for step in range(num_pre_steps):
            # transform target pose to world frame
            target_pos = quat_rotate(self.handle_quat, target_pos_local) + self.handle_pos
            target_rot = quat_mul(self.handle_quat, target_pose_local)

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

        # check contact
        valid = self.dist_object_dof_pos < 0.02
        keep_runs = keep_runs & valid

        self.capture_video = capture_video
        return obs, keep_runs

    