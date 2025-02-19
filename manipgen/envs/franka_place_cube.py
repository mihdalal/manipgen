from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch

from gym import spaces
import numpy as np
import torch
import trimesh as tm

from typing import Dict, Any, Tuple, List, Set
import math

from isaacgymenvs.utils.torch_jit_utils import (
    to_torch, 
    quat_mul, 
    quat_conjugate,
    quat_rotate,
    tf_combine,
)
import isaacgymenvs.tasks.factory.factory_control as fc
from manipgen.envs.franka_env import FrankaEnv
from manipgen.envs.franka_pick_cube import FrankaPickCube
from manipgen.envs.franka_pick import compute_axis_object_alignment
from manipgen.utils.geometry_utils import get_keypoint_offsets_6d

class FrankaPlaceCube(FrankaEnv):
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
            "dist_object_reward_temp": self.cfg["rl"]["dist_object_reward_temp"],
            "dist_xy_reward_temp": self.cfg["rl"]["dist_xy_reward_temp"],
            "object_in_view_reward_temp": self.cfg["rl"]["object_in_view_reward_temp"],
            "object_in_view_reward_threshold": self.cfg["rl"]["object_in_view_reward_threshold"],
            "success_xy_threshold": self.cfg["rl"]["success_xy_threshold"],
            "success_z_threshold": self.cfg["rl"]["success_z_threshold"],
            "success_bonus": self.cfg["rl"]["success_bonus"],
        }

        # initial states loading
        self.prepare_init_states(init_states)

        # initial state sampling & reset
        self.receptacle_pos_radius = 0.2
        self.object_pos_noise = 0.2
        self.object_pos_center = [
            self.table_pose.p.x,
            self.table_pose.p.y,
            self.cfg_base.env.table_height + 0.5 * self.box_size + self.receptacle_size,
        ]

        # Reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    _import_cube_asset = FrankaPickCube._import_cube_asset

    def _import_receptacle_asset(self):
        """Import cube asset."""
        asset_options = gymapi.AssetOptions()
        self.receptacle_size = self.cfg_task.env.receptacle_size
        receptacle_asset = self.gym.create_box(
            self.sim, self.receptacle_size, self.receptacle_size, self.receptacle_size, asset_options
        )
        return receptacle_asset
    
    def create_envs(self):
        """Set env options. Import assets. Create actors."""

        lower = gymapi.Vec3(-self.cfg_base.env.env_spacing, -self.cfg_base.env.env_spacing, 0.0)
        upper = gymapi.Vec3(self.cfg_base.env.env_spacing, self.cfg_base.env.env_spacing, self.cfg_base.env.env_spacing)
        num_per_row = int(np.sqrt(self.num_envs))

        # import assets
        self.print_sdf_warning()
        franka_asset, table_asset = self.import_franka_assets()
        object_asset = self._import_cube_asset()
        receptacle_asset = self._import_receptacle_asset()
        
        # create actors
        self._create_actors(lower, upper, num_per_row, franka_asset, table_asset, object_asset, receptacle_asset)
    
    def _create_actors(self, lower, upper, num_per_row, franka_asset, table_asset, object_asset, receptacle_asset):
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

        receptacle_pose = gymapi.Transform()
        receptacle_pose.p.x = 0.1
        receptacle_pose.p.y = 0.0
        receptacle_pose.p.z = self.cfg_base.env.table_height + 0.5 * self.receptacle_size
        receptacle_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.env_ptrs = []
        self.franka_handles = []
        self.table_handles = []
        self.object_handles = []
        self.receptacle_handles = []
        self.shape_ids = []
        self.franka_actor_ids_sim = []  # within-sim indices
        self.table_actor_ids_sim = []  # within-sim indices
        self.object_actor_ids_sim = []  # within-sim indices
        self.receptacle_actor_ids_sim = []  # within-sim indices
        self.franka_rigid_body_ids_sim = [] # within-sim indices
        self.receptacle_rigid_body_ids_sim = [] # within-sim indices
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
            actor_count += 1

            table_handle = self.gym.create_actor(
                env_ptr, table_asset, table_pose, "table", i, 2, 2
            )
            self.table_actor_ids_sim.append(actor_count)
            actor_count += 1

            object_handle = self.gym.create_actor(
                env_ptr, object_asset, object_pose, "object", i, 4, 3
            )
            self.object_actor_ids_sim.append(actor_count)
            actor_count += 1

            receptacle_handle = self.gym.create_actor(
                env_ptr, receptacle_asset, receptacle_pose, "receptacle", i, 
                2 if self.sample_mode else 8,              # disable collision between receptacle and table in sample mode
                4
            )
            self.receptacle_actor_ids_sim.append(actor_count)
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

            receptacle_link_idx = self.gym.get_actor_rigid_body_index(
                env_ptr, receptacle_handle, 0, gymapi.DOMAIN_SIM
            )
            self.receptacle_rigid_body_ids_sim.append(receptacle_link_idx)

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

            receptacle_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, receptacle_handle)
            receptacle_shape_props[0].friction = self.cfg_env.env.receptacle_friction
            receptacle_shape_props[0].rolling_friction = 0.0  # default = 0.0
            receptacle_shape_props[0].torsion_friction = 0.0  # default = 0.0
            receptacle_shape_props[0].restitution = 0.0  # default = 0.0
            receptacle_shape_props[0].compliance = 0.0  # default = 0.0
            receptacle_shape_props[0].thickness = 0.0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, receptacle_handle, receptacle_shape_props)

            receptacle_color = gymapi.Vec3(0.0, 0.4, 0.1)
            self.gym.set_rigid_body_color(
                env_ptr, receptacle_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, receptacle_color
            )

            self.franka_num_dofs = self.gym.get_actor_dof_count(env_ptr, franka_handle)

            self.gym.enable_actor_dof_force_sensors(env_ptr, franka_handle)

            self.env_ptrs.append(env_ptr)
            self.franka_handles.append(franka_handle)
            self.table_handles.append(table_handle)
            self.object_handles.append(object_handle)
            self.receptacle_handles.append(receptacle_handle)

        self.num_actors = int(actor_count / self.num_envs)  # per env
        self.num_bodies = self.gym.get_env_rigid_body_count(env_ptr)  # per env
        self.num_dofs = self.gym.get_env_dof_count(env_ptr)  # per env

        # For setting targets
        self.franka_actor_ids_sim = torch.tensor(self.franka_actor_ids_sim, dtype=torch.int32, device=self.device)
        self.table_actor_ids_sim = torch.tensor(self.table_actor_ids_sim, dtype=torch.int32, device=self.device)
        self.object_actor_ids_sim = torch.tensor(self.object_actor_ids_sim, dtype=torch.int32, device=self.device)
        self.receptacle_actor_ids_sim = torch.tensor(self.receptacle_actor_ids_sim, dtype=torch.int32, device=self.device)

        # For extracting root pos/quat
        self.franka_actor_id_env = self.gym.find_actor_index(env_ptr, 'franka', gymapi.DOMAIN_ENV)
        self.object_actor_id_env = self.gym.find_actor_index(env_ptr, 'object', gymapi.DOMAIN_ENV)
        self.receptacle_actor_id_env = self.gym.find_actor_index(env_ptr, 'receptacle', gymapi.DOMAIN_ENV)

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

        # object pose and velocity
        self.object_pos = self.root_pos[:, self.object_actor_id_env, 0:3]
        self.object_quat = self.root_quat[:, self.object_actor_id_env, 0:4]
        self.object_linvel = self.root_linvel[:, self.object_actor_id_env, 0:3]
        self.object_angvel = self.root_angvel[:, self.object_actor_id_env, 0:3]

        # receptacle pose and velocity
        self.receptacle_pos = self.root_pos[:, self.receptacle_actor_id_env, 0:3]
        self.receptacle_quat = self.root_quat[:, self.receptacle_actor_id_env, 0:4]
        self.receptacle_linvel = self.root_linvel[:, self.receptacle_actor_id_env, 0:3]
        self.receptacle_angvel = self.root_angvel[:, self.receptacle_actor_id_env, 0:3]

        self.init_receptacle_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self.dist_xy = torch.zeros(self.num_envs, device=self.device)  # distance between current receptacle xy and initial receptacle xy

        # object keypoints
        self.keypoint_offsets = get_keypoint_offsets_6d(self.device) * self.cfg_task.rl.object_keypoint_scale
        self.keypoints_object = torch.zeros((self.num_envs, self.keypoint_offsets.shape[0], 3),
                                            dtype=torch.float32,
                                            device=self.device)
        self.keypoints_target = torch.zeros_like(self.keypoints_object, device=self.device)
        self.identity_quat = (
            torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )

        # encourage including the receptacle in wrist camera view
        if self.cfg_task.rl.enable_object_in_view_reward:
            # sample points on the cube
            cube_mesh = tm.creation.box(extents=[self.receptacle_size,] * 3).subdivide()
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
        self.dist_xy = torch.norm(self.receptacle_pos[:, :2] - self.init_receptacle_xy, dim=-1)

        target_pos = self.receptacle_pos.clone()
        target_pos[:, 2] = self.cfg_base.env.table_height + self.receptacle_size + 0.5 * self.box_size + 0.01
        target_rot = self.receptacle_quat.clone()

        for idx, keypoint_offset in enumerate(self.keypoint_offsets):
            _, self.keypoints_object[:, idx] = tf_combine(self.object_quat,
                                                            self.object_pos,
                                                            self.identity_quat,
                                                            keypoint_offset.repeat(self.num_envs, 1))
            _, self.keypoints_target[:, idx] = tf_combine(target_rot,
                                                            target_pos,
                                                            self.identity_quat,
                                                            keypoint_offset.repeat(self.num_envs, 1))
        
        if self.cfg_task.rl.enable_object_in_view_reward:
            wrist_camera_pos = self.hand_pos + quat_rotate(self.hand_quat, self.wrist_camera_pos_hand_frame)
            wrist_camera_dir = quat_rotate(self.hand_quat, self.wrist_camera_direction_hand_frame)

            wrist_camera_pos_object_frame = quat_rotate(quat_conjugate(self.receptacle_quat), wrist_camera_pos - self.receptacle_pos)   # (n_envs, 3)
            wrist_camera_dir_object_frame = quat_rotate(quat_conjugate(self.receptacle_quat), wrist_camera_dir)                         # (n_envs, 3)

            self.axis_object_alignment[:] = compute_axis_object_alignment(
                wrist_camera_pos_object_frame, wrist_camera_dir_object_frame, self.object_mesh_points
            )

    def _check_place_success(self, xy_threshold=0.015, z_threshold=0.005):
        """Check if the object is placed successfully."""
        target_height = self.cfg_base.env.table_height + self.receptacle_size + 0.5 * self.box_size
        dist_xy = torch.norm(self.object_pos[:, :2] - self.receptacle_pos[:, :2], dim=-1)
        dist_z = torch.abs(self.object_pos[:, 2] - target_height)

        return (dist_xy < xy_threshold) & (dist_z < z_threshold)
    
    def hardcode_control(self, get_camera_images=False):
        camera_images = []

        # open gripper
        self.ctrl_target_fingertip_midpoint_pos[:] = self.fingertip_centered_pos
        self.ctrl_target_fingertip_midpoint_quat[:] = self.fingertip_centered_quat
        images = self.move_gripper_to_target_pose(gripper_dof_pos=0.08, sim_steps=20, get_camera_images=get_camera_images)
        camera_images.extend(images)

        # lift up the arm
        self.ctrl_target_fingertip_midpoint_pos[:, -1] += 0.3
        images = self.move_gripper_to_target_pose(gripper_dof_pos=0.08, sim_steps=40, get_camera_images=get_camera_images)
        camera_images.extend(images)

        return camera_images

    def compute_reward(self):
        """Compute rewards."""
        dist_object_reward_temp = self.reward_settings["dist_object_reward_temp"]
        dist_xy_reward_temp = self.reward_settings["dist_xy_reward_temp"]
        object_in_view_reward_temp = self.reward_settings["object_in_view_reward_temp"]
        object_in_view_reward_threshold = self.reward_settings["object_in_view_reward_threshold"]
        success_xy_threshold = self.reward_settings["success_xy_threshold"]
        success_z_threshold = self.reward_settings["success_z_threshold"]
        success_bonus = self.reward_settings["success_bonus"]

        # reward for moving the object to target pose
        keypoint_dist = torch.mean(torch.norm(self.keypoints_object - self.keypoints_target, p=2, dim=-1), dim=-1)
        dist_object_reward = (
            torch.exp(dist_object_reward_temp * keypoint_dist) 
            + torch.exp(10 * dist_object_reward_temp * keypoint_dist) 
            + torch.exp(100 * dist_object_reward_temp * keypoint_dist) 
            + torch.exp(1000 * dist_object_reward_temp * keypoint_dist)
        ) / 4.0
        # reward for keeping the receptacle static
        dist_xy_reward = torch.exp(dist_xy_reward_temp * self.dist_xy)

        # encourage including the receptacle in wrist camera view
        if self.cfg_task.rl.enable_object_in_view_reward:
            alignment = torch.clamp(self.axis_object_alignment - object_in_view_reward_threshold, min=0.0)
            object_in_view_reward = torch.exp(object_in_view_reward_temp * alignment)
        else:
            object_in_view_reward = torch.ones_like(dist_object_reward)

        total_reward = dist_object_reward * dist_xy_reward * object_in_view_reward

        self.rew_buf[:] = total_reward
        reset = (self.progress_buf >= self.max_episode_length - 1)

        if self.render_hardcode_control:
            self.extras['hardcode_images'] = []

        # In this policy, episode length is constant across all envs
        is_last_step = (self.progress_buf[0] == self.max_episode_length - 1)
        if is_last_step and not self.disable_hardcode_control:
            # log distance
            dist = torch.norm(self.object_pos - self.keypoints_target[:, 0], p=2, dim=-1)
            self.extras['dist'] = dist.mean()
            self.extras['keypoint_dist'] = keypoint_dist.mean()
            # teleportation: open gripper and lift up the arm
            hardcode_images = self.hardcode_control(get_camera_images=self.render_hardcode_control)
            if self.render_hardcode_control:
                self.extras['hardcode_images'] = hardcode_images
            # check place success
            place_success = self._check_place_success(xy_threshold=success_xy_threshold, z_threshold=success_z_threshold)
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
            self.receptacle_pos,                            # position of the receptacle
            self.receptacle_quat,                           # orientation of the receptacle
            self.init_receptacle_xy,                        # initial xy position of the receptacle
            open_gripper.unsqueeze(-1),                     # whether to open the gripper and lift up the arm
        ]

        self.obs_buf = torch.cat(obs_tensors, dim=-1)

        return self.obs_buf

    def _reset_init_cube_state(self, name, env_ids, other_cube_pose, check_valid=True):
        # Initialize buffer to hold sampled values
        num_resets = len(env_ids)
        sampled_cube_pose = torch.zeros(num_resets, 7, device=self.device)

        # Get correct references depending on which one was selected
        if name == "object":
            cube_heights = self.box_size
        elif name == "receptacle":
            cube_heights = self.receptacle_size
        else:
            assert False

        # Minimum distance for guarenteed collision-free sampling is the sum of each cube's effective radius
        # We scale the min dist by 2 so that the cubes aren't too close together
        min_dists = (self.box_size + self.receptacle_size) / np.sqrt(2)
        min_dists = min_dists * 2.0

        # Sampling is "centered" around middle of table
        centered_cube_xy_state = torch.tensor(
            self.object_pos_center[:2], device=self.device, dtype=torch.float32
        )

        # Set z value, which is fixed height
        sampled_cube_pose[:, 2] = self.cfg_base.env.table_height + 0.5 * cube_heights

        # Initialize rotation
        theta_half = (2 * torch.rand(num_resets, device=self.device) - 1) * math.pi / 8
        sampled_cube_pose[:, 5] = torch.sin(theta_half)
        sampled_cube_pose[:, 6] = torch.cos(theta_half)

        # If we're verifying valid sampling, we need to check and re-sample if any are not collision-free
        # We use a simple heuristic of checking based on cubes' radius to determine if a collision would occur
        if check_valid:
            success = False
            # Indexes corresponding to envs we're still actively sampling for
            active_idx = torch.arange(num_resets, device=self.device)
            num_active_idx = len(active_idx)
            for i in range(100):
                # Sample x y values
                sampled_cube_pose[active_idx, :2] = (
                    centered_cube_xy_state
                    + 2.0
                    * self.object_pos_noise
                    * (torch.rand_like(sampled_cube_pose[active_idx, :2]) - 0.5)
                )
                # Check if sampled values are valid
                cube_dist = torch.linalg.norm(
                    sampled_cube_pose[:, :2] - other_cube_pose[:, :2], dim=-1
                )
                active_idx = torch.nonzero(cube_dist < min_dists, as_tuple=True)[0]
                num_active_idx = len(active_idx)
                # If active idx is empty, then all sampling is valid :D
                if num_active_idx == 0:
                    success = True
                    break
            # Make sure we succeeded at sampling
            assert success, "Sampling cube locations was unsuccessful! ):"
        else:
            # We just directly sample
            sampled_cube_pose[:, :2] = centered_cube_xy_state.unsqueeze(
                0
            ) + 2.0 * self.object_pos_noise * (
                torch.rand(num_resets, 2, device=self.device) - 0.5
            )

        return sampled_cube_pose

    def _reset_object_and_receptacle(self, env_ids, object_init_pose, receptacle_init_pose):
        """Reset object and receptacle poses."""
        if object_init_pose is None:
            object_init_pose = self._reset_init_cube_state(
                name="object", env_ids=env_ids, other_cube_pose=None, check_valid=False,
            )

        if receptacle_init_pose is None:
            receptacle_init_pose = self._reset_init_cube_state(
                name="receptacle", env_ids=env_ids, other_cube_pose=object_init_pose, check_valid=True,
            )

        self.object_pos[env_ids] = object_init_pose[:, 0:3]
        self.object_quat[env_ids] = object_init_pose[:, 3:7]
        self.object_linvel[env_ids, :] = 0.0
        self.object_angvel[env_ids, :] = 0.0
        self.receptacle_pos[env_ids] = receptacle_init_pose[:, 0:3]
        self.receptacle_quat[env_ids] = receptacle_init_pose[:, 3:7]
        self.receptacle_linvel[env_ids, :] = 0.0
        self.receptacle_angvel[env_ids, :] = 0.0

        object_actor_ids_sim = self.object_actor_ids_sim[env_ids].to(torch.int32)
        receptacle_actor_ids_sim = self.receptacle_actor_ids_sim[env_ids].to(torch.int32)
        combined_actor_ids_sim = torch.cat([object_actor_ids_sim, receptacle_actor_ids_sim], dim=0)

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state),
            gymtorch.unwrap_tensor(combined_actor_ids_sim),
            len(combined_actor_ids_sim),
        )

        self.init_receptacle_xy[env_ids] = receptacle_init_pose[:, :2]

    def reset_idx(self, env_ids: torch.Tensor, validation_set: bool = False):
        """Reset environments having the provided indices.
        Args:
            env_ids: environments to reset
        """
        # update franka states
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
            receptacle_init_pose = self.init_states["receptacle_pose"][start:end]
        else:
            franka_init_dof_pos = None
            object_init_pose = None
            receptacle_init_pose = None

        self._reset_franka(env_ids, franka_init_dof_pos)
        self._reset_object_and_receptacle(env_ids, object_init_pose, receptacle_init_pose)
        
        self.disable_gravity()
        self.simulate_and_refresh()
        self.enable_gravity()

        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        self.prev_visual_obs = None
        self.prev_seg_obs = None
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def pre_steps(self, obs, keep_runs, num_pre_steps=200):
        """Teleport the gripper to pick up the cube and move to somewhere above the receptacle. 
        The function generates easier initial states to test the policy.
        Args:
            obs: observations before the pre-step
            keep_runs: whether a run is valid or not (a run is invalid if the cube drops off the gripper)
            num_pre_steps: number of steps to run in the pre-step
        Returns:
            obs: observations after the pre-step
            keep_runs: whether a run is valid or not after the pre-step
        """
        capture_video = self.capture_video
        self.capture_video = False

        for step in range(num_pre_steps):
            if step <= num_pre_steps // 2:
                target_pos = self.object_pos
                down_q = to_torch(
                    self.num_envs * [1.0, 0.0, 0.0, 0.0], device=self.device
                ).reshape(-1, 4)
                target_rot = self.object_quat
                target_rot = quat_mul(down_q, quat_conjugate(target_rot))
            else:
                z_offset = 0.15 if step < num_pre_steps - num_pre_steps // 4 else 0.10
                target_pos = self.receptacle_pos + torch.tensor([0.0, 0.0, z_offset], device=self.device)
                target_rot = torch.tensor(
                    [1.0, 0.0, 0.0, 0.0], device=self.device
                ).repeat(self.num_envs, 1)

            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_centered_pos,
                fingertip_midpoint_quat=self.fingertip_centered_quat,
                ctrl_target_fingertip_midpoint_pos=target_pos,
                ctrl_target_fingertip_midpoint_quat=target_rot,
                jacobian_type=self.cfg_ctrl['jacobian_type'],
                rot_error_type='axis_angle')

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)
            actions = delta_hand_pose * 5.0

            if step >= num_pre_steps // 2 - 30:
                # close gripper
                self.cfg_task.randomize.franka_gripper_initial_state = 0.0
            else:
                # open gripper
                self.cfg_task.randomize.franka_gripper_initial_state = 1.0

            obs, reward, done, info = self.step(actions)
            self.progress_buf[:] = 0

        dist = torch.norm(self.object_pos - self.fingertip_centered_pos, dim=-1)
        keep_runs &= dist < self.box_size * np.sqrt(3)

        self.capture_video = capture_video
        return obs, keep_runs
