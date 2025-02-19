from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgymenvs.utils.torch_jit_utils import (
    quat_from_euler_xyz,
    quat_mul,
    quat_rotate,
    quat_conjugate,
    quat_from_angle_axis,
)

import numpy as np
import torch

import math
import os

from manipgen.envs.franka_env import FrankaEnv
from manipgen.utils.geometry_utils import get_keypoint_offsets_6d
import isaacgymenvs.tasks.factory.factory_control as fc

class FrankaPick(FrankaEnv):
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

        # disable clutter and obstacles in sample mode
        if cfg.sample_mode:
            self.cfg_env.env.enable_clutter_and_obstacle = False

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
            "dist_gripper_reward_temp": self.cfg["rl"]["dist_gripper_reward_temp"],
            "dist_xy_reward_temp": self.cfg["rl"]["dist_xy_reward_temp"],
            "gripper_offset_reward_temp": self.cfg["rl"]["gripper_offset_reward_temp"],
            "object_in_view_reward_temp": self.cfg["rl"]["object_in_view_reward_temp"],
            "object_in_view_reward_threshold": self.cfg["rl"]["object_in_view_reward_threshold"],
            "eef_pose_consistency_reward_temp": self.cfg["rl"]["eef_pose_consistency_reward_temp"],
            "finger_contact_force_threshold": self.cfg["rl"]["finger_contact_force_threshold"],
            "finger_contact_reward_temp": self.cfg["rl"]["finger_contact_reward_temp"],
            "success_bonus": self.cfg["rl"]["success_bonus"],
        }

        # reset w/o pre-sampled initial states
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

    def sample_unidexgrasp_clutter_asset(self, scale_data):
        """Sample an object from the unidexgrasp dataset as clutter."""
        # sample object code and scale
        with open(self.cfg_env.env.unidexgrasp_file_list, "r") as f:
            lines = f.readlines()
            code, scale = lines[np.random.randint(len(lines))].strip().split()
            scale = float(scale)
            scale_str = "{:03d}".format(int(100 * scale))

        self.clutter_object_codes_and_scales.append((code, scale))

        if (code, scale) in self.unidexgrasp_clutter_asset_cache:
            clutter_asset = self.unidexgrasp_clutter_asset_cache[(code, scale)]
        else:
            # create clutter asset
            object_asset_options = gymapi.AssetOptions()
            object_asset_options.fix_base_link = True
            object_asset_options.thickness = 0.0    # default = 0.02
            object_asset_options.armature = 0.0    # default = 0.0
            object_asset_options.density = 500
            object_asset_options.use_mesh_materials = True
            object_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
            object_asset_options.override_com = True
            object_asset_options.override_inertia = True
            object_asset_options.vhacd_enabled = True
            object_asset_options.vhacd_params = gymapi.VhacdParams()
            object_asset_options.vhacd_params.resolution = 300000
            object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            if self.cfg_base.mode.export_scene:
                object_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_FACE

            mesh_path = os.path.join(self.asset_root, "unidexgrasp/meshdatav3_scaled")
            scaled_object_asset_file = f"{code}/coacd/coacd_{scale_str}.urdf"
            clutter_asset = self.gym.load_asset(
                self.sim, mesh_path, scaled_object_asset_file, object_asset_options
            )
            self.unidexgrasp_clutter_asset_cache[(code, scale)] = clutter_asset

        # sample rest poses for each object
        franka_grasp_data_path = os.path.join(
            self.asset_root,
            "unidexgrasp/graspdata",
            code.replace("/", "-") + f"-{scale_str}.npy",
        )
        franka_grasp_data = np.load(
            franka_grasp_data_path, allow_pickle=True
        ).item()

        num_rest_poses = franka_grasp_data["object_euler_xy"].shape[0]
        idx = torch.randint(num_rest_poses, (self.cfg_env.env.num_unidexgrasp_clutter_rest_poses,))      
        object_euler_xy = torch.from_numpy(franka_grasp_data["object_euler_xy"])[idx].float().to(self.device)
        object_init_z = torch.from_numpy(franka_grasp_data["object_z"])[idx].float().to(self.device)
        scale = torch.from_numpy(scale_data[code][scale]["radius_xy"])[idx].float().to(self.device)

        return clutter_asset, scale, object_euler_xy, object_init_z
        
    def import_clutter_and_obstacle_assets(self):
        """Import clutter and obstacle assets."""
        
        if self.cfg_env.env.enable_clutter_and_obstacle:
            if self.cfg_env.env.use_unidexgrasp_clutter or len(self.object_codes) > 1:
                # use unidexgrasp objects as clutter if specified or if multitask
                self.cfg_env.env.use_unidexgrasp_clutter = True
                self.clutter_object_codes_and_scales = []
                clutter_assets = [[] for _ in range(self.num_envs)]
                self.clutter_scales = [[] for _ in range(self.num_envs)]
                self.clutter_euler_xy = [[] for _ in range(self.num_envs)]
                self.clutter_init_z = [[] for _ in range(self.num_envs)]
                scale_data = np.load(os.path.join(self.asset_root, "unidexgrasp/datasetv4.1_posedata.npy"), allow_pickle=True).item()
                self.unidexgrasp_clutter_asset_cache = {}
                for env_id in range(self.num_envs):
                    for clutter_id in range(self.cfg_env.env.num_clutter_objects):
                        clutter_asset, scale, object_euler_xy, object_init_z = self.sample_unidexgrasp_clutter_asset(scale_data)
                        clutter_assets[env_id].append(clutter_asset)
                        self.clutter_scales[env_id].append(scale)
                        self.clutter_euler_xy[env_id].append(object_euler_xy)
                        self.clutter_init_z[env_id].append(object_init_z)
                self.clutter_scales = torch.stack([torch.stack(scales, dim=0) for scales in self.clutter_scales], dim=0)
                self.clutter_euler_xy = torch.stack([torch.stack(euler_xy, dim=0) for euler_xy in self.clutter_euler_xy], dim=0)
                self.clutter_init_z = torch.stack([torch.stack(init_z, dim=0) for init_z in self.clutter_init_z], dim=0)
            else:
                # use capsule as clutter
                asset_options = gymapi.AssetOptions()
                asset_options.fix_base_link = True
                radius = self.cfg_env.env.clutter_object_radius
                clutter_asset = self.gym.create_capsule(
                    self.sim, radius, radius, asset_options
                )
                clutter_assets = [[clutter_asset for _ in range(self.cfg_env.env.num_clutter_objects)] for env_id in range(self.num_envs)]

            asset_options = gymapi.AssetOptions()
            asset_options.fix_base_link = True
            width = self.cfg_env.env.obstacle_width
            height = self.cfg_env.env.obstacle_height
            obstacle_asset = self.gym.create_box(
                self.sim, 0.01, width, height, asset_options
            )
        else:
            clutter_assets, obstacle_asset = None, None

        return clutter_assets, obstacle_asset
    
    def create_envs(self):
        """Set env options. Import assets. Create actors."""

        lower = gymapi.Vec3(-self.cfg_base.env.env_spacing, -self.cfg_base.env.env_spacing, 0.0)
        upper = gymapi.Vec3(self.cfg_base.env.env_spacing, self.cfg_base.env.env_spacing, self.cfg_base.env.env_spacing)
        num_per_row = int(np.sqrt(self.num_envs,))

        # import assets
        self.print_sdf_warning()
        franka_asset, table_asset = self.import_franka_assets()
        object_assets = self.import_unidexgrasp_assets()
        clutter_assets, obstacle_asset = self.import_clutter_and_obstacle_assets()
        
        # create actors
        self._create_actors(lower, upper, num_per_row, franka_asset, table_asset, object_assets, clutter_assets, obstacle_asset)

    def _create_actors(self, lower, upper, num_per_row, franka_asset, table_asset, object_assets, clutter_assets, obstacle_asset):
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
        object_pose.p.x = 1.0
        object_pose.p.y = 0.0
        object_pose.p.z = 0.15
        object_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        if self.cfg_env.env.enable_clutter_and_obstacle:
            clutter_pose = gymapi.Transform()
            clutter_pose.p.x = 1.0
            clutter_pose.p.y = 0.0
            clutter_pose.p.z = 0.3
            clutter_pose.r = gymapi.Quat(0.0, np.pi / 4, 0.0, np.pi / 4)

            obstacle_pose = gymapi.Transform()
            obstacle_pose.p.x = -1.0
            obstacle_pose.p.y = 0.0
            obstacle_pose.p.z = 0.3
            obstacle_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

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
        self.object_rigid_body_ids_sim_multitask = [[] for _ in range(self.num_objects)] # within-sim indices
        actor_count = 0

        self.clutter_handles = [[] for _ in range(self.cfg_env.env.num_clutter_objects)]
        self.clutter_actor_ids_sim = [[] for _ in range(self.cfg_env.env.num_clutter_objects)]
        self.clutter_rigid_body_ids_sim = [[] for _ in range(self.cfg_env.env.num_clutter_objects)]

        self.obstacle_handles = [[] for _ in range(self.cfg_env.env.num_obstacles)]
        self.obstacle_actor_ids_sim = [[] for _ in range(self.cfg_env.env.num_obstacles)]
        self.obstacle_rigid_body_ids_sim = [[] for _ in range(self.cfg_env.env.num_obstacles)]

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
                object_handle = self.gym.create_actor(
                    env_ptr, object_asset, object_pose, f"object_{object_id}", i, 4, 3       # disable collision between objects
                )
                self.object_actor_ids_sim_multitask[object_id].append(actor_count)
                self.object_handles_multitask[object_id].append(object_handle)
                actor_count += 1

            if self.cfg_env.env.enable_clutter_and_obstacle:
                for clutter_object_id in range(self.cfg_env.env.num_clutter_objects):
                    clutter_handle = self.gym.create_actor(
                        env_ptr, clutter_assets[i][clutter_object_id], clutter_pose, f"clutter_{clutter_object_id}", i, 2, 4
                    )
                    self.clutter_actor_ids_sim[clutter_object_id].append(actor_count)
                    self.clutter_handles[clutter_object_id].append(clutter_handle)
                    actor_count += 1

                for obstacle_id in range(self.cfg_env.env.num_obstacles):
                    obstacle_handle = self.gym.create_actor(
                        env_ptr, obstacle_asset, obstacle_pose, f"obstacle_{obstacle_id}", i, 2, 4
                    )
                    self.obstacle_actor_ids_sim[obstacle_id].append(actor_count)
                    self.obstacle_handles[obstacle_id].append(obstacle_handle)
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

                object_color = gymapi.Vec3(0.6, 0.1, 0.0)
                self.gym.set_rigid_body_color(
                    env_ptr, object_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, object_color
                )

            if self.cfg_env.env.enable_clutter_and_obstacle:
                color = gymapi.Vec3(0.8, 0.9, 1.0)
                for clutter_object_id in range(self.cfg_env.env.num_clutter_objects):
                    clutter_handle = self.clutter_handles[clutter_object_id][-1]
                    self.gym.set_rigid_body_color(
                        env_ptr, clutter_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color
                    )
                for obstacle_id in range(self.cfg_env.env.num_obstacles):
                    obstacle_handle = self.obstacle_handles[obstacle_id][-1]
                    self.gym.set_rigid_body_color(
                        env_ptr, obstacle_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color
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
                object_body_id = self.gym.get_actor_rigid_body_index(
                    env_ptr, self.object_handles_multitask[object_id][-1], 0, gymapi.DOMAIN_SIM
                )
                self.object_rigid_body_ids_sim_multitask[object_id].append(object_body_id)

            if self.cfg_env.env.enable_clutter_and_obstacle:
                for clutter_object_id in range(self.cfg_env.env.num_clutter_objects):
                    clutter_body_id = self.gym.get_actor_rigid_body_index(
                        env_ptr, self.clutter_handles[clutter_object_id][-1], 0, gymapi.DOMAIN_SIM
                    )
                    self.clutter_rigid_body_ids_sim[clutter_object_id].append(clutter_body_id)

                for obstacle_id in range(self.cfg_env.env.num_obstacles):
                    obstacle_body_id = self.gym.get_actor_rigid_body_index(
                        env_ptr, self.obstacle_handles[obstacle_id][-1], 0, gymapi.DOMAIN_SIM
                    )
                    self.obstacle_rigid_body_ids_sim[obstacle_id].append(obstacle_body_id)

            self.franka_num_dofs = self.gym.get_actor_dof_count(env_ptr, franka_handle)

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
        self.clutter_actor_ids_sim = [
            torch.tensor(clutter_actor_ids_sim, dtype=torch.int32, device=self.device)
            for clutter_actor_ids_sim in self.clutter_actor_ids_sim
        ]
        self.obstacle_actor_ids_sim = [
            torch.tensor(obstacle_actor_ids_sim, dtype=torch.int32, device=self.device)
            for obstacle_actor_ids_sim in self.obstacle_actor_ids_sim
        ]

        # For extracting root pos/quat
        self.franka_actor_id_env = self.gym.find_actor_index(env_ptr, 'franka', gymapi.DOMAIN_ENV)
        self.object_actor_id_env_multitask = [
            self.gym.find_actor_index(env_ptr, f"object_{object_id}", gymapi.DOMAIN_ENV)
            for object_id in range(self.num_objects)
        ]
        if self.cfg_env.env.enable_clutter_and_obstacle:
            self.clutter_actor_id_env = [
                self.gym.find_actor_index(env_ptr, f"clutter_{clutter_object_id}", gymapi.DOMAIN_ENV)
                for clutter_object_id in range(self.cfg_env.env.num_clutter_objects)
            ]
            self.obstacle_actor_id_env = [
                self.gym.find_actor_index(env_ptr, f"obstacle_{obstacle_id}", gymapi.DOMAIN_ENV)
                for obstacle_id in range(self.cfg_env.env.num_obstacles)
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

        # contact force on fingers
        self.lf_contact_force = self.contact_force.view(-1, 3)[self.franka_left_finger_ids_sim, :]
        self.rf_contact_force = self.contact_force.view(-1, 3)[self.franka_right_finger_ids_sim, :]

        # gripper direction
        self.up_v = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)
        self.gripper_direction = torch.zeros(self.num_envs, 3, device=self.device)
        self.gripper_direction_prev = torch.zeros(self.num_envs, 3, device=self.device)
        self.gripper_direction_init = torch.zeros(self.num_envs, 3, device=self.device)

        # reward
        self.num_object_keypoints = self.cfg_env.rl.num_object_keypoints
        self.object_keypoints = torch.zeros(self.num_envs, self.num_object_keypoints * 3, device=self.device)
        self.dist_xy = torch.zeros(self.num_envs, device=self.device)  # distance between current object xy and initial object xy
        self.dist_grasp_keypoints = torch.zeros(self.num_envs, device=self.device)
        self.dist_gripper_direction = torch.zeros(self.num_envs, device=self.device)  # distance between current gripper direction and previous gripper direction in radians
        self.object_height = torch.zeros(self.num_envs, device=self.device)  # object height based on the lowest point on the object
        self.gripper_dof_offset = torch.zeros(self.num_envs, 2, device=self.device)  # gripper dof offset

        # override in reset
        self.object_rest_height = torch.zeros(self.num_envs, device=self.device) + self.cfg_env.env.table_height
        self.init_object_xy = torch.zeros(self.num_envs, 2, device=self.device)
        
        # encourage including the object in wrist camera view
        if self.cfg_task.rl.enable_object_in_view_reward:
            # wrist camera position and direction in the frame of Franka hand
            self.wrist_camera_pos_hand_frame = torch.tensor(
                self.cfg_task.env.local_obs.camera_offset, device=self.device
            ).float().repeat(self.num_envs, 1)
            self.wrist_camera_direction_hand_frame = torch.tensor(
                [-np.sin(self.cfg_task.env.local_obs.camera_angle), 0.0, np.cos(self.cfg_task.env.local_obs.camera_angle)], device=self.device
            ).float().repeat(self.num_envs, 1)

            # evaluate how aligned the object is with the central axis of the wrist camera
            self.axis_object_alignment = torch.zeros((self.num_envs,), device=self.device)

        # clutter and obstacle
        if self.cfg_env.env.enable_clutter_and_obstacle:
            self.clutter_pos = self.root_pos[:, self.clutter_actor_id_env[0]:self.clutter_actor_id_env[-1]+1, 0:3]                              # (n_envs, n_clutter_objects, 3)
            self.clutter_quat = self.root_quat[:, self.clutter_actor_id_env[0]:self.clutter_actor_id_env[-1]+1, :]                              # (n_envs, n_clutter_objects, 4)
            self.clutter_contact_force = self.contact_force.view(-1, 3)[self.clutter_rigid_body_ids_sim, :].transpose(0, 1).norm(dim=-1)        # (n_envs, n_clutter_objects)
            self.obstacle_pos = self.root_pos[:, self.obstacle_actor_id_env[0]:self.obstacle_actor_id_env[-1]+1, 0:3]                           # (n_envs, n_obstacles, 3)
            self.obstacle_quat = self.root_quat[:, self.obstacle_actor_id_env[0]:self.obstacle_actor_id_env[-1]+1, :]
            self.obstacle_contact_force = self.contact_force.view(-1, 3)[self.obstacle_rigid_body_ids_sim, :].transpose(0, 1).norm(dim=-1)      # (n_envs, n_obstacles)

            self.clutter_pos_xy = self.clutter_pos[:, :, :2].reshape(self.num_envs, -1)                                              # (n_envs, n_clutter_objects * 2)
        else:
            self.clutter_pos_xy = torch.zeros((self.num_envs, 0), device=self.device)

    def _refresh_task_tensors(self):
        """Refresh tensors."""
        self.object_keypoints = self._compute_keypoints_pos()
        self.dist_xy = torch.norm(self.object_pos[:, :2] - self.init_object_xy, dim=-1)
        self.object_height = self._compute_lift_height()

        # fingertip pos
        self.left_fingertip_pos = self.body_pos[:, self.left_fingertip_body_id_env, 0:3]
        self.right_fingertip_pos = self.body_pos[:, self.right_fingertip_body_id_env, 0:3]

        # contact force on fingers
        self.lf_contact_force = self.contact_force.view(-1, 3)[self.franka_left_finger_ids_sim, :]
        self.rf_contact_force = self.contact_force.view(-1, 3)[self.franka_right_finger_ids_sim, :]

        self.dist_grasp_keypoints = self._compute_nearest_grasp_pose_keypoint_distance()
        self.gripper_direction = quat_rotate(self.fingertip_centered_quat, self.up_v)
        self.dist_gripper_direction = torch.arccos(
            torch.clamp((self.gripper_direction_prev * self.gripper_direction).sum(dim=-1), min=-1.0, max=1.0)
        )
        self.gripper_direction_prev = self.gripper_direction.clone()

        self.gripper_dof_offset = 0.04 - self.gripper_dof_pos

        if self.cfg_task.rl.enable_object_in_view_reward:
            wrist_camera_pos = self.hand_pos + quat_rotate(self.hand_quat, self.wrist_camera_pos_hand_frame)
            wrist_camera_dir = quat_rotate(self.hand_quat, self.wrist_camera_direction_hand_frame)

            wrist_camera_pos_object_frame = quat_rotate(quat_conjugate(self.object_quat), wrist_camera_pos - self.object_pos)   # (n_envs, 3)
            wrist_camera_dir_object_frame = quat_rotate(quat_conjugate(self.object_quat), wrist_camera_dir)                     # (n_envs, 3)

            self.axis_object_alignment[:] = compute_axis_object_alignment(
                wrist_camera_pos_object_frame, wrist_camera_dir_object_frame, self.object_mesh_points
            )

        # clutter and obstacle
        if self.cfg_env.env.enable_clutter_and_obstacle:
            self.clutter_contact_force = self.contact_force.view(-1, 3)[self.clutter_rigid_body_ids_sim, :].transpose(0, 1).norm(dim=-1)        # (n_envs, n_clutter_objects)
            self.obstacle_contact_force = self.contact_force.view(-1, 3)[self.obstacle_rigid_body_ids_sim, :].transpose(0, 1).norm(dim=-1)      # (n_envs, n_obstacles)
            self.clutter_pos_xy = self.clutter_pos[:, :, :2].reshape(self.num_envs, -1)                                                         # (n_envs, n_clutter_objects * 2)
        else:
            self.clutter_pos_xy = torch.zeros((self.num_envs, 0), device=self.device)
            
    def _compute_nearest_grasp_pose_keypoint_distance(self):
        """Based on current gripper keypoints, compute distance to nearest valid grasp pose keypoints."""
        # convert gripper pose to object local frame
        current_grasp_pos_local = quat_rotate(quat_conjugate(self.object_quat), self.fingertip_centered_pos - self.object_pos)
        current_grasp_rot_local = quat_mul(quat_conjugate(self.object_quat), self.fingertip_centered_quat)
        current_grasp_pose_local = torch.cat([current_grasp_pos_local, current_grasp_rot_local], dim=-1)
        current_grasp_keypoints_local = transform_keypoints_6d(current_grasp_pose_local, self.gripper_keypoint_offsets)
        # compute distance to nearest grasp pose based on keypoints
        dist_grasp_keypoints = compute_nearest_keypoint_6d_distance(current_grasp_keypoints_local, self.target_grasp_keypoints_local)

        return dist_grasp_keypoints

    def _compute_keypoints_pos(self, object_pose=None):
        if object_pose is None:
            object_pos, object_rot = self.object_pos, self.object_quat
        else:
            object_pos, object_rot = object_pose[:, :3], object_pose[:, 3:]
        num_object_poses = object_pos.shape[0]
        key_points_world = quat_rotate(
            object_rot.unsqueeze(1).expand(-1, self.num_object_keypoints, -1).reshape(-1, 4),
            self.object_keypoints_local.expand(num_object_poses, -1, -1).reshape(-1, 3),
        ).reshape(num_object_poses, self.num_object_keypoints, 3) + object_pos.unsqueeze(1)
        key_points_world = key_points_world.reshape(num_object_poses, -1)
        return key_points_world
    
    def _check_lift_success(self, lift_height=0.05):
        lift_success = torch.where(
            self.object_height > lift_height,                               # object is lifted up
            torch.ones((self.num_envs,), device=self.device),
            torch.zeros((self.num_envs,), device=self.device))
        lift_success = torch.where(
            self.gripper_dof_pos[:, -2:].sum(dim=-1) > 0.002,               # gripper is not fully closed
            lift_success,
            torch.zeros((self.num_envs,), device=self.device),
        )

        return lift_success

    def _compute_lift_height(self):
        """ Compute height as distance from the lowest point on the object to the table. """
        object_pos = self.object_pos
        object_rot = self.object_quat

        # vector from object pos to table surface
        v = torch.zeros(self.num_envs, 3, device=self.device)
        v[:, 2] = self.cfg_base.env.table_height - object_pos[:, 2]
        # rotate the vector to the object frame
        v_local = quat_rotate(quat_conjugate(object_rot), v)
        # find the lowest point
        object_height = distance_to_table(
            self.convex_hull_points,
            v_local,
            -v[:, 2]
        )
        object_height = torch.nan_to_num(object_height, nan=0.0, posinf=0.0, neginf=0.0)
        object_height = torch.clamp(object_height, 0.0)

        return object_height
    
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
        gripper_offset_reward_temp = self.reward_settings["gripper_offset_reward_temp"]
        object_in_view_reward_temp = self.reward_settings["object_in_view_reward_temp"]
        eef_pose_consistency_reward_temp = self.reward_settings["eef_pose_consistency_reward_temp"]
        finger_contact_reward_temp = self.reward_settings["finger_contact_reward_temp"]
        finger_contact_force_threshold = self.reward_settings["finger_contact_force_threshold"]
        object_in_view_reward_threshold = self.reward_settings["object_in_view_reward_threshold"]
        success_bonus = self.reward_settings["success_bonus"]

        # component 1: keypoint distance between current gripper pose and nearest pre-sampled grasp pose
        # encourage the gripper to move to the nearest grasp pose
        dist_gripper_reward = (
            torch.exp(dist_gripper_reward_temp * self.dist_grasp_keypoints)
            + torch.exp(dist_gripper_reward_temp * self.dist_grasp_keypoints * 10)
            + torch.exp(dist_gripper_reward_temp * self.dist_grasp_keypoints * 100)
            + torch.exp(dist_gripper_reward_temp * self.dist_grasp_keypoints * 1000)
        ) / 4
        
        # component 2: distance between current object xy and initial object xy
        # penalize the object moving away from the initial position
        dist_xy_reward = torch.exp(dist_xy_reward_temp * self.dist_xy)
        
        # component 3: distance between current gripper width and open gripper width
        # this is to penalize the gripper from pushing against the table
        gripper_offset_reward = torch.exp(gripper_offset_reward_temp * self.gripper_dof_offset.sum(dim=-1))

        # component 4: the angle between the current and previous gripper pose along the gripper's central axis
        # we want the gripper to minimize its change in orientation to avoid collision with invisible obstacles
        eef_pose_consistency_reward = eef_pose_consistency_reward_temp * self.dist_gripper_direction

        # component 5: contact force on the gripper fingers
        # penalize the fingers from contacting the obstacles
        finger_contact_reward = finger_contact_reward_temp * (
            torch.clamp(torch.max(self.lf_contact_force[:, 2], self.rf_contact_force[:, 2]), min=finger_contact_force_threshold) \
            - finger_contact_force_threshold
        )

        # (deprecated) encourage including the object in wrist camera view
        if self.cfg_task.rl.enable_object_in_view_reward:
            alignment = torch.clamp(self.axis_object_alignment - object_in_view_reward_threshold, min=0.0)
            object_in_view_reward = torch.exp(object_in_view_reward_temp * alignment)
        else:
            object_in_view_reward = torch.ones_like(dist_gripper_reward)

        total_reward = dist_gripper_reward * dist_xy_reward * gripper_offset_reward * object_in_view_reward + eef_pose_consistency_reward + finger_contact_reward
        
        self.rew_buf[:] = total_reward
        reset = (self.progress_buf == self.max_episode_length - 1)

        if self.render_hardcode_control:
            self.extras['hardcode_images'] = []

        is_last_step = (self.progress_buf[0] == self.max_episode_length - 1)
        if is_last_step and not self.disable_hardcode_control:
            # log keypoint distance
            keypoint_dist = self.dist_grasp_keypoints.mean()
            self.extras['keypoint_dist'] = keypoint_dist
            self.extras['gripper_direction_dist'] = torch.arccos(
                torch.clamp((self.gripper_direction_init * self.gripper_direction).sum(dim=-1), min=-1.0, max=1.0)
            ).mean()
            self.extras['contact_force_z'] = torch.max(self.lf_contact_force[:, 2], self.rf_contact_force[:, 2]).mean()
            # teleportation: close gripper and lift up
            hardcode_images = self.hardcode_control(get_camera_images=self.render_hardcode_control)
            if self.render_hardcode_control:
                self.extras['hardcode_images'] = hardcode_images
            # check lift success and add success bonus
            lift_success = self._check_lift_success()
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

        lift_up = (self.progress_buf == self.max_episode_length - 1).float()
        if self.disable_hardcode_control:
            lift_up[:] = 0.0
        obs_tensors = [
            local_eef_pos, 
            local_eef_quat,
            self.fingertip_centered_linvel,                 # linear velocity of the gripper
            self.fingertip_centered_angvel,                 # angular velocity of the gripper
            self.object_pos,                                # position of the object
            self.object_quat,                               # orientation of the object
            self.object_keypoints,                          # position of keypoints of the object
            self.init_object_xy,                            # initial xy position of the object (prevent the object from moving)
            self.dist_gripper_direction.unsqueeze(-1),      # distance between current and previous gripper direction
            self.dist_grasp_keypoints.unsqueeze(-1),        # distance between gripper
            self.gripper_dof_offset,                        # gripper dof offset
            self.lf_contact_force,                          # contact force on the left finger
            self.rf_contact_force,                          # contact force on the right finger
            self.clutter_pos_xy,                            # xy position of clutter objects
            self.obstacle_pos.reshape(self.num_envs, -1),   # position of obstacles
            self.obstacle_quat.reshape(self.num_envs, -1),  # orientation of obstacles
            lift_up.unsqueeze(-1),                          # whether to lift up the object with hard-code control
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
            load_grasp_poses=True,
            load_convex_hull_points=True,           # compute object height
            load_object_keypoints=True,
            load_mesh_points=self.cfg_task.rl.enable_object_in_view_reward,
            filter_threshold=self.cfg_task.rl.filter_pose_threshold,
        )

        # process grasp data: keypoints corresponding to pre-sampled grasp poses in the object frame
        gripper_keypoint_scale = self.cfg_task.rl.gripper_keypoint_scale
        self.gripper_keypoint_offsets = get_keypoint_offsets_6d(self.device) * gripper_keypoint_scale

        assert self.cfg_task.rl.gripper_keypoint_dof in (5, 6), "Invalid gripper keypoint DOF."
        if self.cfg_task.rl.gripper_keypoint_dof == 5:
            select = torch.tensor([True, False, True, False, False, True, False]).to(self.device)
            self.gripper_keypoint_offsets = self.gripper_keypoint_offsets[select]

        self.target_grasp_keypoints_local = transform_keypoints_6d(self.grasp_pose, self.gripper_keypoint_offsets)

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

        self.object_rest_height[env_ids] = init_pos[:, 2]
        self.init_object_xy[env_ids] = init_pos[:, :2]

    def _clear_clutter_and_obstacles(self, env_ids):
        """Clear clutter and obstacles from the table before resetting Franka and object."""
        
        if not self.cfg_env.env.enable_clutter_and_obstacle:
            return
        
        self.clutter_pos[:, :, :] = 0.0
        self.obstacle_pos[:, :, :] = 0.0

        clutter_actor_ids_sim = torch.cat(self.clutter_actor_ids_sim, dim=0).to(torch.int32).reshape(-1, self.num_envs)[:, env_ids].reshape(-1)
        obstacle_actor_ids_sim = torch.cat(self.obstacle_actor_ids_sim, dim=0).to(torch.int32).reshape(-1, self.num_envs)[:, env_ids].reshape(-1)
        multi_actor_ids_sim = torch.cat([clutter_actor_ids_sim, obstacle_actor_ids_sim], dim=0)

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state),
            gymtorch.unwrap_tensor(multi_actor_ids_sim),
            len(multi_actor_ids_sim),
        )

        self.simulate_and_refresh()

    def _check_obstacle_pose(self, pos_xy, rot_theta):
        def ccw(u, v, w):
            return (w[:, 1] - u[:, 1]) * (v[:, 0] - u[:, 0]) > (v[:, 1] - u[:, 1]) * (w[:, 0] - u[:, 0])
        
        A = self.fingertip_centered_pos[:, :2]
        B = (self.fingertip_centered_pos - 0.30 * self.gripper_direction)[:, :2]

        C = pos_xy.clone()
        C[:, 0] += self.cfg_env.env.obstacle_width / 2 * torch.sin(rot_theta)
        C[:, 1] -= self.cfg_env.env.obstacle_width / 2 * torch.cos(rot_theta)
        D = pos_xy.clone()
        D[:, 0] -= self.cfg_env.env.obstacle_width / 2 * torch.sin(rot_theta)
        D[:, 1] += self.cfg_env.env.obstacle_width / 2 * torch.cos(rot_theta)


        valid = ~((ccw(A, C, D) != ccw(B, C, D)) & (ccw(A, B, C) != ccw(A, B, D)))
        A = self.object_pos[:, :2]
        valid &= ~((ccw(A, C, D) != ccw(B, C, D)) & (ccw(A, B, C) != ccw(A, B, D)))
        return valid
    
    def _get_random_rotation(self, magnitude=np.pi):
        """ Get a quaternion representing a random rotation around a random axis. """
        
        rot = torch.zeros((self.num_envs, 4), device=self.device)
        p = torch.randn((self.num_envs, 3), device=self.device)
        p = p / p.norm(dim=-1, keepdim=True)
        theta_half = (2 * torch.rand(self.num_envs, device=self.device) - 1) * magnitude * 0.5  # 0.25: half rotation, 0.5: full rotation
        rot[:, :3] = p * torch.sin(theta_half).unsqueeze(-1)
        rot[:, 3] = torch.cos(theta_half)
        
        return rot

    def _reset_clutter_and_obstacles(self, env_ids):
        """Reset clutter and obstacles on the table after resetting Franka and object."""

        if not self.cfg_env.env.enable_clutter_and_obstacle:
            return

        # sample clutter objects position
        if self.cfg_env.env.use_unidexgrasp_clutter:
            # sample rest pose for unidexgrasp clutter objects
            idx = torch.randint(self.cfg_env.env.num_unidexgrasp_clutter_rest_poses, (self.num_envs, self.cfg_env.env.num_clutter_objects), device=self.device)
            clutter_scales = torch.gather(self.clutter_scales, 2, idx.unsqueeze(-1)).squeeze(-1)    # (n_envs, n_clutter_objects)
            clutter_scales = torch.clamp(clutter_scales, min=self.cfg_env.env.clutter_object_radius)
        else:
            clutter_scales = self.cfg_env.env.clutter_object_radius
        
        valid = torch.zeros((self.num_envs, self.cfg_env.env.num_clutter_objects), device=self.device).bool()
        gripper_direction_xy = quat_rotate(self.fingertip_centered_quat, self.up_v)[:, :2]
        gripper_direction_xy = torch.where(gripper_direction_xy.abs() < 1e-6, 1e-6 * torch.sign(gripper_direction_xy), gripper_direction_xy)
        gripper_direction_xy = gripper_direction_xy / torch.norm(gripper_direction_xy, dim=-1, keepdim=True)
        reference_point = self.fingertip_centered_pos[:, :2] + gripper_direction_xy * self.cfg_env.env.clutter_object_radius * 1.5
        num_sample_iters = 50
        for _ in range(num_sample_iters):
            # sample around the object: radius in [object_scale + clutter_object_radius + safety_margin, clutter_dist_max]
            safety_margin = 2e-2
            radius = torch.rand((self.num_envs, self.cfg_env.env.num_clutter_objects), device=self.device) * (
                self.cfg_env.env.clutter_dist_max - self.object_scale - clutter_scales - safety_margin
            ) + self.object_scale + clutter_scales + safety_margin
            theta = torch.rand((self.num_envs, self.cfg_env.env.num_clutter_objects), device=self.device) * 2 * math.pi

            clutter_pos = torch.zeros((self.num_envs, self.cfg_env.env.num_clutter_objects, 3), device=self.device)
            clutter_pos[:, :, 0] = self.object_pos[:, 0].unsqueeze(-1) + radius * torch.cos(theta)
            clutter_pos[:, :, 1] = self.object_pos[:, 1].unsqueeze(-1) + radius * torch.sin(theta)
            clutter_pos[:, :, 2] = self.cfg_env.env.table_height

            # check if the clutter objects are within the range of table
            on_table = ((clutter_pos[:, :, 0] - self.table_pose.p.x).abs() < self.asset_info_franka_table.table_depth / 2 - 0.1) \
                & ((clutter_pos[:, :, 1] - self.table_pose.p.y).abs() < self.asset_info_franka_table.table_width / 2 - 0.1)
            
            # check if the clutter objects are too close to the eef
            v_from_reference_point = clutter_pos[:, :, :2] - reference_point.unsqueeze(-2)
            away = (v_from_reference_point * gripper_direction_xy.unsqueeze(-2)).sum(dim=-1) > 0

            valid_iter = on_table & away

            # (for place only) check if the clutter objects are too close to the target position
            if getattr(self, "target_object_xy", None) is not None:
                dist_to_target = torch.norm(clutter_pos[:, :, :2] - self.target_object_xy.unsqueeze(-2), dim=-1)
                valid_iter = valid_iter & (dist_to_target > (clutter_scales + self.object_scale))

            valid = valid | valid_iter

            self.clutter_pos[valid_iter, :] = clutter_pos[valid_iter, :]
            if valid.all():
                break

        if not valid.all():
            print(f"Warning: failed to sample valid clutter objects in {num_sample_iters} iterations.")
            self.clutter_pos[~valid, :] = clutter_pos[~valid, :]

        if self.cfg_env.env.use_unidexgrasp_clutter:
            clutter_init_z = torch.gather(self.clutter_init_z, 2, idx.unsqueeze(-1)).squeeze(-1)
            clutter_euler_xy = self.clutter_euler_xy[
                torch.arange(self.num_envs, device=self.device).unsqueeze(-1),
                torch.arange(self.cfg_env.env.num_clutter_objects, device=self.device),
                idx, :
            ]
            self.clutter_pos[:, :, 2] += clutter_init_z
            self.clutter_quat[:, :, :] = quat_from_euler_xyz(
                clutter_euler_xy.reshape(-1, 2)[:, 0],
                clutter_euler_xy.reshape(-1, 2)[:, 1],
                torch.rand(self.num_envs * self.cfg_env.env.num_clutter_objects, device=self.device) * 2 * math.pi,
            ).reshape(self.num_envs, self.cfg_env.env.num_clutter_objects, 4)

        # sample obstacle position on a square
        edge_length = torch.rand(self.num_envs, device=self.device) * 0.1 + (self.cfg_env.env.obstacle_width - 0.1)
        safety_margin = 2e-2
        center_noise_scale = torch.rand(self.num_envs, device=self.device) * (0.5 * edge_length - self.object_scale - safety_margin)
        center_noise_dir = torch.rand(self.num_envs, device=self.device) * 2 * math.pi
        center = self.object_pos[:, :2].clone()
        center[:, 0] += center_noise_scale * torch.cos(center_noise_dir)
        center[:, 1] += center_noise_scale * torch.sin(center_noise_dir)
        rotation = (torch.rand(self.num_envs, device=self.device) * 2 - 1) * 0.25 * math.pi

        is_x_upper_edge = torch.where(                                      # keep the edge more distant from the eef
            self.fingertip_centered_pos[:, 0] < center[:, 0], 1.0, -1.0
        )
        is_y_upper_edge = torch.where(
            self.fingertip_centered_pos[:, 1] < center[:, 1], 1.0, -1.0
        )

        obstacle_height = torch.rand(self.num_envs, device=self.device) * self.cfg_env.env.obstacle_height / 2 + self.cfg_env.env.table_height
        keep_obstacle1 = torch.rand(self.num_envs, device=self.device) < self.cfg_env.env.obstacle_show_up_prob
        keep_obstacle2 = torch.rand(self.num_envs, device=self.device) < self.cfg_env.env.obstacle_show_up_prob
        pos_noise1 = (torch.rand(self.num_envs, device=self.device) * 2 - 1) * self.cfg_env.env.obstacle_pos_noise
        pos_noise2 = (torch.rand(self.num_envs, device=self.device) * 2 - 1) * self.cfg_env.env.obstacle_pos_noise
        rot_noise1 = self._get_random_rotation(magnitude=self.cfg_env.env.obstacle_rot_noise)
        rot_noise2 = self._get_random_rotation(magnitude=self.cfg_env.env.obstacle_rot_noise)

        self.obstacle_pos[:, 0, 0] = center[:, 0] + edge_length / 2 * is_x_upper_edge * torch.cos(rotation) + pos_noise1 * torch.sin(rotation)
        self.obstacle_pos[:, 0, 1] = center[:, 1] + edge_length / 2 * is_x_upper_edge * torch.sin(rotation) - pos_noise1 * torch.cos(rotation)
        self.obstacle_pos[:, 0, 2] = obstacle_height
        self.obstacle_quat[:, 0] = quat_from_angle_axis(rotation, self.up_v)
        self.obstacle_quat[:, 0] = quat_mul(rot_noise1, self.obstacle_quat[:, 0])
        keep_obstacle1 &= self._check_obstacle_pose(self.obstacle_pos[:, 0, :2], rotation)
        self.obstacle_pos[~keep_obstacle1, 0, :] = torch.tensor([-1.0, 0.0, 0.0], device=self.device)
        self.obstacle_quat[~keep_obstacle1, 0, :] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device) 

        self.obstacle_pos[:, 1, 0] = center[:, 0] + edge_length / 2 * is_y_upper_edge * torch.cos(rotation + math.pi / 2) + pos_noise2 * torch.sin(rotation + math.pi / 2)
        self.obstacle_pos[:, 1, 1] = center[:, 1] + edge_length / 2 * is_y_upper_edge * torch.sin(rotation + math.pi / 2) - pos_noise2 * torch.cos(rotation + math.pi / 2)
        self.obstacle_pos[:, 1, 2] = obstacle_height * keep_obstacle2
        self.obstacle_quat[:, 1] = quat_from_angle_axis(rotation + math.pi / 2, self.up_v)
        self.obstacle_quat[:, 1] = quat_mul(rot_noise2, self.obstacle_quat[:, 1])
        keep_obstacle2 &= self._check_obstacle_pose(self.obstacle_pos[:, 1, :2], rotation + math.pi / 2)
        self.obstacle_pos[~keep_obstacle2, 1, :] = torch.tensor([-1.0, 0.0, 0.0], device=self.device)
        self.obstacle_quat[~keep_obstacle2, 1, :] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)

        # apply to the simulation
        clutter_actor_ids_sim = torch.cat(self.clutter_actor_ids_sim, dim=0).to(torch.int32).reshape(-1, self.num_envs)[:, env_ids].reshape(-1)
        obstacle_actor_ids_sim = torch.cat(self.obstacle_actor_ids_sim, dim=0).to(torch.int32).reshape(-1, self.num_envs)[:, env_ids].reshape(-1)
        multi_actor_ids_sim = torch.cat([clutter_actor_ids_sim, obstacle_actor_ids_sim], dim=0)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state),
            gymtorch.unwrap_tensor(multi_actor_ids_sim),
            len(multi_actor_ids_sim),
        )

        self.simulate_and_refresh()

        # check contact with the clutter and obstacles
        clutter_contact = self.clutter_contact_force != 0.0
        obstacle_contact = self.obstacle_contact_force != 0.0
        self.clutter_pos[clutter_contact, :] = torch.tensor([-1.0, 0.0, 0.0], device=self.device)
        self.clutter_quat[clutter_contact, :] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
        self.obstacle_pos[obstacle_contact, :] = torch.tensor([-1.0, 0.0, 0.0], device=self.device)
        self.obstacle_quat[obstacle_contact, :] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state),
            gymtorch.unwrap_tensor(multi_actor_ids_sim),
            len(multi_actor_ids_sim),
        )
        self.simulate_and_refresh()

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
        self.simulate_and_refresh()

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

    def pre_steps(self, obs, keep_runs, num_pre_steps=100):
        """Teleport the gripper to somewhere above the object. 
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

        xy_offset = (
            2 * torch.rand_like(self.fingertip_centered_pos[:, :2])
            - 1
        ) * 0.05
        for step in range(num_pre_steps):
            target_pos = self.object_pos + torch.tensor([0.0, 0.0, 0.10], device=self.device)
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

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def transform_keypoints_6d(pose, keypoint_offsets):
    # type: (Tensor, Tensor) -> Tensor

    num_poses = pose.shape[0]
    num_dofs = keypoint_offsets.shape[0]

    pos = pose[:, 0:3].unsqueeze(1).expand(-1, num_dofs, -1).reshape(-1, 3)                 # (num_poses * 7, 3)
    rot = pose[:, 3:7].unsqueeze(1).expand(-1, num_dofs, -1).reshape(-1, 4)
    offsets = keypoint_offsets.unsqueeze(0).expand(num_poses, -1, -1).reshape(-1, 3)        # (num_poses * 7, 3)

    keypoints = (quat_rotate(rot, offsets) + pos).reshape(num_poses, num_dofs, 3)

    return keypoints

@torch.jit.script
def compute_nearest_keypoint_6d_distance(current_keypoints, target_keypoints):
    # type: (Tensor, Tensor) -> Tensor
    num_cur = current_keypoints.shape[0]
    num_tar = target_keypoints.shape[0]

    target_keypoints = target_keypoints.unsqueeze(0).expand(num_cur, -1, -1, -1)            # (num_envs, num_cur, 7, 3)
    current_keypoints = current_keypoints.unsqueeze(1).expand(-1, num_tar, -1, -1)          # (num_envs, num_tar, 7, 3)

    dist = torch.norm(target_keypoints - current_keypoints, dim=-1).mean(dim=-1)            # (num_envs, num_tar)
    min_dist, min_idx = torch.min(dist, dim=-1)
    return min_dist

@torch.jit.script
def compute_axis_object_alignment(wrist_camera_pos_object_frame, wrist_camera_dir_object_frame, object_mesh_points):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    n_points = object_mesh_points.shape[0]

    v1 = object_mesh_points.reshape(1, -1, 3) - wrist_camera_pos_object_frame.reshape(-1, 1, 3)     # (n_envs, n_points, 3)
    v1 = v1 / torch.norm(v1, dim=-1, keepdim=True)
    v2 = wrist_camera_dir_object_frame.unsqueeze(1).expand(-1, n_points, -1)                        # (n_envs, n_points, 3)

    dot_prod = torch.sum(v1 * v2, dim=-1)
    max_dot_prod = torch.max(dot_prod, dim=-1)[0]
    max_dot_prod = torch.clamp(max_dot_prod, -1.0, 1.0)
    alignment = torch.acos(max_dot_prod)

    return alignment


@torch.jit.script
def distance_to_table(convex_hull_points, v_local, table_dist):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    """ Compute the distance from the table surface to the closest point on the convex hull
    Args:
        convex_hull_points: points on the convex hull in the object frame
        v_local: the vector from object center to the table surface, but converted to the object frame
        table_dist: distance from table to the object center
    Returns:
        dist: the minimum distance from the table surface to the convex hull
    """

    num_envs = v_local.shape[0]
    num_points = convex_hull_points.shape[0]
    direction = v_local / (torch.norm(v_local, dim=-1, keepdim=True) + 1e-8)     # num_envs x 3

    v1 = convex_hull_points.expand(num_envs, num_points, 3)
    v2 = direction.unsqueeze(1).expand(num_envs, num_points, 3)
    d = torch.sum(v1 * v2, dim=-1)
    convex_hull_dist = torch.max(d, dim=-1)[0]

    dist = table_dist - convex_hull_dist

    return dist
