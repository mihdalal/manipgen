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

import numpy as np
import torch
import trimesh as tm

import os
import yaml
import random
import math

from manipgen.envs.franka_env import FrankaEnv
from manipgen.utils.geometry_utils import (
    sample_grasp_points, 
    sample_mesh_keypoints,
    sample_mesh_points,
)
import isaacgymenvs.tasks.factory.factory_control as fc

class FrankaArticulatedEnv(FrankaEnv):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
    
    #=============================================================
    # Camera and Visualization
    #=============================================================
    def create_viewer_and_camera(self):
        """Change location of viewer camera in articulated tasks"""
        super().create_viewer_and_camera()
        if self.capture_video:
            for id in range(self.num_envs):
                if id < self.capture_envs:
                    camera_handle = self.camera_handles[id][0]
                    camera_position = gymapi.Vec3(-0.9, 1.4, 2.0)
                    camera_target = gymapi.Vec3(
                        self.table_pose.p.x, self.table_pose.p.y, self.cfg_base.env.table_height
                    )
                    self.gym.set_camera_location(
                        camera_handle, self.env_ptrs[id], camera_position, camera_target
                    )

    def set_segmentation_id(self):
        for id in range(self.num_envs):
            env_ptr = self.env_ptrs[id]
            for object_id in range(self.num_objects):
                handle_idx_actor_domain = self.gym.find_actor_rigid_body_index(
                    env_ptr, self.object_handles_multitask[object_id][id], "handle_link", gymapi.DOMAIN_ACTOR
                )
                self.gym.set_rigid_body_segmentation_id(
                    env_ptr, self.object_handles_multitask[object_id][id], handle_idx_actor_domain, 3
                )
    
    def prepare_init_states(self, init_states):
        """ Prepare initial states for the environment """
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

        # set number of initial states
        self.num_init_states = len(self.init_states[first_key])
        self.episode_cur = 0
        self.val_num = int(self.num_init_states * self.cfg["env"].get("val_ratio", 0.1))
        self.episode_cur_val = self.num_init_states - self.val_num
        print("Number of initial states:", self.num_init_states)
        print("Validation set size:", self.val_num)
    
    def update_extras(self, camera_images):
        super().update_extras(camera_images)
        self.extras[f'{self.object_code}_success'] = self.success_buf

    #=============================================================
    # Partnet Assets
    #=============================================================
    def import_partnet_assets(self):
        """Load partnet object and set properties"""

        self.object_codes = []
        object_list = self.cfg["env"]["object_list"]
        if len(object_list) > 0:
            # multi-task mode
            for code in object_list:
                self.object_codes.append(code)
        else:
            # single-task mode
            self.object_codes.append(self.cfg["env"]["object_code"])

        # load handle mesh
        mesh_path = os.path.join(self.asset_root, "partnet/meshdata")

        # create object asset
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.fix_base_link = True
        object_asset_options.thickness = 0.0    # default = 0.02
        object_asset_options.armature = 0.0    # default = 0.0
        object_asset_options.density = 30      # use small density here - mainly consider damping
        object_asset_options.disable_gravity = False
        object_asset_options.collapse_fixed_joints = False
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

        object_assets = []
        object_dof_props_list = []
        self.object_dof_lower_limits_multitask = []
        self.object_dof_upper_limits_multitask = []
        for code in self.object_codes:
            object_asset_file = f"{code}/coacd.urdf"
            object_asset = self.gym.load_asset(
                self.sim, mesh_path, object_asset_file, object_asset_options
            )
            object_assets.append(object_asset)

            object_dof_props = self.gym.get_asset_dof_properties(object_asset)
            object_dof_props_list.append(object_dof_props)
            self.object_dof_lower_limits_multitask.append(
                to_torch(object_dof_props["lower"], device=self.device)
            )
            self.object_dof_upper_limits_multitask.append(
                to_torch(object_dof_props["upper"], device=self.device)
            )

        self.num_objects = len(object_assets)

        return object_assets
        
    def load_partnet_data(
        self,
        object_code,
        load_grasp_poses=False,
        load_finger_pos=False,
        load_mesh_points=False,
        load_object_keypoints=False,
        num_grasp_poses=2048,
        num_mesh_points=256,
    ):
        # load handle mesh: different from unidexgrasp, we only load handle mesh here
        mesh_path = os.path.join(self.asset_root, "partnet/meshdata")
        self.object_mesh = tm.load(
            os.path.join(
                mesh_path,
                f"{object_code}/decomposed.obj",
            )
        )

        # load meta data
        with open(f"{mesh_path}/{object_code}/meta.yaml", "r") as f:
            self.object_meta = yaml.safe_load(f)

        # load grasp data
        franka_grasp_data_path = os.path.join(
            self.asset_root, f"partnet/graspdata/{object_code}.npy",
        )
        if os.path.exists(franka_grasp_data_path):
            franka_grasp_data = np.load(
                franka_grasp_data_path, allow_pickle=True
            ).item()
        else:
            franka_grasp_data = None

        if load_grasp_poses:
            # use grasp points to compute distance
            if franka_grasp_data is None:
                assert self.sample_grasp_pose
                self.grasp_pose = torch.zeros((1, 7), device=self.device)
                self.grasp_pose[0, 3] = 1.0
            else:
                grasp_pose_np = sample_grasp_points(franka_grasp_data, self.object_mesh, num_grasp_poses, uniform=True, rotation=True, threshold=0.0)
                grasp_pose = torch.from_numpy(grasp_pose_np).float().to(self.device)                       # (num_grasp_poses, 7)
                # accept symmetric grasp poses: rotate along (0, 0, 1) by pi in hand frame
                symmetric_transform = quat_from_angle_axis(
                    torch.tensor([math.pi], device=self.device), torch.tensor([0.0, 0.0, 1.0], device=self.device)
                ).expand(grasp_pose.shape[0], -1)
                grasp_pose_symmetric = grasp_pose.clone()
                grasp_pose_symmetric[:, 3:7] = quat_mul(grasp_pose_symmetric[:, 3:7], symmetric_transform)
                self.grasp_pose = torch.cat((grasp_pose, grasp_pose_symmetric), dim=0)

        if load_finger_pos:
            # use finger positions to compute distance
            if franka_grasp_data is None:
                assert self.sample_grasp_pose
                self.finger_pos = torch.zeros((1, 6), device=self.device)
            else:
                finger_pos_np = sample_grasp_points(franka_grasp_data, self.object_mesh, num_grasp_poses, uniform=False, finger_pos=True, threshold=0.00)
                self.finger_pos = torch.from_numpy(finger_pos_np).float().to(self.device)                   # (num_finger_points, 6)

        if load_object_keypoints:
            # sample keypoints on the object
            key_points_np = sample_mesh_keypoints(self.object_mesh, self.num_object_keypoints)
            self.object_keypoints_local = torch.from_numpy(key_points_np).float().to(self.device)           # (num_object_keypoints, 3)

        if load_mesh_points:
            # randomly sample points on the handle mesh
            object_mesh_points_np = sample_mesh_points(self.object_mesh, num_mesh_points)
            self.object_mesh_points = torch.from_numpy(object_mesh_points_np).float().to(self.device)                       # (num_mesh_points, 3)

    #=============================================================
    # Control
    #=============================================================
    def _set_dof_torque(self):
        """Set Franka DOF torque to move fingertips towards target pose."""
        
        # override the implementation in IndustrealBase as we only want to set the first 9 dofs of dof_torque
        self.dof_torque[:, :self.franka_num_dofs] = fc.compute_dof_torque(
            cfg_ctrl=self.cfg_ctrl,
            dof_pos=self.dof_pos,
            dof_vel=self.dof_vel,
            fingertip_midpoint_pos=self.fingertip_centered_pos,
            fingertip_midpoint_quat=self.fingertip_centered_quat,
            fingertip_midpoint_linvel=self.fingertip_centered_linvel,
            fingertip_midpoint_angvel=self.fingertip_centered_angvel,
            left_finger_force=self.left_finger_force,
            right_finger_force=self.right_finger_force,
            jacobian=self.fingertip_midpoint_jacobian_tf,
            arm_mass_matrix=self.arm_mass_matrix,
            ctrl_target_gripper_dof_pos=self.ctrl_target_gripper_dof_pos,
            ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_centered_pos,
            ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_centered_quat,
            ctrl_target_fingertip_contact_wrench=self.ctrl_target_fingertip_contact_wrench,
            device=self.device)
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.dof_torque),
                                                        gymtorch.unwrap_tensor(self.franka_actor_ids_sim),
                                                        len(self.franka_actor_ids_sim))
    
    #=============================================================
    # Object Properties
    #=============================================================
    def _set_object_init_and_target_dof_pos(self):
        """ Set initial and target object dof positions """
        raise NotImplementedError
    
    def _randomize_object_dof_properties(self):
        for id in range(self.num_envs):
            env_ptr = self.env_ptrs[id]
            object_dof_props = self.gym.get_actor_dof_properties(env_ptr, self.object_handles[id])
            object_dof_props["damping"][0] = random.uniform(    # default: 0.05
                self.cfg_task.randomize.object_dof_damping_lower, 
                self.cfg_task.randomize.object_dof_damping_upper,
            )
            object_dof_props["friction"][0] = random.uniform(   # default: 0.025
                self.cfg_task.randomize.object_dof_friction_lower,
                self.cfg_task.randomize.object_dof_friction_upper,
            )
            if self.object_meta["type"] == "door":
                object_dof_props["damping"][0] *= self.cfg_task.randomize.door_damping_scale
                object_dof_props["friction"][0] *= self.cfg_task.randomize.door_friction_scale
            object_dof_props["velocity"][0] = 2.0               # default: 0.1
            self.gym.set_actor_dof_properties(env_ptr, self.object_handles[id], object_dof_props)
