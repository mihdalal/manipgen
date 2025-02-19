from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym import torch_utils
from isaacgymenvs.utils.torch_jit_utils import (
    to_torch, 
    quat_from_angle_axis, 
    quat_mul,
    quat_conjugate,
    get_euler_xyz,
)
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from isaacgymenvs.tasks.industreal.industreal_base import IndustRealBase
from isaacgymenvs.tasks.factory.factory_schema_config_base import FactorySchemaConfigBase
import isaacgymenvs.tasks.factory.factory_control as fc
from isaacgymenvs.tasks.factory.factory_control import axis_angle_from_quat
from manipgen.utils.media_utils import camera_shot
from manipgen.utils.geometry_utils import (
    sample_grasp_points, 
    sample_mesh_points_on_convex_hull, 
    sample_mesh_keypoints,
    sample_mesh_points,
)

import hydra
import trimesh as tm
import numpy as np
import torch
import math
import abc
import os
import copy

from typing import Dict, Any, Tuple, List, Set
import torchvision.transforms.functional as fn
from numpy import inf
import torchvision.transforms as T


class FrankaEnv(IndustRealBase):
    def __init__(
        self,
        cfg,
        headless=True,
        virtual_screen_capture=False,
        force_render=False,
    ):
        """
        Args:
            cfg: config dictionary for the environment.
            headless: Set to False to disable viewer rendering.
            virtual_screen_capture: Set to True to allow the users get captured screen in RGB array via `env.render(mode='rgb_array')`. 
            force_render: Set to True to always force rendering in the steps (if the `control_freq_inv` is greater than 1 we suggest stting this arg to True)
        """
        rl_device = sim_device = cfg.device
        graphics_device_id = (
            cfg.device
            if (cfg.render or cfg.capture_video or cfg.local_obs)
            else -1
        )
        cfg_task_dict = omegaconf_to_dict(cfg.task)
        self.render_viewer = cfg.render
        self.local_obs = cfg.local_obs
        self.global_obs = cfg.get("global_obs", False)
        self.capture_video = cfg.capture_video
        self.sample_mode = cfg.sample_mode
        self.asset_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../assets"
        )
        super().__init__(
            cfg_task_dict,
            rl_device,
            sim_device,
            graphics_device_id,
            headless,
            virtual_screen_capture,
            force_render,
        )
        self.capture_interval = cfg.capture_interval
        self.capture_length = cfg.capture_length
        self.sample_mode = cfg.sample_mode
        self.task_cfg = cfg.task
        self.capture_envs = min(cfg.capture_envs, self.num_envs)
        self.capture_obs_camera = cfg.get("capture_obs_camera", False)
        self.capture_depth = cfg.get("capture_depth", False)
        
        self.enable_camera_latency = cfg.get("enable_camera_latency", False)
        self.prev_visual_obs = None
        self.prev_seg_obs = None

        self.total_steps = 0

        # Total number of training frames since the beginning of the experiment.
        # We get this information from the learning algorithm rather than tracking ourselves.
        self.total_train_env_frames: int = 0

        # if not set, the environment will automatically reset after `self.episode_length` steps
        # in the test scripts, we want to manually reset the environment, so the automatic reset is disabled
        self.disable_automatic_reset = False

        # disable automatic hardcode control in the environment
        self.disable_hardcode_control = False

        # render hardcode control proccess - only effective when `self.disable_hardcode_control` is False        
        self.render_hardcode_control = False

        # export rigid body poses for rendering
        self.export_rigid_body_poses_dir = ""

        # this should only be True when sampling grasp poses
        self.sample_grasp_pose = cfg.get("sample_grasp_pose", False)

        self.acquire_base_tensors()  # defined in superclass
        self.refresh_base_tensors()  # defined in superclass
    
    def _get_base_yaml_params(self):
        """Initialize instance variables from YAML files."""

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name="factory_schema_config_base", node=FactorySchemaConfigBase)

        config_path = (
            "task/IndustRealBase.yaml"  # relative to Gym's Hydra search path (cfg dir)
        )
        self.cfg_base = hydra.compose(config_name=config_path)
        self.cfg_base = self.cfg_base["task"]  # strip superfluous nesting
        asset_info_path = "../assets/industreal/yaml/industreal_asset_info_franka_table.yaml"  # relative to Gym's Hydra search path (cfg dir)
        self.asset_info_franka_table = hydra.compose(config_name=asset_info_path)
        self.asset_info_franka_table = self.asset_info_franka_table[""][""][
            ""
        ]["assets"]["industreal"][
            "yaml"
        ]  # strip superfluous nesting
        
    def _get_env_yaml_params(self, cfg):
        """Initialize instance variables from YAML files."""
        self.cfg_env = cfg['task']

    def _get_task_yaml_params(self, cfg):
        """Initialize instance variables from YAML files."""

        self.cfg_task = cfg.task
        self.max_episode_length = (
            self.cfg_task.rl.max_episode_length
        )  # required instance var for VecTask

        self.cfg_ppo = cfg.train

    def _acquire_env_tensors(self):
        """Acquire and wrap tensors. Create views."""
        self.hit_joint_limits = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    
    def refresh_env_tensors(self):
        """Refresh tensors."""
        arm_dof_ratio = (self.arm_dof_pos - self.franka_dof_lower_limits[:7].unsqueeze(0)) \
            / (self.franka_dof_upper_limits[:7] - self.franka_dof_lower_limits[:7]).unsqueeze(0)
        self.hit_joint_limits = (arm_dof_ratio < 0.001).any(dim=1) | (arm_dof_ratio > 0.999).any(dim=1)

    def create_sim(self):
        """Set sim and PhysX params. Create sim object, ground plane, and envs."""

        if self.cfg_base.mode.export_scene:
            self.sim_params.use_gpu_pipeline = False

        graphics_id = (
            self.device_id
            if (self.render_viewer or self.capture_video or self.local_obs)
            else -1
        )

        # `create_sim` in vec_env only creates sim once in a process
        # but we need to destory it and create new sim in some cases (e.g., filter_vision for initial state sampler)
        self.sim = self.gym.create_sim(
            self.device_id, graphics_id, self.physics_engine, self.sim_params
        )

        self._create_ground_plane()
        self.create_envs()  # defined in subclass
        
    def prepare_init_states(self, init_states):
        """ Prepare initial states for the environment """
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

    def create_viewer_and_camera(self):
        """
        Create viewer and camera sensors.
        A global and a wrist camera are created for each environment.
        """
        # create viewer
        self.viewer = None
        if self.render_viewer:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            cam_pos = gymapi.Vec3(4, 3, 2)
            cam_target = gymapi.Vec3(-4, -3, 0)
            num_per_row = int(math.sqrt(self.num_envs))
            middle_env = self.env_ptrs[self.num_envs // 2 + num_per_row // 2]
            self.gym.viewer_camera_look_at(self.viewer, middle_env, cam_pos, cam_target)
        
        self.camera_handles = []
        self.obs_camera_handles = []
        self.camera_tensors = []
        self.depth_camera_tensors = []
        self.rgb_camera_tensors = []
        self.segmentation_camera_tensors = []

        if self.capture_video:
            camera_properties = gymapi.CameraProperties()
            camera_properties.width = self.cfg["env"]["camera"]["width"]
            camera_properties.height = self.cfg["env"]["camera"]["height"]

        if self.global_obs:
            assert not self.local_obs, "cannot use global and local obs at same time"
            global_obs_camera_properties = gymapi.CameraProperties()
            global_obs_camera_properties.width = self.cfg["env"]["global_obs"]["width"]
            global_obs_camera_properties.height = self.cfg["env"]["global_obs"]["height"]
            global_obs_camera_properties.horizontal_fov = self.cfg["env"]["global_obs"]["horizontal_fov"]
            global_obs_camera_properties.enable_tensors = True
            global_obs_camera_positions = self.cfg["env"]["global_obs"]["locations"]
            global_obs_render_types = self.cfg["env"]["global_obs"]["render_type"]
            print(f"Mounted {len(global_obs_camera_positions)} global camera sensors.")    

        if self.local_obs:
            local_obs_camera_properties = gymapi.CameraProperties()
            local_obs_camera_properties.width = self.cfg["env"]["local_obs"]["width"]
            local_obs_camera_properties.height = self.cfg["env"]["local_obs"]["height"]
            local_obs_camera_properties.horizontal_fov = self.cfg["env"]["local_obs"]["horizontal_fov"]
            local_obs_camera_properties.enable_tensors = True
            local_obs_camera_offset = self.cfg["env"]["local_obs"]["camera_offset"]
            local_obs_camera_angle = self.cfg["env"]["local_obs"]["camera_angle"]
            local_obs_render_types = self.cfg["env"]["local_obs"]["render_type"]
            self.use_back_wrist_camera = self.cfg["env"]["local_obs"]["use_back_wrist_camera"]
            print(f"Attached {2 if self.use_back_wrist_camera else 1} camera sensors to franka wrist.")

        camera_render_modes = {
            "depth": (gymapi.IMAGE_DEPTH, self.depth_camera_tensors),
            "rgb": (gymapi.IMAGE_COLOR, self.rgb_camera_tensors),
            "segmentation": (gymapi.IMAGE_SEGMENTATION, self.segmentation_camera_tensors),
        }

        for i in range(self.num_envs):      
            self.camera_handles.append([])
            self.obs_camera_handles.append([])
            
            if i < self.capture_envs and self.capture_video:
                # global
                camera_handle = self.gym.create_camera_sensor(
                    self.env_ptrs[i], camera_properties
                )
                camera_position = gymapi.Vec3(1.5, 0.0, 1.4)
                camera_target = gymapi.Vec3(
                    self.table_pose.p.x, self.table_pose.p.y, self.cfg_base.env.table_height
                )
                self.gym.set_camera_location(
                    camera_handle, self.env_ptrs[i], camera_position, camera_target
                )
                self.camera_handles[i].append(camera_handle)
                
            if self.local_obs:
                rigid_hand_idx = self.gym.find_actor_rigid_body_index(
                    self.env_ptrs[i], self.franka_actor_id_env, "panda_hand", gymapi.DOMAIN_ENV
                )
                # by default, we have two wrist cameras on both sides of the franka hand
                # if only one wrist camera is used, we use the front wrist camera
                for wrist_idx in range(2 if self.use_back_wrist_camera else 1):
                    wrist_camera_handle = self.gym.create_camera_sensor(
                        self.env_ptrs[i], local_obs_camera_properties
                    )
                    if wrist_idx == 0:
                        # front wrist camera
                        camera_offset = gymapi.Vec3(*local_obs_camera_offset)
                        camera_rotation = gymapi.Quat.from_euler_zyx(0, -math.pi / 2 + local_obs_camera_angle, math.pi)
                    else:
                        # back wrist camera
                        camera_offset = gymapi.Vec3(-0.05, 0.0, 0.0)
                        camera_rotation = gymapi.Quat.from_euler_zyx(0, -math.pi / 2, 0)

                    self.gym.attach_camera_to_body(
                        wrist_camera_handle,
                        self.env_ptrs[i],
                        rigid_hand_idx,
                        gymapi.Transform(camera_offset, camera_rotation),
                        gymapi.FOLLOW_TRANSFORM,
                    )
                    self.camera_handles[i].append(wrist_camera_handle)
                    self.obs_camera_handles[i].append(wrist_camera_handle)

                    for render_type_name, (image_type_id, camera_tensors) in camera_render_modes.items():
                        if local_obs_render_types[render_type_name]:
                            cam_tensor = self.gym.get_camera_image_gpu_tensor(
                                self.sim, self.env_ptrs[i], wrist_camera_handle, image_type_id
                            )  # of shape (camera_width, camera_height)
                            torch_cam_tensor = gymtorch.wrap_tensor(cam_tensor)
                            camera_tensors.append(torch_cam_tensor)
                            self.camera_tensors.append(torch_cam_tensor)

            if self.global_obs:
                camera_target = gymapi.Vec3(
                    self.table_pose.p.x, self.table_pose.p.y, self.cfg_base.env.table_height
                )
                for location in global_obs_camera_positions:
                    global_camera_handle = self.gym.create_camera_sensor(
                        self.env_ptrs[i], global_obs_camera_properties
                    )
                    assert len(location) == 3
                    global_camera_position = gymapi.Vec3(*location)
                    self.gym.set_camera_location(
                        global_camera_handle, self.env_ptrs[i], global_camera_position, camera_target
                    )
                    self.camera_handles[i].append(global_camera_handle)
                    self.obs_camera_handles[i].append(global_camera_handle)

                    for render_type_name, (image_type_id, camera_tensors) in camera_render_modes.items():
                        if global_obs_render_types[render_type_name]:
                            cam_tensor = self.gym.get_camera_image_gpu_tensor(
                                self.sim, self.env_ptrs[i], global_camera_handle, image_type_id
                            )  # of shape (camera_width, camera_height)
                            torch_cam_tensor = gymtorch.wrap_tensor(cam_tensor)
                            camera_tensors.append(torch_cam_tensor)
                            self.camera_tensors.append(torch_cam_tensor)

        # segmentation id
        self.set_segmentation_id()

    def render(self):
        """Render the viewer."""
        if self.render_viewer:
            if self.device != "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)
            self.gym.sync_frame_time(self.sim)

    def set_segmentation_id(self):
        """Set segmentation id for each actor in the scene"""
        pass

    def get_camera_images(self, env_ids=None, wrist_camera=False):
        """Get camera images
        Args:
            wrist_camera: if true, also get images from wrist camera. Otherwise, only get images from global camera
        Returns:
            images: a list of lists of images, each list of images is for one environment
        """
        if (
            not self.capture_video
            or self.total_steps % self.capture_interval >= self.capture_length
        ):
            return None
        if self.capture_obs_camera:
            assert len(self.obs_camera_handles) > 0, "No camera sensors are attached to the environment."
            camera_id = self.obs_camera_handles[0][0]
        else:
            camera_id = self.camera_handles[0][0]

        images, depths, _ = camera_shot(
            self, env_ids=list(range(self.capture_envs)), camera_ids=[camera_id,], use_depth=self.capture_depth
        )
        return images if not self.capture_depth else depths

    def import_franka_assets(self):
        """Set Franka and table asset options. Import assets."""

        # we replaced visual mesh of franka grippers with the original one
        # mesh provided in industreal is very complicated and significantly slows down wrist camera rendering
        urdf_root = os.path.join(
            os.path.dirname(__file__), "..", "assets", "industreal", "urdf"
        )

        franka_file = "industreal_franka_finray_finger.urdf"

        franka_options = gymapi.AssetOptions()
        franka_options.flip_visual_attachments = True
        franka_options.fix_base_link = True
        franka_options.collapse_fixed_joints = False
        franka_options.thickness = 0.0  # default = 0.02
        franka_options.density = 1000.0  # default = 1000.0
        franka_options.armature = 0.01  # default = 0.0
        franka_options.use_physx_armature = True
        if self.cfg_base.sim.add_damping:
            franka_options.linear_damping = (
                1.0  # default = 0.0; increased to improve stability
            )
            franka_options.max_linear_velocity = (
                1.0  # default = 1000.0; reduced to prevent CUDA errors
            )
            franka_options.angular_damping = (
                5.0  # default = 0.5; increased to improve stability
            )
            franka_options.max_angular_velocity = (
                2 * math.pi
            )  # default = 64.0; reduced to prevent CUDA errors
        else:
            franka_options.linear_damping = 0.0  # default = 0.0
            franka_options.max_linear_velocity = 1.0  # default = 1000.0
            franka_options.angular_damping = 0.5  # default = 0.5
            franka_options.max_angular_velocity = 2 * math.pi  # default = 64.0
        franka_options.disable_gravity = True
        franka_options.enable_gyroscopic_forces = True
        franka_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        franka_options.use_mesh_materials = True
        if self.cfg_base.mode.export_scene:
            franka_options.mesh_normal_mode = gymapi.COMPUTE_PER_FACE

        table_options = gymapi.AssetOptions()
        table_options.flip_visual_attachments = False  # default = False
        table_options.fix_base_link = True
        table_options.thickness = 0.0  # default = 0.02
        table_options.density = 1000.0  # default = 1000.0
        table_options.armature = 0.0  # default = 0.0
        table_options.use_physx_armature = True
        table_options.linear_damping = 0.0  # default = 0.0
        table_options.max_linear_velocity = 1000.0  # default = 1000.0
        table_options.angular_damping = 0.0  # default = 0.5
        table_options.max_angular_velocity = 64.0  # default = 64.0
        table_options.disable_gravity = False
        table_options.enable_gyroscopic_forces = True
        table_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        table_options.use_mesh_materials = False
        if self.cfg_base.mode.export_scene:
            table_options.mesh_normal_mode = gymapi.COMPUTE_PER_FACE

        franka_asset = self.gym.load_asset(
            self.sim, urdf_root, franka_file, franka_options
        )
        table_asset = self.gym.create_box(
            self.sim,
            self.asset_info_franka_table.table_depth,
            self.asset_info_franka_table.table_width,
            self.cfg_base.env.table_height,
            table_options,
        )

        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        self.franka_dof_lower_limits = to_torch(
            franka_dof_props["lower"], device=self.device
        )
        self.franka_dof_upper_limits = to_torch(
            franka_dof_props["upper"], device=self.device
        )
        self.gripper_dof_lower_limits = self.franka_dof_lower_limits[-2:]
        self.gripper_dof_upper_limits = self.franka_dof_upper_limits[-2:]
        self.franka_default_dof_pos = to_torch(
            [0, 0.1963, 0, -2.0, 0, 2.3416, 0.7854, 0.04, 0.04], device=self.device
        )

        return franka_asset, table_asset

    def import_unidexgrasp_assets(self):
        """Load unidexgrasp object and set properties"""

        self.object_codes = []
        self.object_scales = []
        object_list = self.cfg["env"]["object_list"]
        if len(object_list) > 0:
            # multi-task mode
            for code, scale in object_list:
                self.object_codes.append(code)
                self.object_scales.append(float(scale))
        else:
            # single-task mode
            self.object_codes.append(self.cfg["env"]["object_code"])
            self.object_scales.append(float(self.cfg["env"]["object_scale"]))

        mesh_path = os.path.join(self.asset_root, "unidexgrasp/meshdatav3_scaled")

        # create object asset
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.fix_base_link = False
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

        object_assets = []
        for code, scale in zip(self.object_codes, self.object_scales):
            scale_str = "{:03d}".format(int(100 * scale))
            scaled_object_asset_file = f"{code}/coacd/coacd_{scale_str}.urdf"
            scaled_object_asset = self.gym.load_asset(
                self.sim, mesh_path, scaled_object_asset_file, object_asset_options
            )
            object_assets.append(scaled_object_asset)

        self.num_objects = len(object_assets)

        return object_assets

    def load_unidexgrasp_data(
        self, 
        object_code, 
        object_scale, 
        load_grasp_poses=False,
        load_finger_pos=False,
        load_mesh_points=False,
        load_convex_hull_points=False,
        load_object_keypoints=False,
        load_rest_object_pose=False,
        filter_threshold=0.02,
        num_grasp_poses=2048,
        num_convex_hull_points=512,
        num_mesh_points=256,
    ):
        """Process unidexgrasp data for the given object code and scale
        Args:
            object_code: object code
            object_scale: object scale
            load_grasp_poses: whether to load pre-sampled grasp poses
            load_finger_pos: whether to load pre-sampled finger positions
            load_mesh_points: whether to sample points on the object mesh
            load_convex_hull_points: whether to sample points on the convex hull of the object
            load_object_keypoints: whether to sample keypoints on the object
            filter_threshold: threshold to filter bad rest poses based on grasp success rate
            num_grasp_poses: number of grasp poses to sample (effective only when load_grasp_poses or load_finger_pos is True)
            num_convex_hull_points: number of points to sample on the convex hull of the object
            num_mesh_points: number of points to sample on the object mesh
        """
        # load object mesh
        mesh_path = os.path.join(self.asset_root, "unidexgrasp/meshdatav3_scaled")
        self.object_mesh = tm.load(
            os.path.join(
                mesh_path,
                f"{object_code}/coacd/decomposed_{int(100 * object_scale):03d}.obj",
            )
        )

        # load grasp data
        franka_grasp_data_path = os.path.join(
            self.asset_root,
            "unidexgrasp/graspdata",
            object_code.replace("/", "-") + f"-{int(100 * object_scale):03d}.npy",
        )
        franka_grasp_data = None
        if os.path.exists(franka_grasp_data_path):
            franka_grasp_data = np.load(
                franka_grasp_data_path, allow_pickle=True
            ).item()

            # filter initial object poses where the grasp success rate is lower than a threshold
            # some rest poses are not suitable for franka grippers
            filter_bad_pose = (
                franka_grasp_data["success"] / franka_grasp_data["trials"]
            ) > filter_threshold
            self.grasp_data = {object_code: {object_scale: {}}}
            self.grasp_data[object_code][object_scale]["object_euler_xy"] = torch.from_numpy(
                franka_grasp_data["object_euler_xy"][filter_bad_pose]
            ).float().to(self.device)
            self.grasp_data[object_code][object_scale]["object_init_z"] = torch.from_numpy(
                franka_grasp_data["object_z"][filter_bad_pose]
            ).float().to(self.device)
            print("Filtered bad rest poses:", len(filter_bad_pose) - filter_bad_pose.sum())
        else:
            # if the grasp data is not available (this should only happen when generating the grasp dataset),
            # use the original grasp data from unidexgrasp
            # we only load rest pose of the object
            shadow_hand_grasp_data = np.load(os.path.join(self.asset_root, "unidexgrasp/datasetv4.1_posedata.npy"), allow_pickle=True).item()
            self.grasp_data = {self.object_code: {self.object_scale: {}}}
            self.grasp_data[self.object_code][self.object_scale]["object_euler_xy"] = torch.from_numpy(
                shadow_hand_grasp_data[self.object_code][self.object_scale]["object_euler_xy"]
            ).float().to(self.device)
            self.grasp_data[self.object_code][self.object_scale]["object_init_z"] = torch.from_numpy(
                shadow_hand_grasp_data[self.object_code][self.object_scale]["object_init_z"]
            ).float().to(self.device).reshape(-1)
        
        if load_grasp_poses:
            # use grasp points to compute distance
            if franka_grasp_data is None:
                assert self.sample_grasp_pose
                self.grasp_pose = torch.zeros((1, 7), device=self.device)
                self.grasp_pose[0, 3] = 1.0
            else:
                grasp_pose_np = sample_grasp_points(franka_grasp_data, self.object_mesh, num_grasp_poses, uniform=True, rotation=True, threshold=filter_threshold)
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
                finger_pos_np = sample_grasp_points(franka_grasp_data, self.object_mesh, num_grasp_poses, uniform=False, finger_pos=True, threshold=filter_threshold)
                self.finger_pos = torch.from_numpy(finger_pos_np).float().to(self.device)                       # (num_finger_points, 6)

        if load_rest_object_pose:
            # save rest object pose
            self.rest_object_euler_xy = self.grasp_data[self.object_code][self.object_scale]["object_euler_xy"]
            self.rest_object_init_z = self.grasp_data[self.object_code][self.object_scale]["object_init_z"]

        if load_convex_hull_points:
            # sample points on the convex hull of the object
            convex_hull_points_np = sample_mesh_points_on_convex_hull(self.object_mesh, num_convex_hull_points)
            self.convex_hull_points = torch.from_numpy(convex_hull_points_np).float().to(self.device)

        if load_object_keypoints:
            # sample keypoints on the object
            key_points_np = sample_mesh_keypoints(self.object_mesh, self.num_object_keypoints)
            self.object_keypoints_local = torch.from_numpy(key_points_np).float().to(self.device)           # (num_object_keypoints, 3)

        if load_mesh_points:
            # randomly sample points on the object mesh
            object_mesh_points_np = sample_mesh_points(self.object_mesh, num_mesh_points)
            self.object_mesh_points = torch.from_numpy(object_mesh_points_np).float().to(self.device)                       # (num_mesh_points, 3)

    def allocate_buffers(self):
        super().allocate_buffers()
        self.success_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float
        )
        self.consecutive_successes = torch.zeros(
            1, device=self.device, dtype=torch.float
        )
    
    def _reset_franka(self, env_ids, franka_init_dof_pos=None):
        """Reset DOF states, DOF torques, and DOF targets of Franka.
        Args:
            env_ids: environment ids to reset
            franka_init_dof_pos: presampled DOF positions of Franka.
                use default arm and gripper pos if None
        """
        # Randomize DOF pos
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
        self.ctrl_target_dof_pos[env_ids] = self.dof_pos[env_ids].clone()
        self.ctrl_target_fingertip_centered_pos[env_ids] = self.fingertip_centered_pos[env_ids].clone()
        self.ctrl_target_fingertip_centered_quat[env_ids] = self.fingertip_centered_quat[env_ids].clone()

        # Set DOF state
        franka_actor_ids_sim = self.franka_actor_ids_sim[env_ids].clone().to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(franka_actor_ids_sim),
            len(franka_actor_ids_sim),
        )

        # Set DOF torque
        self.gym.set_dof_actuation_force_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(torch.zeros_like(self.dof_torque)),
            gymtorch.unwrap_tensor(franka_actor_ids_sim),
            len(franka_actor_ids_sim),
        )

    def pre_physics_step(self, actions):
        """Reset environments. Apply actions from policy as position/rotation targets, force/torque targets, and/or PD gains."""

        self.actions = actions.clone().to(
            self.device
        )  # shape = (num_envs, num_actions); values = [-1, 1]

        gripper_target = 0.08 if self.cfg_task.randomize.franka_gripper_initial_state == 1 else -0.04
        self._apply_actions_as_ctrl_targets(
            actions=self.actions, 
            ctrl_target_gripper_dof_pos=gripper_target, 
            do_scale=True
        )
    
    def _apply_actions_as_ctrl_targets(
        self, actions, ctrl_target_gripper_dof_pos, do_scale
    ):
        """Apply actions from policy as position/rotation targets."""

        # Interpret actions as target pos displacements and set pos target
        pos_actions = actions[:, 0:3]
        if do_scale:
            pos_actions = pos_actions @ torch.diag(
                torch.tensor(self.cfg_task.rl.pos_action_scale, device=self.device)
            )
        self.ctrl_target_fingertip_centered_pos = (
            self.fingertip_centered_pos + pos_actions
        )

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = actions[:, 3:6]
        if do_scale:
            rot_actions = rot_actions @ torch.diag(
                torch.tensor(self.cfg_task.rl.rot_action_scale, device=self.device)
            )

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)
        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        if self.cfg_task.rl.clamp_rot:
            rot_actions_quat = torch.where(
                angle.unsqueeze(-1).repeat(1, 4) > self.cfg_task.rl.clamp_rot_thresh,
                rot_actions_quat,
                torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(
                    self.num_envs, 1
                ),
            )
        self.ctrl_target_fingertip_centered_quat = torch_utils.quat_mul(
            rot_actions_quat, self.fingertip_centered_quat
        )

        self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos

        self.generate_ctrl_signals()

    def post_physics_step(self):
        """Step buffers. Refresh tensors. Compute observations and reward."""

        self.progress_buf[:] += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.refresh_base_tensors()
        self.refresh_env_tensors()
        self._refresh_task_tensors()

        if self.export_rigid_body_poses_dir:
            self.export_rigid_body_poses_to_file()

        self.compute_observations()
        self.compute_reward()

    def step(
        self, actions: torch.Tensor, reset_on_success: bool = False, get_camera_images=True
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the physics of the environment.
        Args:
            actions: actions to apply
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs', visual inputs can be added later)
        """
        action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # apply actions
        self.pre_physics_step(action_tensor)

        # step physics
        self.gym.simulate(self.sim)
        if self.device == "cpu":
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()

        # render each frame
        self.render()
        
        if get_camera_images:
            camera_images = self.get_camera_images()
        else:
            camera_images = None
        
        # rlgames need this value for value bootstrapping
        self.timeout_buf = (self.progress_buf >= self.max_episode_length - 1) & (self.reset_buf != 0)
        # terminate on success for bc:
        if reset_on_success:
            self.reset_buf = ((self.success_buf == 1.0) | self.reset_buf)
        self.update_extras(camera_images)
        self.obs_dict["eef_pos"] = self.global_eef_pos
        self.obs_dict["eef_quat"] = self.global_eef_quat
        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs)
        self.total_steps += 1

        return self.obs_dict, self.rew_buf, self.reset_buf, self.extras
    
    def update_extras(self, camera_images):
        self.extras["time_outs"] = self.timeout_buf
        self.extras["success"] = self.success_buf
        self.extras["consecutive_successes"] = self.consecutive_successes
        self.extras["camera_images"] = camera_images
        self.extras["total_steps"] = self.total_steps
        self.extras["hit_joint_limits"] = self.hit_joint_limits
        
    def move_gripper_to_target_pose(self, gripper_dof_pos, sim_steps, action_scale=0.2, stablize=True, get_camera_images=False):
        """Move gripper to control target pose."""

        camera_images_list = []

        for _ in range(sim_steps):
            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_centered_pos,
                fingertip_midpoint_quat=self.fingertip_centered_quat,
                ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
                ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
                jacobian_type=self.cfg_ctrl["jacobian_type"],
                rot_error_type="axis_angle",
            )

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)
            actions = torch.zeros(
                (self.num_envs, self.cfg_task.env.numActions), device=self.device
            )
            actions[:, :6] = delta_hand_pose * action_scale

            self._apply_actions_as_ctrl_targets(
                actions=actions,
                ctrl_target_gripper_dof_pos=gripper_dof_pos,
                do_scale=False,
            )

            # Simulate one step
            self.simulate_and_refresh()

            if get_camera_images:
                camera_images = self.get_camera_images()
                camera_images_list.append(camera_images)

            if self.export_rigid_body_poses_dir:
                self.export_rigid_body_poses_to_file()

        if stablize:
            # Stabilize Franka
            self.dof_vel[:, :self.franka_num_dofs] = 0.0
            self.dof_torque[:, :self.franka_num_dofs] = 0.0
            self.ctrl_target_fingertip_centered_pos = self.fingertip_centered_pos.clone()
            self.ctrl_target_fingertip_centered_quat = self.fingertip_centered_quat.clone()

            # Set DOF state
            franka_actor_ids_sim = self.franka_actor_ids_sim.clone().to(dtype=torch.int32)
            self.gym.set_dof_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.dof_state),
                gymtorch.unwrap_tensor(franka_actor_ids_sim),
                len(franka_actor_ids_sim),
            )

            # Set DOF torque
            self.gym.set_dof_actuation_force_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.dof_torque),
                gymtorch.unwrap_tensor(franka_actor_ids_sim),
                len(franka_actor_ids_sim),
            )

            # Simulate one step to apply changes
            self.simulate_and_refresh()

        return camera_images_list
    
    def enable_gravity(self, gravity=None):
        """Enable gravity. The implementation in industreal cannot specify gravity manually
        We need it to check whether the grasp is robust by changing direction of gravity."""

        sim_params = self.gym.get_sim_params(self.sim)
        if gravity is None:
            gravity = self.cfg_base.sim.gravity
        sim_params.gravity = gymapi.Vec3(*gravity)
        self.gym.set_sim_params(self.sim, sim_params)

    def update_franka_visual(self, env_id, color=None):
        """Update franka visual color
        Args:
            env_id: update franka visual in environment `env_id`
            color: color to set, if None, reset to original color
        """
        if color is None:
            self.gym.reset_actor_materials(
                self.env_ptrs[env_id],
                self.franka_handles[env_id],
                gymapi.MESH_VISUAL_AND_COLLISION,
            )
        else:
            num_rigid_bodies = self.gym.get_actor_rigid_body_count(
                self.env_ptrs[env_id], self.franka_handles[env_id]
            )
            for i in range(num_rigid_bodies):
                self.gym.set_rigid_body_color(
                    self.env_ptrs[env_id],
                    self.franka_handles[env_id],
                    i,
                    gymapi.MESH_VISUAL_AND_COLLISION,
                    gymapi.Vec3(color[0], color[1], color[2]),
                )

    def check_contact(self, rigid_body_idxs):
        """Check whether the rigid bodies specified by `rigid_body_idxs` has non-zero contact force"""
        contact = (self.contact_force.view(-1, 3)[rigid_body_idxs, :].view(self.num_envs, -1) != 0.0).any(
            dim=-1
        )
        return contact
    
    def get_delta_eef_pose(self, cur_eef_pos, cur_eef_quat, prev_eef_pos, prev_eef_quat, scale=60.0):
        """ Compute delta proprioception to approximate velocity information. """

        delta_pos = (cur_eef_pos - prev_eef_pos) * scale
        delta_quat = quat_mul(cur_eef_quat, quat_conjugate(prev_eef_quat))

        axis_angle = axis_angle_from_quat(delta_quat)
        delta_rot = axis_angle * scale

        delta_eef_pose = torch.cat([delta_pos, delta_rot], dim=-1)
        return delta_eef_pose
    
    def _format_quaternion(self, quat):
        """Format quaternion to ensure w >= 0"""
        # Ensure the discontinuity in quaternion occurs at the initial eef orientation (pointing upwards).
        # In this case, we will hardly reach the discontinuity in the quaternion space in practice.
        down_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand(self.num_envs, -1)
        quat = quat_mul(quat, down_q)

        # Follow specific convention w >= 0.
        # This is to ensure that the quaternion is unique.
        flip = quat[:, 3] < 0
        quat[flip] = -quat[flip]

        return quat

    def destroy(self):
        """Destroy simulation and viewer. Only one Isaac Gym instance can exist at a time."""
        if self.render_viewer:
            self.gym.destroy_viewer(self.viewer)

        for env_id, camera_handles in enumerate(self.camera_handles):
            for camera_handle in camera_handles:
                self.gym.destroy_camera_sensor(self.sim, self.env_ptrs[env_id], camera_handle)

        self.gym.destroy_sim(self.sim)

    def get_segmentation_observations(self):
        """
        Get segmentation masks for object to manipulate from camera sensors
        """
        assert self.local_obs or self.global_obs

        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)

        self.gym.start_access_image_tensors(self.sim)

        # We ensure that object to manipulate always has segmentation id 3.
        # In articulated object tasks, 3 is id for the handle part.
        object_seg_id = 3

        if self.global_obs:
            height, width = self.cfg['env']['global_obs']['height'], self.cfg['env']['global_obs']['width']
        else:  # self.local_obs
            height, width = self.cfg['env']['local_obs']['height'], self.cfg['env']['local_obs']['width']
        segmentation_camera_buf_tensor = torch.stack(self.segmentation_camera_tensors, dim=0).reshape(self.num_envs, -1, height, width)
        segmentation_camera_buf_tensor = segmentation_camera_buf_tensor == object_seg_id
        self.segmentation_camera_buf = segmentation_camera_buf_tensor

        self.gym.end_access_image_tensors(self.sim)

        self.segmentation_camera_buf = self.segmentation_camera_buf.float()

        new_seg_obs = self.segmentation_camera_buf

        if self.enable_camera_latency:
            # return the previous segmentation observation
            seg_obs = copy.deepcopy(new_seg_obs) if self.prev_seg_obs is None else self.prev_seg_obs
            self.prev_seg_obs = new_seg_obs
            return seg_obs
        else:
            return new_seg_obs

    def get_depth_observations(self, for_point_cloud=False):
        """
        Get depth observations from camera sensors
        NOTE: this function assumes that the camera frequency is the same as the simulation frequency, and the depth image is captured at the same time as the simulation step
        """
        assert self.local_obs or self.global_obs

        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)

        self.gym.start_access_image_tensors(self.sim)

        # TODO: add code to check camera frequency here (no need to check, since sim and depth frequency are same now)
        # assert 1. / self.camera_frequency == round(self.dt, 2), "Camera and simulation frequency are not same; query depth as per frequency of simulation"
        if self.global_obs:
            height, width = self.cfg['env']['global_obs']['height'], self.cfg['env']['global_obs']['width']
        else:  # self.local_obs
            height, width = self.cfg['env']['local_obs']['height'], self.cfg['env']['local_obs']['width']
            clamp_depth = self.cfg['env']['local_obs']['clamp_depth']
        depth_image_buf_tensor = torch.stack(self.depth_camera_tensors, dim=0).reshape(self.num_envs, -1, height, width)

        depth_image_buf_tensor = torch.where(
            depth_image_buf_tensor == -inf,
            -255.0,
            depth_image_buf_tensor.double(),
        )
        depth_image_buf_tensor = torch.where(
            depth_image_buf_tensor == inf, -255.0, depth_image_buf_tensor
        )
        if not for_point_cloud:
            depth_image_buf_tensor *= -1.0
            depth_image_buf_tensor = torch.clamp(
                depth_image_buf_tensor, 0.0, clamp_depth
            )

        self.depth_image_buf = depth_image_buf_tensor

        if for_point_cloud:
            # also add projection matrices
            camera_proj_mats, camera_view_mats = [], []
            for env, camera_handles in zip(self.env_ptrs, self.obs_camera_handles):
                for cam_handle in camera_handles:
                    view_mat = torch.tensor(self.gym.get_camera_view_matrix(self.sim, env, cam_handle))
                    camera_view_mats.append(view_mat)
                    proj_mat = torch.tensor(self.gym.get_camera_proj_matrix(self.sim, env, cam_handle))
                    camera_proj_mats.append(proj_mat)
            self.camera_proj_mats = torch.stack(camera_proj_mats, dim=0).reshape(self.num_envs, -1, 4, 4).float()
            camera_view_mats = torch.stack(camera_view_mats, dim=0).reshape(self.num_envs, -1, 4, 4).float()
            # We observed an issue where, for global camera positions, the cameras'
            # positions were varying (somewhat linearly) across parallel simulations.
            # As a result, we temporarily fix this issue by setting the camera positions
            # (within their view transformation matrices) to the positions of the camera
            # for simulation idx=0, which appears to have the correct positions.
            camera_view_mats[:, :, -1, :3] = camera_view_mats[0, :, -1, :3]  # reset camera positions for envs 1+
            self.camera_view_mats = camera_view_mats

        # self.depth_image_buf = fn.resize(depth_image_buf_tensor, size=(self.cfg.env.local_obs.width, self.cfg.env.local_obs.height))
        self.gym.end_access_image_tensors(self.sim)

        if not for_point_cloud:
            # normalize for values to be between 0 and 1
            self.depth_image_buf = self.depth_image_buf / clamp_depth

        self.depth_image_buf = self.depth_image_buf.float()

        additional_data = []
        if len(self.rgb_camera_tensors) > 0:
            rgb_tensor = torch.stack(self.rgb_camera_tensors, dim=0).reshape(self.num_envs, -1, height, width, 4)
            additional_data += [rgb_tensor]
        else:
            rgb_tensor = None
        seg_tensor = None

        if for_point_cloud:
            new_visual_obs = (
                self.depth_image_buf,
                self.camera_proj_mats,
                self.camera_view_mats,
                self.global_eef_pos,
                self.global_eef_quat,
                additional_data
            )
        else:
            new_visual_obs = (
                self.depth_image_buf,
                seg_tensor,
                rgb_tensor,
            )

        if self.enable_camera_latency:
            # return the previous visual observation
            visual_obs = copy.deepcopy(new_visual_obs) if self.prev_visual_obs is None else self.prev_visual_obs
            self.prev_visual_obs = new_visual_obs
            return visual_obs
        else:
            return new_visual_obs

    def export_rigid_body_poses_to_file(self):
        """Export rigid body poses to file"""

        if not os.path.exists(self.export_rigid_body_poses_dir):
            os.makedirs(self.export_rigid_body_poses_dir)
            self.rigid_body_poses_id = 0

        euler = get_euler_xyz(self.body_quat.reshape(-1, 4))
        euler = np.concatenate([euler[i].reshape(self.num_envs, self.num_bodies, 1).cpu().numpy() for i in range(3)], axis=-1)

        rigid_body_poses = {
            "position": self.body_pos.cpu().numpy(),
            "rotation": euler,
        }
        np.save(
            os.path.join(self.export_rigid_body_poses_dir, f"{self.rigid_body_poses_id:05d}.npy"),
            rigid_body_poses,
        )

        self.rigid_body_poses_id += 1
