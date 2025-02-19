import hydra
from isaacgym import gymapi, gymtorch, gymutil, torch_utils
from gym import spaces
import numpy as np
import torch

from typing import Dict, Any, Tuple, List, Set
import math

import isaacgymenvs.tasks.industreal.industreal_algo_utils as algo_utils
from isaacgymenvs.utils import torch_jit_utils
import isaacgymenvs.tasks.factory.factory_control as fc
from manipgen.envs.franka_env import FrankaEnv
from manipgen.utils.geometry_utils import get_keypoint_offsets_6d

class FrankaReach(FrankaEnv):
    def __init__(
        self,
        cfg,
        init_states=None,
        **kwargs,
    ):
        """
        Args:
            num_envs: number of environments
            device: device to run the environments
            render: if true, open GUI to view the simulation process
            capture_video: if true, set up camera sensors
            capture_interval: capture videos every N steps
            capture_length: capture video for M steps
            capture_envs: number of environments to capture videos
            init_states: pre-sampled initial states for reset
            cfg: configuration for the environment
        """        
        self._get_env_yaml_params(cfg)
        self._get_task_yaml_params(cfg)

        super().__init__(cfg, **kwargs)
        # max episode length in policy learning
        self.max_episode_length = self.cfg["env"]["episode_length"]

        # observation space
        self.num_observations = self.cfg['env']['numObservations']
        self.obs_space = spaces.Box(
            np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf
        )

        # action space
        self.num_actions = self.cfg['env']['numActions']
        self.act_space = spaces.Box(
            np.ones(self.num_actions) * -1.0, np.ones(self.num_actions) * 1.0
        )

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
        self.reward_settings = {}
        
        # Reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def _acquire_task_tensors(self):
        """Acquire tensors."""

        self.identity_quat = (
            torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )

        # Define keypoint tensors
        if self.cfg_task.rl.use_keypoints_6D:
            self.keypoint_offsets = get_keypoint_offsets_6d(self.device) * self.cfg_task.rl.keypoint_scale
        else:
            self.keypoint_offsets = (
                algo_utils.get_keypoint_offsets(self.cfg_task.rl.num_keypoints, self.device)
                * self.cfg_task.rl.keypoint_scale
            )
        self.keypoints_gripper = torch.zeros((self.num_envs, self.keypoint_offsets.shape[0], 3),
                                             dtype=torch.float32,
                                             device=self.device)
        self.actions = torch.zeros(
            (self.num_envs, self.cfg_task.env.numActions), device=self.device
        )
        self.keypoints_target = torch.zeros_like(self.keypoints_gripper, device=self.device)
        
        self.target_pos = torch.tensor([0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.target_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        
        # record previous eef pose
        self.prev_local_eef_pos = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.prev_local_eef_quat = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)

    def _refresh_task_tensors(self):
        # Compute pos of keypoints on gripper and target in world frame
        for idx, keypoint_offset in enumerate(self.keypoint_offsets):
            _, self.keypoints_gripper[:, idx] = torch_jit_utils.tf_combine(self.fingertip_centered_quat,
                                                                           self.fingertip_centered_pos,
                                                                           self.identity_quat,
                                                                           keypoint_offset.repeat(self.num_envs, 1))
            _, self.keypoints_target[:, idx] = torch_jit_utils.tf_combine(self.target_quat,
                                                                          self.target_pos,
                                                                          self.identity_quat,
                                                                          keypoint_offset.repeat(self.num_envs, 1))
    
    def _reset_target(self, env_ids):
        """Reset target."""

        # Set target pos in robot frame
        random_pos_x = self.cfg_task.randomize.rl_workspace_bounds[0][0] + \
                       torch.rand((self.num_envs, 1), dtype=torch.float32, device=self.device) \
                        * (self.cfg_task.randomize.rl_workspace_bounds[0][1] - self.cfg_task.randomize.rl_workspace_bounds[0][0])
        random_pos_y = self.cfg_task.randomize.rl_workspace_bounds[1][0] + \
                       torch.rand((self.num_envs, 1), dtype=torch.float32, device=self.device) \
                        * (self.cfg_task.randomize.rl_workspace_bounds[1][1] - self.cfg_task.randomize.rl_workspace_bounds[1][0])
        random_pos_z = self.cfg_task.randomize.rl_workspace_bounds[2][0] + \
                       torch.rand((self.num_envs, 1), dtype=torch.float32, device=self.device) \
                        * (self.cfg_task.randomize.rl_workspace_bounds[2][1] - self.cfg_task.randomize.rl_workspace_bounds[2][0])
        random_pos = torch.hstack((random_pos_x, random_pos_y, random_pos_z))
        self.target_pos[env_ids] = (torch.tensor(self.robot_base_pos[0], device=self.device) + random_pos)[env_ids]

        # Generate random target quat, pointing down and with random spin
        target_rot_euler = torch.tensor(self.cfg_task.randomize.target_rot_initial,
                                        device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

        target_rot_noise = 2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        target_rot_noise = target_rot_noise @ torch.diag(torch.tensor(self.cfg_task.randomize.target_rot_noise_level,
                                                                      device=self.device))
        target_rot_euler += target_rot_noise
        self.target_quat[env_ids] = torch_utils.quat_from_euler_xyz(target_rot_euler[:, 0],
                                                           target_rot_euler[:, 1],
                                                           target_rot_euler[:, 2])[env_ids]
        
    def create_envs(self):
        """Set env options. Import assets. Create actors."""

        lower = gymapi.Vec3(
            -self.cfg_base.env.env_spacing, -self.cfg_base.env.env_spacing, 0.0
        )
        upper = gymapi.Vec3(
            self.cfg_base.env.env_spacing,
            self.cfg_base.env.env_spacing,
            self.cfg_base.env.env_spacing,
        )
        num_per_row = int(np.sqrt(self.num_envs))
        self.print_sdf_warning()
        franka_asset, table_asset = self.import_franka_assets()
        self._create_actors(
            lower,
            upper,
            num_per_row,
            franka_asset,
            table_asset,
        )
        
    def _create_actors(
        self,
        lower,
        upper,
        num_per_row,
        franka_asset,
        table_asset,
    ):
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

        self.env_ptrs = []
        self.franka_handles = []
        self.table_handles = []
        self.shape_ids = []
        self.franka_actor_ids_sim = []  # within-sim indices
        self.table_actor_ids_sim = []  # within-sim indices
        actor_count = 0

        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            if self.cfg_env.sim.disable_franka_collisions:
                franka_handle = self.gym.create_actor(
                    env_ptr,
                    franka_asset,
                    franka_pose,
                    "franka",
                    i + self.num_envs,
                    1,
                    0,
                )
            else:
                franka_handle = self.gym.create_actor(
                    env_ptr, franka_asset, franka_pose, "franka", i, 1, 0
                )
            self.franka_actor_ids_sim.append(actor_count)
            actor_count += 1

            table_handle = self.gym.create_actor(
                env_ptr, table_asset, table_pose, "table", i, 2, 0
            )
            self.table_actor_ids_sim.append(actor_count)
            actor_count += 1

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

            self.franka_num_dofs = self.gym.get_actor_dof_count(env_ptr, franka_handle)

            self.gym.enable_actor_dof_force_sensors(env_ptr, franka_handle)

            self.env_ptrs.append(env_ptr)
            self.franka_handles.append(franka_handle)
            self.table_handles.append(table_handle)

        self.num_actors = int(actor_count / self.num_envs)  # per env
        self.num_bodies = self.gym.get_env_rigid_body_count(env_ptr)  # per env
        self.num_dofs = self.gym.get_env_dof_count(env_ptr)  # per env

        # For setting targets
        self.franka_actor_ids_sim = torch.tensor(
            self.franka_actor_ids_sim, dtype=torch.int32, device=self.device
        )

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

        # global indices
        num_actors = 2
        self.global_indice = torch.arange(
            self.num_envs * num_actors, dtype=torch.int32, device=self.device
        ).view(self.num_envs, -1)
        self.actor_table_idx = 1
        self.actor_franka_idx = 0

    def set_segmentation_id(self):
        for env_id in range(self.num_envs):
            env = self.env_ptrs[env_id]
            num_franka_rigid_bodies = self.gym.get_actor_rigid_body_count(
                env, self.franka_handles[env_id]
            )
            for idx in range(num_franka_rigid_bodies):
                self.gym.set_rigid_body_segmentation_id(
                    env, self.actor_franka_idx, idx, self.actor_franka_idx + 1
                )
            self.gym.set_rigid_body_segmentation_id(
                env, self.actor_table_idx, 0, self.actor_table_idx + 1
            )

    def compute_reward(self):
        self.rew_buf[:], reset, self.success_buf[:], self.consecutive_successes[:], dist, keypoint_dist = (
            compute_franka_reward(
                self.reset_buf,
                self.progress_buf,
                self.success_buf,
                self.consecutive_successes,
                self.max_episode_length,
                self.keypoints_gripper,
                self.keypoints_target,
                self.fingertip_centered_pos,
                self.target_pos,
            )
        )
        if not self.disable_automatic_reset:
            self.reset_buf[:] = reset
        # In this policy, episode length is constant
        is_last_step = (self.progress_buf[0] == self.max_episode_length - 1)
        if is_last_step:
            print("Final dist (policy): ", dist)
            print("Final keypoint dist (policy): ", keypoint_dist)
            self.extras['dist'] = dist
            self.extras['keypoint_dist'] = keypoint_dist

    def compute_observations(self):
        self.global_eef_pos = self.fingertip_centered_pos       # position of the gripper
        self.global_eef_quat = self.fingertip_centered_quat     # orientation of the gripper

        local_eef_pos, local_eef_quat = self.pose_world_to_robot_base(
            self.global_eef_pos,    
            self.global_eef_quat, 
        )

        local_target_eef_pos, local_target_eef_quat = self.pose_world_to_robot_base(
            self.target_pos,
            self.target_quat,
        )

        local_eef_quat = self._format_quaternion(local_eef_quat)
        local_target_eef_quat = self._format_quaternion(local_target_eef_quat)

        delta_eef_pose = self.get_delta_eef_pose(
            local_eef_pos, local_eef_quat,
            self.prev_local_eef_pos, self.prev_local_eef_quat,
        )

        obs_tensors = [
            self.arm_dof_pos,           # 7
            local_eef_pos,              # 3
            local_eef_quat,             # 4
            delta_eef_pose,             # 6
            local_target_eef_pos,       # 3
            local_target_eef_quat,      # 4
        ]

        self.obs_buf = torch.cat(obs_tensors, dim=-1)
        self.prev_local_eef_pos = local_eef_pos
        self.prev_local_eef_quat = local_eef_quat

        return self.obs_buf

    def apply_random_controller_gains(self, env_ids):
        low = torch.tensor(self.cfg['env']['kp_range_low'], device=self.device).float()
        high = torch.tensor(self.cfg['env']['kp_range_high'], device=self.device).float()
        self.kp[env_ids] =  torch.distributions.uniform.Uniform(low, high).sample([len(env_ids)]) 
        self.kd[env_ids] = 2.0 * torch.sqrt(self.kp[env_ids])
    
    def _randomize_gripper_pose(self, env_ids, sim_steps):
        """Move gripper to random pose."""

        # Set target pos in robot frame
        random_pos_x = self.cfg_task.randomize.ctrl_workspace_bounds[0][0] + \
                       torch.rand((self.num_envs, 1), dtype=torch.float32, device=self.device) \
                        * (self.cfg_task.randomize.ctrl_workspace_bounds[0][1] - self.cfg_task.randomize.ctrl_workspace_bounds[0][0])
        random_pos_y = self.cfg_task.randomize.ctrl_workspace_bounds[1][0] + \
                       torch.rand((self.num_envs, 1), dtype=torch.float32, device=self.device) \
                        * (self.cfg_task.randomize.ctrl_workspace_bounds[1][1] - self.cfg_task.randomize.ctrl_workspace_bounds[1][0])
        random_pos_z = self.cfg_task.randomize.ctrl_workspace_bounds[2][0] + \
                       torch.rand((self.num_envs, 1), dtype=torch.float32, device=self.device) \
                        * (self.cfg_task.randomize.ctrl_workspace_bounds[2][1] - self.cfg_task.randomize.ctrl_workspace_bounds[2][0])
        random_pos = torch.hstack((random_pos_x, random_pos_y, random_pos_z))
        self.ctrl_target_fingertip_centered_pos = torch.tensor(self.robot_base_pos[0], device=self.device) + random_pos

        # Set target rot
        ctrl_target_fingertip_centered_euler = torch.tensor(self.cfg_task.randomize.fingertip_centered_rot_initial,
                                                            device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

        fingertip_centered_rot_noise = 2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        fingertip_centered_rot_noise = fingertip_centered_rot_noise \
                                        @ torch.diag(torch.tensor(self.cfg_task.randomize.fingertip_centered_rot_noise, device=self.device))
        ctrl_target_fingertip_centered_euler += fingertip_centered_rot_noise
        self.ctrl_target_fingertip_centered_quat = torch_utils.quat_from_euler_xyz(
            ctrl_target_fingertip_centered_euler[:, 0],
            ctrl_target_fingertip_centered_euler[:, 1],
            ctrl_target_fingertip_centered_euler[:, 2])
        
        # Step sim and render
        target_pos = self.ctrl_target_fingertip_centered_pos.clone().to(self.device)
        target_quat = self.ctrl_target_fingertip_centered_quat.clone().to(self.device)
        for _ in range(sim_steps):
            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_centered_pos,
                fingertip_midpoint_quat=self.fingertip_centered_quat,
                ctrl_target_fingertip_midpoint_pos=target_pos,
                ctrl_target_fingertip_midpoint_quat=target_quat,
                jacobian_type=self.cfg_ctrl['jacobian_type'],
                rot_error_type='axis_angle')
            
            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)
            actions = torch.zeros((self.num_envs, self.cfg_task.env.numActions), device=self.device)
            actions[:, :6] = delta_hand_pose

            self._apply_actions_as_ctrl_targets(actions=actions,
                                                ctrl_target_gripper_dof_pos=self.gripper_dof_pos,
                                                do_scale=False,
                                                )
            
            self.simulate_and_refresh()

            if self.cfg_task.debug.visualize:
                self.target_pos = self.ctrl_target_fingertip_centered_pos.clone().to(self.device)
                self.target_quat = self.ctrl_target_fingertip_centered_quat.clone().to(self.device)
                self._draw_debug_output(target_color=(0.0, 0.0, 1.0))
            
        print("Final pos error (reset): ", torch.norm(pos_error, p=2, dim=-1).mean())
        print("Final rot error (reset): ", torch.norm(axis_angle_error, p=2, dim=-1).mean())

        self.dof_vel[env_ids, :] = torch.zeros_like(self.dof_vel[env_ids])

        # Set DOF state
        multi_env_ids_int32 = self.franka_actor_ids_sim[env_ids].flatten()
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))

        # Set DOF torque
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                gymtorch.unwrap_tensor(torch.zeros_like(self.dof_torque)),
                                                gymtorch.unwrap_tensor(self.franka_actor_ids_sim),
                                                len(self.franka_actor_ids_sim))
        
    def reset_idx(self, env_ids: torch.Tensor, validation_set: bool = False):
        """Reset environments having the provided indices.
        Args:
            env_ids: environments to reset
        """
        self._reset_franka(env_ids)
        self._randomize_gripper_pose(env_ids, sim_steps=self.cfg_task.env.num_gripper_move_sim_steps)

        self._reset_target(env_ids)
        
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        local_eef_pos, local_eef_quat = self.pose_world_to_robot_base(
            self.fingertip_centered_pos, 
            self.fingertip_centered_quat, 
        )
        self.prev_local_eef_pos = local_eef_pos
        self.prev_local_eef_quat = local_eef_quat

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        
    def pre_steps(self, obs, keep_runs, num_pre_steps=100):
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
        return obs, keep_runs

    def _check_eef_close_to_target(self):
        """Check if end effector is close to the target."""
    
        dist = torch.norm(self.fingertip_centered_pos - self.target_pos, p=2, dim=-1)

        is_eef_close_to_target = torch.where(dist < self.cfg_task.rl.close_error_thresh,
                                             torch.ones_like(self.progress_buf),
                                             torch.zeros_like(self.progress_buf))

        return is_eef_close_to_target

    def _draw_debug_output(self, target_color):
        """Draw output used for debugging."""

        if self.render_viewer:
            self.gym.clear_lines(self.viewer)
            self._draw_target_bounds()
            self._draw_target(color=target_color)
            # if self.cfg_task.rl.use_keypoints_6D:
            #     self._draw_keypoints_6d()
            # else:
            #     self._draw_keypoints()

    def _draw_target_bounds(self):
        """Draw boundaries within which reach targets are sampled."""

        workspace_dims = [self.cfg_task.randomize.rl_workspace_bounds[0][1] - self.cfg_task.randomize.rl_workspace_bounds[0][0],
                          self.cfg_task.randomize.rl_workspace_bounds[1][1] - self.cfg_task.randomize.rl_workspace_bounds[1][0],
                          self.cfg_task.randomize.rl_workspace_bounds[2][1] - self.cfg_task.randomize.rl_workspace_bounds[2][0]]
        workspace_center = [self.robot_base_pos[0][0] + (self.cfg_task.randomize.rl_workspace_bounds[0][0] + self.cfg_task.randomize.rl_workspace_bounds[0][1]) / 2.0,
                            self.robot_base_pos[0][1] + (self.cfg_task.randomize.rl_workspace_bounds[1][0] + self.cfg_task.randomize.rl_workspace_bounds[1][1]) / 2.0,
                            self.robot_base_pos[0][2] + (self.cfg_task.randomize.rl_workspace_bounds[2][0] + self.cfg_task.randomize.rl_workspace_bounds[2][1]) / 2.0]
        self._draw_box_at_pose(dims=workspace_dims,
                               pos=workspace_center,
                               quat=[0.0, 0.0, 0.0, 1.0])

    def _draw_box_at_pose(self, dims, pos, quat):
        """Draw box with specified dimensions and pose."""

        geom = gymutil.WireframeBoxGeometry(xdim=dims[0], 
                                            ydim=dims[1], 
                                            zdim=dims[2], 
                                            pose=gymapi.Transform(),
                                            color=(0.0, 0.0, 0.0))

        for i in range(self.num_envs):
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(*pos)
            pose.r = gymapi.Quat(*quat)
            gymutil.draw_lines(geom, self.gym, self.viewer, self.env_ptrs[i], pose)

    def _draw_target(self, color):
        """Draw sampled target as sphere."""

        geom = gymutil.WireframeSphereGeometry(radius=0.01,
                                               num_lats=10, 
                                               num_lons=10, 
                                               pose=gymapi.Transform(), 
                                               color=color)
        for i in range(self.num_envs):
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(*self.target_pos[i].cpu().numpy().tolist())
            pose.r = gymapi.Quat(*self.target_quat[i].cpu().numpy().tolist())
            gymutil.draw_lines(geom, self.gym, self.viewer, self.env_ptrs[i], pose)

    def _draw_keypoints_6d(self):
        """Draw keypoints on gripper and target."""
        self.draw_keypoints_helper(self.keypoints_gripper)
        self.draw_keypoints_helper(self.keypoints_target)

    def draw_keypoints_helper(self, keypoints):
        num_keypoints = keypoints.shape[1]
        for env_id in range(self.num_envs):
            # plot keypoints
            colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            for kp in range(1, num_keypoints):
                line = [*keypoints[env_id][0].cpu().numpy(), *keypoints[env_id][kp].cpu().numpy()]
                self.gym.add_lines(self.viewer, self.env_ptrs[env_id], 1, line, colors[kp % 3])
            # for offset in [[-0.0005, -0.0005, 0], [-0.0005, 0.0005, 0], [0.0005, -0.0005, 0], [0.0005, 0.0005, 0]]:
            #     for kp in range(1, num_keypoints):
            #         line = [*keypoints[env_id][0].cpu().numpy() + np.array(offset),
            #                 *keypoints[env_id][kp].cpu().numpy() + np.array(offset)]
            #         self.gym.add_lines(self.viewer, self.env_ptrs[env_id], 1, line, colors[kp % 3])

    def _draw_keypoints(self):
        """Draw keypoints on gripper and target."""

        for i in range(self.num_envs):
            keypoints_gripper = self.keypoints_gripper[i].cpu().numpy()  # shape = (num_keypoints, 3)
            keypoints_target = self.keypoints_target[i].cpu().numpy()  # shape = (num_keypoints, 3)
            # Draw lines from first keypoint to third keypoint, and third keypoint to last keypoint
            gymutil.draw_line(gymapi.Vec3(*keypoints_gripper[0].tolist()),
                              gymapi.Vec3(*keypoints_gripper[2].tolist()),
                              gymapi.Vec3(1, 0, 0),
                              self.gym,
                              self.viewer,
                              self.env_ptrs[i])
            gymutil.draw_line(gymapi.Vec3(*keypoints_gripper[2].tolist()),
                              gymapi.Vec3(*keypoints_gripper[-1].tolist()),
                              gymapi.Vec3(0, 1, 0),
                              self.gym,
                              self.viewer,
                              self.env_ptrs[i])
            gymutil.draw_line(gymapi.Vec3(*keypoints_target[0].tolist()),
                              gymapi.Vec3(*keypoints_target[2].tolist()),
                              gymapi.Vec3(1, 0, 0),
                              self.gym,
                              self.viewer,
                              self.env_ptrs[i])
            gymutil.draw_line(gymapi.Vec3(*keypoints_target[2].tolist()),
                              gymapi.Vec3(*keypoints_target[-1].tolist()),
                              gymapi.Vec3(0, 1, 0),
                              self.gym,
                              self.viewer,
                              self.env_ptrs[i])

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_franka_reward(
    reset_buf,
    progress_buf,
    successes,
    consecutive_successes,
    max_episode_length,
    keypoints_gripper,
    keypoints_target,
    fingertip_centered_pos,
    target_pos,
):
    # type: (Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, float, float]

    distance_threshold = 0.005  # 5mm

    # Define the total reward
    total_reward = -1 * torch.sum(torch.norm(keypoints_gripper - keypoints_target, p=2, dim=-1), dim=-1)
    
    keypoint_dist = torch.norm(keypoints_gripper - keypoints_target, p=2, dim=-1).mean()
    
    total_reward = (torch.exp(total_reward) + torch.exp(10*total_reward) + torch.exp(100*total_reward) + torch.exp(1000*total_reward)) / 4.0

    dist = torch.norm(fingertip_centered_pos - target_pos, p=2, dim=-1)
    # Reset buffer and success
    reset_buf = torch.where(
        (progress_buf >= max_episode_length - 1), torch.ones_like(reset_buf), reset_buf
    )
    successes = torch.where(
        dist < distance_threshold,
        torch.ones_like(successes),
        torch.zeros_like(successes),
    )
    consecutive_successes = torch.where(
        reset_buf > 0, successes * reset_buf, consecutive_successes
    ).mean()

    return total_reward, reset_buf, successes, consecutive_successes, dist.mean(), keypoint_dist
