from isaacgym import gymtorch
from isaacgymenvs.utils.torch_jit_utils import (
    tensor_clamp, 
    quat_from_angle_axis, 
    quat_mul, 
    quat_rotate,
    quat_from_euler_xyz,
    quat_conjugate,
    to_torch, 
)

import numpy as np
import torch
import math
import time

from manipgen.utils.initial_state_samplers.base_sampler import BaseSampler
from manipgen.utils.initial_state_samplers.grasp_handle_sampler import FrankaGraspHandlePoseSampler
from manipgen.utils.rlgames_utils import load_model, get_actions

class FrankaOpenPoseSampler(BaseSampler):
    """Sample initial poses for franka open/close tasks. we need external policy to generate grasping poses.
    Procedures:
        1. grasp handle with external RL policy
        2. randomize franka pose with teleportation
    """

    def __init__(
        self,
        env,
        vision_based=False,
        cfg=None,
    ):
        """
        Args:
            env: isaacgym environment
            vision_based: whether the policy is vision-based
            cfg: config for the sampler
        """
        super().__init__(env, vision_based, cfg)

        assert self.cfg.policy_init_states_path is not None
        policy_init_states = torch.load(self.cfg.policy_init_states_path)
        self.policy_init_object_pose = policy_init_states["object_pose"].to(self.device)
        self.policy_init_franka_dof_pos = policy_init_states["franka_dof_pos"].to(self.device)
        self.policy_init_object_dof_pos = policy_init_states["init_object_dof_pos"].to(self.device)
        num_policy_init_states = self.policy_init_object_pose.shape[0]
        idx = torch.randperm(num_policy_init_states)
        self.policy_init_object_pose = self.policy_init_object_pose[idx]
        self.policy_init_franka_dof_pos = self.policy_init_franka_dof_pos[idx]
        self.policy_init_object_dof_pos = self.policy_init_object_dof_pos[idx]
        self.policy_init_state_cur = 0

        # load model
        assert self.cfg.policy_checkpoint_path is not None and self.cfg.policy_config_path is not None
        self.grasp_policy = load_model(
            6,              # action dim
            (73,),          # obs shape
            self.device,
            self.cfg.policy_checkpoint_path,
            config_path=self.cfg.policy_config_path,
        )

        # steps for each stage
        self.num_rl_steps = self.env.cfg_task.sampler.num_rl_steps
        self.num_close_gripper_steps = self.env.cfg_task.sampler.num_close_gripper_steps
        self.num_randomization_steps = self.env.cfg_task.sampler.num_randomization_steps
        self.num_randomization_per_policy = self.env.cfg_task.sampler.num_randomization_per_policy

        self.object_points_local = self._sample_object_points()

    _get_handle_pose = FrankaGraspHandlePoseSampler._get_handle_pose
    filter_vision = FrankaGraspHandlePoseSampler.filter_vision
    _sample_object_points = FrankaGraspHandlePoseSampler._sample_object_points

    def _deploy_states(self, object_pos, object_quat, object_dof_pos, franka_dof_pos, close_gripper=False):
        """Deploy states and take a simulation step"""
        # set object states
        self.env.object_dof_pos[:, :] = object_dof_pos
        self.env.object_pos[:, :] = object_pos
        self.env.object_quat[:, :] = object_quat
        self.env.object_linvel[:, :] = 0.0
        self.env.object_angvel[:, :] = 0.0

        # set franka states
        self.env.dof_pos[:, :self.env.franka_num_dofs] = franka_dof_pos
        self.env.dof_vel[:, :] = 0.0
        self.env.dof_torque[:, :] = 0.0
        self.env.ctrl_target_dof_pos[:] = self.env.dof_pos[:].clone()
        if close_gripper:
            self.env.dof_torque[:, self.env.franka_num_dofs-2:self.env.franka_num_dofs] = -20.0
            self.env.ctrl_target_dof_pos[:, self.env.franka_num_dofs-2:self.env.franka_num_dofs] = -0.04

        # Set DOF state
        franka_actor_ids_sim = self.env.franka_actor_ids_sim.clone().to(dtype=torch.int32)
        object_actor_ids_sim = self.env.object_actor_ids_sim.to(torch.int32)
        merged_actor_ids_sim = torch.cat((franka_actor_ids_sim, object_actor_ids_sim), dim=0)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.env.dof_state),
            gymtorch.unwrap_tensor(merged_actor_ids_sim),
            len(merged_actor_ids_sim),
        )

        # Set DOF torque
        self.gym.set_dof_actuation_force_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(torch.zeros_like(self.env.dof_torque)),
            gymtorch.unwrap_tensor(merged_actor_ids_sim),
            len(merged_actor_ids_sim),
        )

        # Set root state
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.env.root_state),
            gymtorch.unwrap_tensor(object_actor_ids_sim),
            len(object_actor_ids_sim),
        )

        # Simulate one step
        self.env.simulate_and_refresh()

    def _set_init_states(self):
        """ Set initial states for policy execution """
        assert (
            self.policy_init_state_cur + self.num_envs
            <= self.policy_init_object_pose.shape[0]
        )
        start = self.policy_init_state_cur
        end = self.policy_init_state_cur = self.policy_init_state_cur + self.num_envs
        
        init_object_pose = self.policy_init_object_pose[start:end]
        init_object_pos = init_object_pose[:, :3]
        init_object_quat = init_object_pose[:, 3:]
        init_object_dof_pos = self.policy_init_object_dof_pos[start:end]

        init_franka_dof_pos = self.policy_init_franka_dof_pos[start:end]

        self._deploy_states(init_object_pos, init_object_quat, init_object_dof_pos, init_franka_dof_pos)

        # record intial dof for observation
        self.env.init_object_dof_pos[:] = init_object_dof_pos.clone()

    def _compute_grasp_handle_obs(self):
        """Compute observation for grasp handle policy"""
        local_eef_pos, local_eef_quat = self.env.pose_world_to_robot_base(
            self.env.fingertip_centered_pos,    # position of the gripper
            self.env.fingertip_centered_quat,   # orientation of the gripper
        )
        local_eef_quat = self.env._format_quaternion(local_eef_quat)

        grasp = torch.zeros(self.num_envs, device=self.device)
        obs_tensors = [
            local_eef_pos,
            local_eef_quat,
            self.env.fingertip_centered_linvel,                 # linear velocity of the gripper
            self.env.fingertip_centered_angvel,                 # angular velocity of the gripper
            self.env.handle_pos,                                # position of the handle
            self.env.handle_quat,                               # orientation of the handle
            self.env.object_dof_pos,                            # dof pos of the object
            self.env.object_keypoints,                          # position of keypoints of the object
            self.env.object_dof_diff.unsqueeze(-1),             # change in object dof pos
            self.env.dist_object_gripper.unsqueeze(-1),         # distance between object and gripper
            self.env.eef_height_diff.unsqueeze(-1),             # change in eef height
            grasp.unsqueeze(-1),                                # whether to close the gripper with hard-code control
        ]

        obs = torch.cat(obs_tensors, dim=-1)
        return {"obs": obs}

    def _sample_target_eef_pose(self):
        """ Sample target eef pose for randomization """
        # sample target dof pos
        dof_upper_limits = self.env.object_dof_upper_limits.repeat(self.num_envs, 1)
        dof_lower_limits = self.env.object_dof_lower_limits.repeat(self.num_envs, 1)

        dof_shift = (0.15 * torch.rand((self.num_envs, self.env.object_num_dofs), device=self.device) + 0.02) \
            * (dof_upper_limits - dof_lower_limits)
        dof_shift = torch.where(
            torch.rand((self.num_envs, self.env.object_num_dofs), device=self.device) < 0.5, 
            dof_shift, -dof_shift
        )

        dof_target = self.env.object_dof_pos + dof_shift
        dof_target = torch.clamp(dof_target, dof_lower_limits, dof_upper_limits)

        # compute target handle pose based on target dof pos
        object_pose = torch.cat([self.env.object_pos, self.env.object_quat], dim=-1)
        target_handle_pose = self._get_handle_pose(object_pose, dof_target)
        target_handle_pos = target_handle_pose[:, :3]
        target_handle_quat = target_handle_pose[:, 3:]

        # compute current eef pose in handle frame
        current_grasp_pos_local = quat_rotate(
            quat_conjugate(self.env.handle_quat), 
            self.env.fingertip_centered_pos - self.env.handle_pos
        )
        current_grasp_rot_local = quat_mul(
            quat_conjugate(self.env.handle_quat), 
            self.env.fingertip_centered_quat
        )

        # compute target eef pose in world frame
        target_pos = target_handle_pos + quat_rotate(target_handle_quat, current_grasp_pos_local)
        target_quat = quat_mul(target_handle_quat, current_grasp_rot_local)

        return target_pos, target_quat

    def _check_state(self):
        """ Check if current state is valid. """
        # gripper not fully closed
        valid = self.env.gripper_dof_pos[:, -2:].sum(dim=-1) > 0.005

        # check if the handle is in the camera view
        if self.vision_based:
            object_pos = self.env.object_pos.clone()
            object_quat = self.env.object_quat.clone()
            object_pose = torch.cat([object_pos, object_quat], dim=-1)
            object_dof_pos = self.env.object_dof_pos.clone()
            handle_pose = self._get_handle_pose(object_pose, object_dof_pos)
            camera_pose = self._get_camera_pose()
            visible = self._check_object_in_camera_view(
                camera_pose, handle_pose, self.object_points_local,
                fov=self.env.cfg_task.env.local_obs.horizontal_fov, clamp_dist=0.28,
                fov_threshold=2.5,
            )
            valid = valid & visible

        return valid

    def generate(self, num_samples):
        """Sample initial states for franka pick tasks
        Args:
            num_samples: number of samples
        Returns:
            init_states: a dict containing object pose and franka dof pos
        """
        franka_dof_pos_list = []
        object_pose_list = []
        object_dof_pos_list = []

        num_iters = 0
        total_samples = 0
        total_trials = 0

        while total_samples < num_samples:
            num_iters += 1
            self._set_init_states()

            # take policy steps
            self.env.cfg_task.randomize.franka_gripper_initial_state = 1.0 # open gripper
            for step in range(self.num_rl_steps):
                obs = self._compute_grasp_handle_obs()
                actions = get_actions(obs, self.grasp_policy, is_deterministic=False)
                self.env.step(actions)

            # teleportation: close gripper
            self.env.ctrl_target_fingertip_midpoint_pos[:] = self.env.fingertip_centered_pos
            self.env.ctrl_target_fingertip_midpoint_quat[:] = self.env.fingertip_centered_quat
            self.env.move_gripper_to_target_pose(gripper_dof_pos=-0.04, sim_steps=self.num_close_gripper_steps)

            # sample 
            self.env.cfg_task.randomize.franka_gripper_initial_state = 0.0 # close gripper
            for franka_object_pose_id in range(self.num_randomization_per_policy):
                if franka_object_pose_id > 0:
                    # randomization: sample franka pose and move to the pose
                    target_eef_pos, target_eef_pose = self._sample_target_eef_pose()
                    self.env.ctrl_target_fingertip_midpoint_pos[:] = target_eef_pos
                    self.env.ctrl_target_fingertip_midpoint_quat[:] = target_eef_pose
                    self.env.move_gripper_to_target_pose(gripper_dof_pos=-0.04, stablize=False, sim_steps=self.num_randomization_steps)
            
                # check if the object is in the gripper
                found = self._check_state()
                if found.any():
                    franka_dof_pos = self.env.dof_pos[found, :self.env.franka_num_dofs].clone()
                    object_pos = self.env.object_pos[found].clone()
                    object_quat = self.env.object_quat[found].clone()
                    oject_pose = torch.cat([object_pos, object_quat], dim=-1)
                    object_dof_pos = self.env.object_dof_pos[found].clone()

                    franka_dof_pos_list.append(franka_dof_pos)
                    object_pose_list.append(oject_pose)
                    object_dof_pos_list.append(object_dof_pos)

                total_samples += found.sum().item()
                total_trials += self.num_envs
                success_rate = total_samples / total_trials
                print(
                    f"#{num_iters} | total: {total_samples} | found: {found.sum().item()} | success rate: {100 * success_rate:.2f} "
                )
                
        init_states = {
            "object_pose": torch.cat(object_pose_list, dim=0).cpu(),
            "franka_dof_pos": torch.cat(franka_dof_pos_list, dim=0).cpu(),
            "init_object_dof_pos": torch.cat(object_dof_pos_list, dim=0).cpu(),
        }
        return init_states
