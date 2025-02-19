from isaacgym import gymtorch
from isaacgymenvs.utils.torch_jit_utils import (
    quat_rotate,
    quat_conjugate, 
    quat_mul, 
    to_torch
)

import numpy as np
import torch
import math
import time

from manipgen.utils.initial_state_samplers.base_sampler import BaseSampler
from manipgen.utils.initial_state_samplers.pick_cube_sampler import FrankaPickCubePoseSampler
from manipgen.utils.initial_state_samplers.pick_sampler import FrankaPickPoseSampler
from manipgen.utils.rlgames_utils import load_model, get_actions


class FrankaPlacePoseSampler(BaseSampler):
    """Sample initial poses for franka place tasks. we need external policy to generate picking poses.
    Procedures:
        1. pick object with external RL policy
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

        # load init states
        assert self.cfg.policy_init_states_path is not None
        policy_init_states = torch.load(self.cfg.policy_init_states_path)
        self.policy_init_object_pose = policy_init_states["object_pose"].to(self.device)
        self.policy_init_franka_dof_pos = policy_init_states["franka_dof_pos"].to(
            self.device
        )
        num_policy_init_states = self.policy_init_object_pose.shape[0]
        idx = torch.randperm(num_policy_init_states)
        self.policy_init_object_pose = self.policy_init_object_pose[idx]
        self.policy_init_franka_dof_pos = self.policy_init_franka_dof_pos[idx]
        self.policy_init_state_cur = 0

        # load model
        assert self.cfg.policy_checkpoint_path is not None and self.cfg.policy_config_path is not None
        self.pick_policy = load_model(
            6,                  # action dim
            (105,),             # obs shape
            self.device,
            self.cfg.policy_checkpoint_path,
            config_path=self.cfg.policy_config_path,
        )

        # steps for each stage
        self.num_rl_steps = self.cfg.num_rl_steps
        self.num_close_gripper_steps = self.cfg.num_close_gripper_steps
        self.num_lift_up_steps = self.cfg.num_lift_up_steps
        self.num_randomization_steps = self.cfg.num_randomization_steps
        self.num_filter_steps = self.cfg.num_filter_steps

        self.num_randomization_per_policy = self.cfg.num_randomization_per_policy

        # object position
        self.object_pos_center = [
            self.env.table_pose.p.x,
            self.env.table_pose.p.y,
            self.env.cfg_base.env.table_height
        ]
        self.object_pos_noise = self.cfg.object_pos_noise
        self.object_height_lower = self.env.cfg_base.env.table_height + self.cfg.object_height_lower
        self.object_height_upper = self.env.cfg_base.env.table_height + self.cfg.object_height_upper

        self.object_points_local = self._sample_object_points()

    def _deploy_states(self, object_pos, object_quat, franka_dof_pos, close_gripper=False):
        """Deploy states and take a simulation step"""
        # set object pose
        self.env.object_pos[:, :] = object_pos
        self.env.object_quat[:, :] = object_quat
        self.env.object_linvel[:, :] = 0.0
        self.env.object_angvel[:, :] = 0.0

        # set initial franka dof pos
        self.env.dof_pos[:, :self.env.franka_num_dofs] = franka_dof_pos
        self.env.dof_vel[:, :self.env.franka_num_dofs] = 0.0
        self.env.dof_torque[:, :self.env.franka_num_dofs] = 0.0
        self.env.ctrl_target_dof_pos[:] = self.env.dof_pos[:].clone()
        if close_gripper:
            self.env.dof_torque[:, self.env.franka_num_dofs-2:self.env.franka_num_dofs] = -20.0
            self.env.ctrl_target_dof_pos[:, self.env.franka_num_dofs-2:self.env.franka_num_dofs] = -0.04

        object_actor_ids_sim = self.env.object_actor_ids_sim[:].to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.env.root_state),
            gymtorch.unwrap_tensor(object_actor_ids_sim),
            len(object_actor_ids_sim),
        )

        franka_actor_ids_sim = self.env.franka_actor_ids_sim[:].to(torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.env.dof_state),
            gymtorch.unwrap_tensor(franka_actor_ids_sim),
            len(franka_actor_ids_sim),
        )
        self.gym.set_dof_actuation_force_tensor_indexed(
            self.sim, 
            gymtorch.unwrap_tensor(self.env.dof_torque),
            gymtorch.unwrap_tensor(franka_actor_ids_sim),
            len(franka_actor_ids_sim),
        )

        # Simulate one step
        self.env.disable_gravity()
        self.env.simulate_and_refresh()
        self.env.enable_gravity()

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

        init_franka_dof_pos = self.policy_init_franka_dof_pos[start:end]

        self.env._clear_clutter_and_obstacles(torch.arange(self.num_envs, device=self.device))
        self._deploy_states(init_object_pos, init_object_quat, init_franka_dof_pos)
        self.env._reset_clutter_and_obstacles(torch.arange(self.num_envs, device=self.device))
        if self.env.cfg_task.env.enable_clutter_and_obstacle:
            self._deploy_states(init_object_pos, init_object_quat, init_franka_dof_pos)

        self.init_object_xy = init_object_pos[:, :2].clone()
        self.env.gripper_direction_prev = quat_rotate(self.env.fingertip_centered_quat, self.env.up_v)
        return init_object_pose

    def _compute_pick_obs(self):
        """ Observations for pick policy """
        local_eef_pos, local_eef_quat = self.env.pose_world_to_robot_base(
            self.env.fingertip_centered_pos,    # position of the gripper
            self.env.fingertip_centered_quat,   # orientation of the gripper
        )
        local_eef_quat = self.env._format_quaternion(local_eef_quat)
        gripper_dof_offset = 0.04 - self.env.gripper_dof_pos

        lift_up = torch.zeros(self.num_envs, device=self.device)
        obs_tensors = [
            local_eef_pos,
            local_eef_quat,
            self.env.fingertip_centered_linvel,                 # linear velocity of the gripper
            self.env.fingertip_centered_angvel,                 # angular velocity of the gripper
            self.env.object_pos,                                # position of the object
            self.env.object_quat,                               # orientation of the object
            self.env.object_keypoints,                          # position of keypoints of the object
            self.init_object_xy,                                # initial xy position of the object (prevent the object from moving)
            self.env.dist_gripper_direction.unsqueeze(-1),      # distance between gripper direction
            self.env.dist_grasp_keypoints.unsqueeze(-1),        # distance between gripper
            gripper_dof_offset,                                 # gripper dof offset
            self.env.lf_contact_force,                          # contact force on the left finger
            self.env.rf_contact_force,                          # contact force on the right finger
            self.env.clutter_pos_xy,                            # xy position of clutter objects
            self.env.obstacle_pos.reshape(self.num_envs, -1),   # position of obstacles
            self.env.obstacle_quat.reshape(self.num_envs, -1),  # orientation of obstacles
            lift_up.unsqueeze(-1),                              # whether to lift up the object with hard-code control
        ]

        obs = torch.cat(obs_tensors, dim=-1)
        return {"obs": obs}

    def _sample_target_eef_pose(self):
        """ Sample target eef pose for randomization """
        cur_fingertip_centered_pos = self.env.fingertip_centered_pos.clone()
        cur_fingertip_centered_quat = self.env.fingertip_centered_quat.clone()

        gripper_object_offset_z = cur_fingertip_centered_pos[:, 2] - self.env.object_height - self.env.cfg_base.env.table_height

        # position
        target_pos = cur_fingertip_centered_pos + torch.rand(
            self.num_envs, 3, device=self.device
        ) * 0.2 - 0.1
        target_pos[:, 0] = torch.clamp(
            target_pos[:, 0],
            self.object_pos_center[0] - self.object_pos_noise,
            self.object_pos_center[0] + self.object_pos_noise,
        )
        target_pos[:, 1] = torch.clamp(
            target_pos[:, 1],
            self.object_pos_center[1] - self.object_pos_noise,
            self.object_pos_center[1] + self.object_pos_noise,
        )
        target_pos[:, 2] = torch.clamp(
            target_pos[:, 2] - gripper_object_offset_z,
            self.object_height_lower, 
            self.object_height_upper
        ) + gripper_object_offset_z

        # rotation
        target_rot = cur_fingertip_centered_quat
        invalid = torch.ones(self.num_envs, device=self.device).bool()
        up_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)
        num_attempts = 0
        while invalid.any():
            # rotation noise: along any axis by an angle < 30 degree
            rot_noise = torch.zeros((self.num_envs, 4), device=self.device)
            p = torch.randn((self.num_envs, 3), device=self.device)
            p = p / p.norm(dim=-1, keepdim=True)
            theta_half = (2 * torch.rand(self.num_envs, device=self.device) - 1) * math.pi / 6
            rot_noise[:, :3] = p * torch.sin(theta_half).unsqueeze(-1)
            rot_noise[:, 3] = torch.cos(theta_half)

            target_rot[invalid] = quat_mul(rot_noise[invalid], target_rot[invalid])

            # rotation is valid if the finger points downwards (allow 60 degree noise)
            eef_direction = quat_rotate(target_rot, up_axis)
            invalid = eef_direction[:, 2] > -0.5

            num_attempts += 1
            assert num_attempts < 1000, "Failed to sample target eef pose"
        
        return target_pos, target_rot

    def _check_state(self):
        """ Check if current state is valid. """
        # gripper not fully closed
        caging = self.env.gripper_dof_pos[:, -2:].sum(dim=-1) > 0.002

        # object is at least 2cm above the table
        lifted = self.env.object_height > 0.02

        # object is visible
        visible = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        if self.vision_based:
            camera_pose = self._get_camera_pose()
            object_pose = torch.cat([self.env.object_pos, self.env.object_quat], dim=-1)
            visible = self._check_object_in_camera_view(
                camera_pose, object_pose, self.object_points_local,
                fov=self.env.cfg_task.env.local_obs.horizontal_fov, clamp_dist=0.28
            )

        valid = caging & lifted & visible
        return valid

    def generate(self, num_samples):
        """Sample initial states for franka place tasks
        Args:
            num_samples: number of samples
        Returns:
            init_states: a dict containing object pose and franka dof pos
        """
        franka_dof_pos_list = []
        object_pose_list = []
        rest_object_pose_list = []

        num_iters = 0
        total_samples = 0
        total_trials = 0

        while total_samples < num_samples:
            num_iters += 1
            init_object_pose = self._set_init_states()

            # take policy steps
            self.env.cfg_task.randomize.franka_gripper_initial_state = 1.0 # open gripper
            for step in range(self.num_rl_steps):
                obs = self._compute_pick_obs()
                actions = get_actions(obs, self.pick_policy, is_deterministic=True)
                self.env.step(actions)

            # teleportation: close gripper and lift up
            self.env.ctrl_target_fingertip_midpoint_pos[:] = self.env.fingertip_centered_pos
            self.env.ctrl_target_fingertip_midpoint_quat[:] = self.env.fingertip_centered_quat
            self.env.move_gripper_to_target_pose(gripper_dof_pos=-0.04, sim_steps=self.num_close_gripper_steps)
            self.env.ctrl_target_fingertip_midpoint_pos[:, -1] += 0.15
            self.env.move_gripper_to_target_pose(gripper_dof_pos=-0.04, sim_steps=self.num_lift_up_steps)

            # sample 
            self.env.cfg_task.randomize.franka_gripper_initial_state = 0.0 # close gripper
            for franka_object_pose_id in range(self.num_randomization_per_policy):
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
                    rest_object_pose = init_object_pose[found].clone()

                    franka_dof_pos_list.append(franka_dof_pos)
                    object_pose_list.append(oject_pose)
                    rest_object_pose_list.append(rest_object_pose)

                total_samples += found.sum().item()
                total_trials += self.num_envs
                success_rate = total_samples / total_trials
                print(
                    f"#{num_iters} | total: {total_samples} | found: {found.sum().item()} | success rate: {100 * success_rate:.2f} "
                )
                
        init_states = {
            "object_pose": torch.cat(object_pose_list, dim=0).cpu(),
            "franka_dof_pos": torch.cat(franka_dof_pos_list, dim=0).cpu(),
            "rest_object_pose": torch.cat(rest_object_pose_list, dim=0).cpu(),
        }
        return init_states

    def filter(self, init_states):
        """Filter initial states that seem to be unstable. We initialize the environment with the initial states 
        and keep Franka static for a few steps. If the object drops, we filter the initial states.
        Args:
            init_states: a dict containing initial states
        Returns:
            init_states: filtered initial states
        """
        for key in init_states:
            init_states[key] = init_states[key].to(self.device)

        num_samples = len(init_states["franka_dof_pos"])
        original_num_samples = num_samples
        while num_samples < self.num_envs:
            for key in init_states:
                init_states[key] = torch.cat([init_states[key], init_states[key]], dim=0)
            num_samples *= 2
        if original_num_samples != num_samples:
            print(f"Expand the number of samples from {original_num_samples} to {num_samples}.")

        start = 0

        filtered = torch.zeros(num_samples, dtype=torch.bool, device=self.device)

        while start < num_samples:
            end = start + self.num_envs
            if end > num_samples:
                end = num_samples
                start = end - self.num_envs

            # set initial states
            object_pose = init_states["object_pose"][start:end]
            object_pos = object_pose[:, :3]
            object_quat = object_pose[:, 3:]
            franka_dof_pos = init_states["franka_dof_pos"][start:end]
            self._deploy_states(object_pos, object_quat, franka_dof_pos, close_gripper=True)

            # keep static
            self.env.ctrl_target_fingertip_midpoint_pos[:] = self.env.fingertip_centered_pos
            self.env.ctrl_target_fingertip_midpoint_quat[:] = self.env.fingertip_centered_quat
            self.env.move_gripper_to_target_pose(gripper_dof_pos=-0.04, sim_steps=self.num_filter_steps, stablize=False)

            # filter
            valid = self._check_state()
            filtered[start:end] = ~valid

            start += self.num_envs

            print(
                f"Filtered {filtered.sum()} out of {start} samples (ratio {filtered.sum() / start * 100:.2f}% | total {num_samples} samples)."
            )

        if original_num_samples != num_samples:
            for key in init_states:
                init_states[key] = init_states[key][:original_num_samples]
            filtered = filtered[:original_num_samples]

        for key in init_states:
            init_states[key] = init_states[key][~filtered].cpu()

        return init_states
    
    filter_vision = FrankaPickCubePoseSampler.filter_vision
    _sample_object_points = FrankaPickPoseSampler._sample_object_points
