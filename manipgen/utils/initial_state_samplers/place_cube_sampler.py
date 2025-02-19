from isaacgym import gymtorch
from isaacgymenvs.utils.torch_jit_utils import (
    quat_conjugate, 
    quat_rotate, 
    quat_mul, 
    to_torch
)

import numpy as np
import torch
import math
import time

from manipgen.utils.initial_state_samplers.base_sampler import BaseSampler
from manipgen.utils.initial_state_samplers.pick_cube_sampler import FrankaPickCubePoseSampler
from manipgen.utils.rlgames_utils import load_model, get_actions


class FrankaPlaceCubePoseSampler(BaseSampler):
    """Sample initial poses for franka place tasks. we need external policy to generate picking poses.
    Procedures:
        1. pick object with external RL policy
        2. randomize franka pose with teleportation
        3. fix franka and object pose, sample receptacle pose
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
        self.policy_init_state_cur = 0

        # load pick_cube policy
        assert self.cfg.policy_checkpoint_path is not None and self.cfg.policy_config_path is not None
        self.pick_cube_policy = load_model(
            6,
            (22,),
            self.device,
            self.cfg.policy_checkpoint_path,
            config_path=self.cfg.policy_config_path,
        )

        self.num_rl_steps = self.cfg.num_rl_steps
        self.num_close_gripper_steps = self.cfg.num_close_gripper_steps
        self.num_lift_up_steps = self.cfg.num_lift_up_steps
        self.num_randomization_steps = self.cfg.num_randomization_steps

        self.num_randomization_per_policy = self.cfg.num_randomization_per_policy
        self.num_receptacle_pose_per_randomization = self.cfg.num_receptacle_pose_per_randomization

        self.object_height_lower = self.env.cfg_base.env.table_height + self.cfg.object_height_lower
        self.object_height_upper = self.env.cfg_base.env.table_height + self.cfg.object_height_upper

        self.object_pos_center = [
            self.env.table_pose.p.x,
            self.env.table_pose.p.y,
            self.env.cfg_base.env.table_height + 0.5 * self.env.box_size + self.env.receptacle_size,
        ]
        self.object_pos_noise = 0.2
        self.receptacle_pos_noise = self.cfg.receptacle_pos_noise
        if vision_based:
            self.receptacle_pos_noise *= 0.5
        self.receptacle_pos_upper_limit = [
            self.env.table_pose.p.x + 0.2,
            self.env.table_pose.p.y + 0.2,
        ]
        self.receptacle_pos_lower_limit = [
            self.env.table_pose.p.x - 0.2,
            self.env.table_pose.p.y - 0.2,
        ]

        self.object_points_local = self._sample_object_points(cube_size=self.env.box_size)
        self.receptacle_points_local = self._sample_object_points(cube_size=self.env.receptacle_size)

    _sample_object_points = FrankaPickCubePoseSampler._sample_object_points

    def _deploy_states(self, object_pos, object_quat, receptacle_pos, receptacle_quat, franka_dof_pos, close_gripper=False):
        """Deploy states and take a simulation step"""
        # set object pose
        self.env.object_pos[:, :] = object_pos
        self.env.object_quat[:, :] = object_quat
        self.env.object_linvel[:, :] = 0.0
        self.env.object_angvel[:, :] = 0.0
        self.init_object_xy = self.env.object_pos[:, :2].clone()

        # set initial franka dof pos
        self.env.dof_pos[:, :self.env.franka_num_dofs] = franka_dof_pos
        self.env.dof_vel[:, :self.env.franka_num_dofs] = 0.0
        self.env.dof_torque[:, :self.env.franka_num_dofs] = 0.0
        self.env.ctrl_target_dof_pos[:] = self.env.dof_pos[:].clone()
        if close_gripper:
            self.env.dof_torque[:, self.env.franka_num_dofs-2:self.env.franka_num_dofs] = -20.0
            self.env.ctrl_target_dof_pos[:, self.env.franka_num_dofs-2:self.env.franka_num_dofs] = -0.04

        # set initial receptacle pose: off the table at the beginning to avoid collision
        self.env.receptacle_pos[:, :] = receptacle_pos
        self.env.receptacle_quat[:, :] = receptacle_quat
        self.env.receptacle_linvel[:, :] = 0.0
        self.env.receptacle_angvel[:, :] = 0.0

        object_actor_ids_sim = self.env.object_actor_ids_sim[:].to(torch.int32)
        receptacle_actor_ids_sim = self.env.receptacle_actor_ids_sim[:].to(torch.int32)
        merged_actor_ids_sim = torch.cat([object_actor_ids_sim, receptacle_actor_ids_sim])
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.env.root_state),
            gymtorch.unwrap_tensor(merged_actor_ids_sim),
            len(merged_actor_ids_sim),
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

        init_receptacle_pos = torch.FloatTensor(
            [1.0, 0.0, 0.1],
        ).to(self.device).repeat(self.num_envs, 1)
        init_receptacle_quat = torch.FloatTensor(
            [0.0, 0.0, 0.0, 1.0],
        ).to(self.device).repeat(self.num_envs, 1)

        self._deploy_states(init_object_pos, init_object_quat, init_receptacle_pos, init_receptacle_quat, init_franka_dof_pos)

    def _recover_states(self, object_pos, object_quat, franka_dof_pos):
        """Recover states during randomization. 
        Sometimes the receptacle collides with the object / franka.
        We need to recover the states after sampling a new receptacle pose.
        """

        receptacle_pos = torch.FloatTensor(
            [1.0, 0.0, 0.1],
        ).to(self.device).repeat(self.num_envs, 1)
        receptacle_quat = torch.FloatTensor(
            [0.0, 0.0, 0.0, 1.0],
        ).to(self.device).repeat(self.num_envs, 1)

        self._deploy_states(object_pos, object_quat, receptacle_pos, receptacle_quat, franka_dof_pos, close_gripper=True)

    def _compute_pick_cube_obs(self):
        """ Observations for pick cube policy """
        local_eef_pos, local_eef_quat = self.env.pose_world_to_robot_base(
            self.env.fingertip_centered_pos,    # position of the gripper
            self.env.fingertip_centered_quat,   # orientation of the gripper
        )
        local_eef_quat = self.env._format_quaternion(local_eef_quat)

        obs_tensors = [
            local_eef_pos,
            local_eef_quat,
            self.env.fingertip_centered_linvel,
            self.env.fingertip_centered_angvel,
            self.env.object_pos,
            self.env.object_quat,
            self.init_object_xy,
        ]

        obs = torch.cat(obs_tensors, dim=-1)
        return {"obs": obs}

    def _sample_target_eef_pose(self):
        """ Sample target eef pose for randomization """
        cur_fingertip_centered_pos = self.env.fingertip_centered_pos.clone()
        cur_fingertip_centered_quat = self.env.fingertip_centered_quat.clone()

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
            target_pos[:, 2],
            self.object_height_lower, 
            self.object_height_upper
        )

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

    def _sample_receptacle_pose(self, object_pos, object_quat, franka_dof_pos):
        # receptacle position
        p = torch.randn((self.num_envs, 2), device=self.device)
        p = p / p.norm(dim=-1, keepdim=True)
        u = torch.rand(self.num_envs, device=self.device).unsqueeze(-1) ** (1.0 / 2)
        receptacle_pos = torch.zeros_like(object_pos)
        receptacle_pos[:, :2] = object_pos[:, :2] + self.receptacle_pos_noise * u * p
        receptacle_pos[:, 0].clamp_(
            self.receptacle_pos_lower_limit[0], self.receptacle_pos_upper_limit[0]
        )
        receptacle_pos[:, 1].clamp_(
            self.receptacle_pos_lower_limit[1], self.receptacle_pos_upper_limit[1]
        )
        receptacle_pos[:, 2] = (
            self.env.cfg_base.env.table_height + 0.5 * self.env.receptacle_size
        )
        # receptacle rotation
        receptacle_quat = torch.zeros_like(object_quat)
        theta_half = (
            (2 * torch.rand(self.num_envs, device=self.device) - 1) * math.pi / 8
        )  # rotation along z-axis
        receptacle_quat[:, 2] = torch.sin(theta_half)
        receptacle_quat[:, 3] = torch.cos(theta_half)

        # deploy states
        self._deploy_states(object_pos, object_quat, receptacle_pos, receptacle_quat, franka_dof_pos)

        # check valid
        # receptacle not in contact with object or franka (we have disabled contact with table in the environment)
        contact1 = self.env.check_contact(self.env.receptacle_rigid_body_ids_sim)
        vel = self.env.receptacle_linvel[:, :].norm(dim=-1)
        contact2 = vel > 1e-3
        contact = contact1 | contact2

        # object should be lifted
        lifted = self.env.object_pos[:, 2] > self.env.cfg_base.env.table_height + 0.05
        # franka gripper should be open
        open = self.env.gripper_dof_pos[:, -2:].sum(dim=-1) > 0.01
        # object and receptacle should be visible
        visible = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        if self.vision_based:
            camera_pose = self._get_camera_pose()
            object_pose = torch.cat([self.env.object_pos, self.env.object_quat], dim=-1)
            receptacle_pose = torch.cat([receptacle_pos, receptacle_quat], dim=-1)
            visible = self._check_object_in_camera_view(
                camera_pose, object_pose, self.object_points_local,
                fov=self.env.cfg_task.env.local_obs.horizontal_fov, clamp_dist=0.28
            )
            visible &= self._check_object_in_camera_view(
                camera_pose, receptacle_pose, self.receptacle_points_local,
                fov=self.env.cfg_task.env.local_obs.horizontal_fov, clamp_dist=0.28
            )

        found = (~contact) & lifted & open & visible

        return receptacle_pos, receptacle_quat, found

    def generate(self, num_samples):
        """Sample initial states for franka place tasks
        Args:
            num_samples: number of samples
        Returns:
            init_states: a dict containing object pose, receptacle pose, and franka dof pos
        """
        franka_dof_pos_list = []
        object_pose_list = []
        receptacle_pose_list = []

        num_iters = 0
        total_samples = 0
        total_trials = 0

        while total_samples < num_samples:
            num_iters += 1
            self._set_init_states()

            # take policy steps
            self.env.cfg_task.randomize.franka_gripper_initial_state = 1.0 # open gripper
            for step in range(self.num_rl_steps):
                self.env.progress_buf[:] = 0
                obs = self._compute_pick_cube_obs()
                actions = get_actions(obs, self.pick_cube_policy, is_deterministic=False)
                self.env.step(actions)

            # apply noise to eef pos: make place cube policy more robust to grasp errors
            self.env.ctrl_target_fingertip_midpoint_pos[:] = self.env.fingertip_centered_pos + 1e-2 * (torch.rand(self.num_envs, 3, device=self.device) * 2 - 1)
            self.env.ctrl_target_fingertip_midpoint_quat[:] = self.env.fingertip_centered_quat
            self.env.move_gripper_to_target_pose(gripper_dof_pos=-0.04, sim_steps=10)

            # teleportation: close gripper and lift up
            self.env.ctrl_target_fingertip_midpoint_pos[:] = self.env.fingertip_centered_pos
            self.env.ctrl_target_fingertip_midpoint_quat[:] = self.env.fingertip_centered_quat
            self.env.move_gripper_to_target_pose(gripper_dof_pos=-0.04, sim_steps=self.num_close_gripper_steps)
            self.env.ctrl_target_fingertip_midpoint_pos[:, -1] = self.env.cfg_base.env.table_height + 0.1
            self.env.move_gripper_to_target_pose(gripper_dof_pos=-0.04, sim_steps=self.num_lift_up_steps)

            # sample 
            self.env.cfg_task.randomize.franka_gripper_initial_state = 0.0 # close gripper
            for franka_object_pose_id in range(self.num_randomization_per_policy):
                # randomization: sample franka pose and move to the pose
                target_eef_pos, target_eef_pose = self._sample_target_eef_pose()
                self.env.ctrl_target_fingertip_midpoint_pos[:] = target_eef_pos
                self.env.ctrl_target_fingertip_midpoint_quat[:] = target_eef_pose
                self.env.move_gripper_to_target_pose(gripper_dof_pos=-0.04, stablize=False, sim_steps=self.num_randomization_steps)
                
                # sample receptacle pose
                object_pos_rec = self.env.object_pos.clone()
                object_quat_rec = self.env.object_quat.clone()
                franka_dof_pos_rec = self.env.dof_pos.clone()
                object_pose = torch.cat([object_pos_rec, object_quat_rec], dim=-1)
                for receptacle_pose_id in range(self.num_receptacle_pose_per_randomization):
                    receptacle_pos, receptacle_quat, found = self._sample_receptacle_pose(
                        object_pos_rec, object_quat_rec, franka_dof_pos_rec
                    )
                    receptacle_pose = torch.cat([receptacle_pos, receptacle_quat], dim=-1)

                    object_pose_list.append(object_pose[found])
                    receptacle_pose_list.append(receptacle_pose[found])
                    franka_dof_pos_list.append(franka_dof_pos_rec[found])

                    total_samples += found.sum().item()
                    total_trials += self.num_envs
                
                success_rate = total_samples / total_trials
                print(
                    f"#{num_iters} | total: {total_samples} | success rate: {100 * success_rate:.2f} "
                )

                self._recover_states(object_pos_rec, object_quat_rec, franka_dof_pos_rec)

        init_states = {
            "object_pose": torch.cat(object_pose_list, dim=0).cpu(),
            "receptacle_pose": torch.cat(receptacle_pose_list, dim=0).cpu(),
            "franka_dof_pos": torch.cat(franka_dof_pos_list, dim=0).cpu(),
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
            receptacle_pose = init_states["receptacle_pose"][start:end]
            receptacle_pos = receptacle_pose[:, :3]
            receptacle_quat = receptacle_pose[:, 3:]
            franka_dof_pos = init_states["franka_dof_pos"][start:end]
            self._deploy_states(object_pos, object_quat, receptacle_pos, receptacle_quat, franka_dof_pos, close_gripper=True)

            # keep static
            self.env.ctrl_target_fingertip_midpoint_pos[:] = self.env.fingertip_centered_pos
            self.env.ctrl_target_fingertip_midpoint_quat[:] = self.env.fingertip_centered_quat
            self.env.move_gripper_to_target_pose(gripper_dof_pos=-0.04, sim_steps=20, stablize=False)

            # discard if the cube drops
            filtered[start:end] = self.env.gripper_dof_pos[:, :].sum(dim=-1) < 0.01

            start += self.num_envs

            print(
                f"Filtered {filtered.sum()} out of {start} samples (total {num_samples} samples)."
            )

        for key in init_states:
            init_states[key] = init_states[key][~filtered].cpu()

        return init_states
    
    def filter_vision(self, init_states, filter_vision_threshold=0.001):
        """Filter initial states where the object or receptacle takes up less than `filter_vision_threshold` of the camera input
        Args:
            init_states: a dict containing object pose, receptacle pose, and franka dof pos
            filter_vision_threshold: threshold for filtering
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
            receptacle_pose = init_states["receptacle_pose"][start:end]
            receptacle_pos = receptacle_pose[:, :3]
            receptacle_quat = receptacle_pose[:, 3:]
            franka_dof_pos = init_states["franka_dof_pos"][start:end]
            self._deploy_states(object_pos, object_quat, receptacle_pos, receptacle_quat, franka_dof_pos, close_gripper=True)

            # render camera sensors
            if self.device != "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)

            self.gym.start_access_image_tensors(self.sim)
            wrist_seg_image_buf_tensor = torch.stack(self.env.camera_tensors, dim=0).reshape(self.num_envs, -1)
            num_pixels = wrist_seg_image_buf_tensor.shape[-1]
            num_object_pixels = (wrist_seg_image_buf_tensor == self.env.object_actor_id_env + 1).sum(dim=-1)
            num_receptacle_pixels = (wrist_seg_image_buf_tensor == self.env.receptacle_actor_id_env + 1).sum(dim=-1)
            self.gym.end_access_image_tensors(self.sim)
            # filter if the object or receptacle is not visiable: ratio of pixels < filter_vision_threshold
            filtered[start:end] = num_object_pixels < max(5, int(filter_vision_threshold * num_pixels))
            filtered[start:end] |= num_receptacle_pixels < max(5, int(filter_vision_threshold * num_pixels))

            start += self.num_envs

            print(
                f"Filtered {filtered.sum()} out of {start} samples (total {num_samples} samples)."
            )

        for key in init_states:
            init_states[key] = init_states[key][~filtered].cpu()

        return init_states

