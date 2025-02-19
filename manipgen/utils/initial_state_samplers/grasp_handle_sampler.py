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
from manipgen.utils.initial_state_samplers.pick_cube_sampler import (
    FrankaPickCubePoseSampler,
)
from manipgen.utils.initial_state_samplers.pick_sampler import (
    FrankaPickPoseSampler,
)

class FrankaGraspHandlePoseSampler(BaseSampler):
    """Sample initial poses for franka grasp handle tasks using inverse kinematics
    Procedures:
        1. sample target end-effector pose and object pose
        2. use inverse kinematics to find franka dof pos
        3. backtrack in joint space if collision exists
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
            damping: inverse kinematics coefficient
            ik_solver_pos_tolerance: accept IK result if position error is below this threshold
            ik_solver_rot_tolerance: accept IK result if rotation error is below this threshold
            max_ik_iters: max number of IK iterations
            max_backtrack_iters: max number of backtrack iterations
        """
        super().__init__(env, vision_based, cfg)
        self.damping = self.cfg.damping  # inverse kinematics coefficient
        self.max_ik_iters = self.cfg.max_ik_iters
        self.max_backtrack_iters = self.cfg.max_backtrack_iters
        self.ik_solver_pos_tolerance = self.cfg.ik_solver_pos_tolerance
        self.ik_solver_rot_tolerance = self.cfg.ik_solver_rot_tolerance

        self.target_eef_radius = self.cfg.target_eef_radius
        self.target_eef_rotation_mode = 2   # 0: fully random, 1: more limited, 2: point towards the object

        self.object_points_local = self._sample_object_points()

        # for backtracking
        self.franka_default_arm_dof_pos = torch.tensor(
            self.env.cfg_task.randomize.franka_arm_initial_dof_pos,
            device=self.device,
        )

    _sample_object_points = FrankaPickPoseSampler._sample_object_points
    _fetch_object_points = FrankaPickCubePoseSampler._fetch_object_points

    def _get_handle_pose(self, object_pose, object_dof_pos):
        """Compute handle pose in world frame from object pose and object dof pos.
        Args:
            object_pose: object pose in world frame, (num_envs, 7)
            object_dof_pos: object dof pos, (num_envs, 1)
        Returns:
            handle_pose: handle pose in world frame, (num_envs, 7)
        """
        num_create = object_pose.shape[0]
        handle_pose = torch.zeros((num_create, 7), device=self.device)
        object_dof_pos = object_dof_pos[:num_create, :]
        
        if self.env.object_meta["type"] == "door":
            # revolute joint, we only rotate along z-axis
            handle_pose[:, :3] = object_pose[:, :3]
            rot = quat_from_angle_axis(
                object_dof_pos[:, 0], 
                to_torch([0., 0., self.env.object_meta["joint_val"]], device=self.device)   # joint_val is -1 if the joint is on the left side
            )
            handle_pose[:, 3:] = quat_mul(rot, object_pose[:, 3:])
        elif self.env.object_meta["type"] == "drawer":
            # prismatic joint, we only move along x-axis
            offset = object_dof_pos * to_torch([1., 0., 0.], device=self.device)
            handle_pose[:, :3] = object_pose[:, :3] + quat_rotate(object_pose[:, 3:], offset)
            handle_pose[:, 3:] = object_pose[:, 3:]
        else:
            raise ValueError(f"Unsupported object type: {self.env.object_meta['type']}")
        
        return handle_pose

    def _check_state(self, stage, steps, object_pose, init_object_dof_pos, target_eef_pose):
        # ik solved
        eef_pos = self.env.fingertip_centered_pos
        target_pos = target_eef_pose[:, :3]
        pos_error = torch.norm(eef_pos - target_pos, dim=-1)
        ik_solved = (pos_error < self.ik_solver_pos_tolerance) & (stage == 0)
        if self.vision_based:
            eef_rot = self.env.fingertip_centered_quat
            up_axis = to_torch([0., 0., 1.], device=self.device).repeat(self.env.num_envs, 1)
            cur_v = quat_rotate(eef_rot, up_axis)
            tgt_rot = target_eef_pose[:, 3:]
            tgt_v = quat_rotate(tgt_rot, up_axis)
            rot_error = (cur_v * tgt_v).sum(dim=-1)
            ik_solved = ik_solved & (rot_error > 1 - self.ik_solver_rot_tolerance)

        # ik timeout
        ik_timeout = (steps == self.max_ik_iters) & (stage == 0) & (~ik_solved)

        # check contact
        contact = self.env.check_contact(self.env.franka_rigid_body_ids_sim)
        contact |= (torch.norm(self.env.handle_linvel, dim=-1) > 1e-3) | (torch.norm(self.env.handle_angvel, dim=-1) > 1e-3)

        # the gripper should not be behind the board
        # transform eef_pos to handle local frame
        handle_pose = self._get_handle_pose(object_pose, init_object_dof_pos.unsqueeze(-1))
        eef_pos_local = quat_rotate(quat_conjugate(handle_pose[:, 3:]), eef_pos - handle_pose[:, :3])
        valid_pos = eef_pos_local[:, 0] > 0

        # check if the handle is in the camera view
        visible = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        if self.vision_based:
            camera_pose = self._get_camera_pose()
            visible = self._check_object_in_camera_view(
                camera_pose, handle_pose, self.object_points_local,
                fov=self.env.cfg_task.env.local_obs.horizontal_fov,
            )

        # backtrack found
        found = (~contact) & (stage == 1) & valid_pos & visible
        # backtrack timeout
        backtrack_timeout = (
            (steps == self.max_backtrack_iters) & (stage == 1) & (~found)
        )
        # assert backtrack_timeout.sum() == 0     # this should not happen

        fail = ik_timeout | backtrack_timeout

        return ik_solved, found, fail

    def _sample_object_pose(self, num_create):
        """Sample object poses (root link)
        """
        object_pos, object_rot = self.env._sample_object_pose(num_create)
        object_pose = torch.cat([object_pos, object_rot], dim=-1)
        init_object_dof_pos = self.env.rest_object_dof_pos + self.env.cfg_task.randomize.max_object_init_dof_ratio * (
            self.env.target_object_dof_pos - self.env.rest_object_dof_pos
        ) * torch.rand((num_create, ), device=self.device)
        return object_pose, init_object_dof_pos

    def _sample_target_eef_pose(self, num_create, object_pose, init_object_dof_pos):
        """Sample target eef pose. We first sample in handle local frame, then transform to world frame.
        """
        rest_object_pose = torch.zeros((num_create, 7), device=self.device)
        rest_object_pose[:, -1] = 1.0
        
        # sample target pos
        p = torch.randn((num_create, 3), device=self.device)
        p = p / p.norm(dim=-1, keepdim=True)
        p[:, 0] = torch.abs(p[:, 0])
        u = torch.rand(num_create, device=self.device).unsqueeze(-1) ** (1.0 / 3)
        object_points = self._fetch_object_points(num_create, rest_object_pose)
        target_pos = p * u * self.target_eef_radius + object_points
        target_pos[:, 0] += 0.02
        target_pos[:, 2] = torch.clamp(target_pos[:, 2], 0.05, 0.8)

        # sample target rot
        raw = torch.zeros((num_create,), device=self.device)
        pitch = torch.zeros((num_create,), device=self.device) - math.pi / 2
        yaw = torch.zeros((num_create,), device=self.device)
        down_q = quat_from_euler_xyz(raw, pitch, yaw)       # point towards the object from the front in handle frame
        if self.target_eef_rotation_mode == 0:
            # version 1: fully random rotation
            target_rot = torch.zeros((num_create, 4), device=self.device)
            p = torch.randn((num_create, 3), device=self.device)
            p = p / p.norm(dim=-1, keepdim=True)
            theta_half = (2 * torch.rand(num_create, device=self.device) - 1) * math.pi * 0.5  # 0.25: half rotation, 0.5: full rotation
            target_rot[:, :3] = p * torch.sin(theta_half).unsqueeze(-1)
            target_rot[:, 3] = torch.cos(theta_half)
            target_rot = quat_mul(target_rot, down_q)
        elif self.target_eef_rotation_mode == 1:
            # version 2: more limited rotation
            # the gripper points towards the object from the front in handle frame at the beginning
            # step 1: rotate along x-axis by an angle in [-pi/2, pi/2]
            # step 2: rotate along a random axis by an angle in [-pi/2, pi/2]
            rot1 = torch.zeros((num_create, 4), device=self.device)
            theta_half = (
                (2 * torch.rand(num_create, device=self.device) - 1) * math.pi * 0.25
            )
            rot1[:, 0] = torch.sin(theta_half)
            rot1[:, 3] = torch.cos(theta_half)
            rot2 = torch.zeros((num_create, 4), device=self.device)
            p = torch.randn((num_create, 3), device=self.device)
            p = p / p.norm(dim=-1, keepdim=True)
            theta_half = (
                (2 * torch.rand(num_create, device=self.device) - 1) * math.pi * 0.25
            )
            rot2[:, :3] = p * torch.sin(theta_half).unsqueeze(-1)
            rot2[:, 3] = torch.cos(theta_half)

            target_rot = quat_mul(rot1, down_q)
            target_rot = quat_mul(rot2, target_rot)
        elif self.target_eef_rotation_mode == 2:
            # version 3: point towards the object
            # rot1: rotate along z-axis by an angle in [-pi, pi] (franka gripper originally points upwards)
            rot1 = torch.zeros((num_create, 4), device=self.device)
            theta_half = (
                (2 * torch.rand(num_create, device=self.device) - 1) * math.pi * 0.5
            )
            rot1[:, 2] = torch.sin(theta_half)
            rot1[:, 3] = torch.cos(theta_half)

            # rot2: point the gripper to the object
            v1 = to_torch(num_create * [0., 0., 1.], device=self.device).reshape(num_create, 3)
            v2 = object_points - target_pos
            v2[:, 0] = torch.clamp(v2[:, 0], max=-1e-2)
            v2 = v2 / v2.norm(dim=-1, keepdim=True)

            axis = torch.cross(v1, v2)
            angle = torch.acos((v1 * v2).sum(dim=-1))
            rot2 = quat_from_angle_axis(angle, axis)

            target_rot = quat_mul(rot2, rot1)

            if not self.vision_based:
                # add noise to the rotation
                noise_rot = torch.zeros((num_create, 4), device=self.device)
                p = torch.randn((num_create, 3), device=self.device)
                p = p / p.norm(dim=-1, keepdim=True)
                theta_half = (2 * torch.rand(num_create, device=self.device) - 1) * math.pi / 12
                noise_rot[:, :3] = p * torch.sin(theta_half).unsqueeze(-1)
                noise_rot[:, 3] = torch.cos(theta_half)
                target_rot = quat_mul(noise_rot, target_rot)

        # transform to world frame
        handle_pose = self._get_handle_pose(object_pose, init_object_dof_pos.unsqueeze(-1))
        target_pos = quat_rotate(handle_pose[:, 3:], target_pos) + handle_pose[:, :3]
        target_rot = quat_mul(handle_pose[:, 3:], target_rot)

        return torch.cat([target_pos, target_rot], dim=-1)

    def _sample_states(self, stage, steps, object_pose, init_object_dof_pos, target_eef_pose):
        resample = (stage == 0) & (steps == 0)
        num_resample = resample.sum()

        if num_resample != 0:
            object_pose[resample], init_object_dof_pos[resample] = self._sample_object_pose(num_resample)
            target_eef_pose[resample] = self._sample_target_eef_pose(
                num_resample, object_pose[resample], init_object_dof_pos[resample]
            )

        return object_pose, init_object_dof_pos, target_eef_pose

    def _generation_step(self, stage, steps, object_pose, init_object_dof_pos, target_eef_pose):
        # set franka states
        ik = stage == 0
        backtrack = stage == 1
        reset = stage == 2

        franka_arm_dof_pos = self.env.arm_dof_pos.clone()

        if ik.any():
            eef_pos = self.env.fingertip_centered_pos
            eef_rot = self.env.fingertip_centered_quat
            target_pos = target_eef_pose[:, :3]
            target_rot = target_eef_pose[:, 3:]

            pos_err = target_pos - eef_pos
            orn_err = self._orientation_error(target_rot, eef_rot)
            dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
            franka_arm_dof_pos[ik, :7] += self._control_ik(dpose)[ik]

        if backtrack.any():
            t = 1 / (self.max_backtrack_iters - steps[backtrack])
            t[steps[backtrack] == 0] = 0.0
            t = t.unsqueeze(-1)
            franka_arm_dof_pos[backtrack] = (1 - t) * franka_arm_dof_pos[
                backtrack
            ] + t * self.franka_default_arm_dof_pos.unsqueeze(0)

        if reset.any():
            franka_arm_dof_pos[reset, :] = self.franka_default_arm_dof_pos

        franka_gripper_pos = torch.ones(self.num_envs, 2, device=self.device) \
            * self.env.asset_info_franka_table.franka_gripper_width_max / 2.0 * self.env.cfg_task.randomize.franka_gripper_initial_state

        franka_dof_pos = torch.cat([franka_arm_dof_pos, franka_gripper_pos], dim=-1)
        franka_dof_pos = tensor_clamp(
            franka_dof_pos,
            self.env.franka_dof_lower_limits,
            self.env.franka_dof_upper_limits,
        )

        self.env.dof_pos[:, :self.env.franka_num_dofs] = franka_dof_pos
        self.env.dof_vel[:] = 0.0
        self.env.dof_torque[:] = 0.0

        # set object states
        self.env.object_dof_pos[:] = init_object_dof_pos.unsqueeze(-1)
        self.env.object_pos[:, :] = object_pose[:, :3]
        self.env.object_quat[:, :] = object_pose[:, 3:]
        self.env.object_linvel[:, :] = 0.0
        self.env.object_angvel[:, :] = 0.0

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

        # Set object state
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.env.root_state),
            gymtorch.unwrap_tensor(object_actor_ids_sim),
            len(object_actor_ids_sim),
        )

        # simulate
        self.env.simulate_and_refresh()

    def generate(self, num_samples):
        """Sample initial states for franka pick tasks
        Args:
            num_samples: number of samples
        Returns:
            init_states: a dict containing object pose and franka dof pos
        """
        franka_dof_pos_list = []
        object_pose_list = []
        init_object_dof_pos_list = []
        total_samples = 0
        total_iters = 0
        total_fails = 0
        ik_solve_avg_steps = 0.0
        backtrack_avg_steps = 0.0

        stage = (
            torch.zeros(self.env.num_envs, dtype=torch.int32, device=self.env.device)
            + 2
        )  # reset mode at first
        steps = torch.zeros(
            self.env.num_envs, dtype=torch.int32, device=self.env.device
        )
        object_pose, init_object_dof_pos = self._sample_object_pose(self.env.num_envs)
        target_eef_pose = torch.zeros(
            (self.env.num_envs, 7), dtype=torch.float32, device=self.env.device
        )

        while total_samples < num_samples:
            object_pose, init_object_dof_pos, target_eef_pose = self._sample_states(
                stage, steps, object_pose, init_object_dof_pos, target_eef_pose
            )
            self._generation_step(stage, steps, object_pose, init_object_dof_pos, target_eef_pose)

            steps += 1
            ik_solved, found, fail = self._check_state(stage, steps, object_pose, init_object_dof_pos, target_eef_pose)

            franka_dof_pos_list.append(self.env.dof_pos[found, :self.env.franka_num_dofs])
            object_pose_list.append(object_pose[found])
            init_object_dof_pos_list.append(init_object_dof_pos[found].reshape(-1, 1))
            total_samples += found.sum()
            total_fails += fail.sum()

            if ik_solved.any():
                ik_solve_avg_steps = (
                    0.1 * steps[ik_solved].float().mean() + 0.9 * ik_solve_avg_steps
                )
            if found.any():
                backtrack_avg_steps = (
                    0.1 * steps[found].float().mean() + 0.9 * backtrack_avg_steps
                )
            print(
                f"Total iters: {total_iters} | Total samples: {total_samples} | IK Steps: {ik_solve_avg_steps:.1f} | Backtrack Steps: {backtrack_avg_steps:.1f} | Fail Ratio: {total_fails / (total_samples + total_fails):.4f}"
            )

            # transfer stage
            reset = stage == 2
            stage[ik_solved] = 1
            stage[found | fail] = 2
            stage[reset] = 0
            steps[ik_solved | reset] = 0

            total_iters += 1

        init_states = {
            "object_pose": torch.cat(object_pose_list, dim=0).cpu(),
            "franka_dof_pos": torch.cat(franka_dof_pos_list, dim=0).cpu(),
            "init_object_dof_pos": torch.cat(init_object_dof_pos_list, dim=0).cpu(),
        }
        return init_states

    def filter_vision(self, init_states, filter_vision_threshold=0.001):
        """Filter initial states where the handle takes up less than `filter_vision_threshold` of the camera input
        Args:
            init_states: a dict containing object pose and franka dof pos
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

            # set object states
            self.env.object_dof_pos[:, :] = init_states["init_object_dof_pos"][start:end]
            self.env.object_pos[:, :] = init_states["object_pose"][start:end, 0:3]
            self.env.object_quat[:, :] = init_states["object_pose"][start:end, 3:7]
            self.env.object_linvel[:, :] = 0.0
            self.env.object_angvel[:, :] = 0.0

            # set franka state
            self.env.dof_pos[:, :self.env.franka_num_dofs] = init_states["franka_dof_pos"][start:end]
            self.env.dof_vel[:, :] = 0.0
            self.env.dof_torque[:] = 0.0

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

            # Set object state
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.env.root_state),
                gymtorch.unwrap_tensor(object_actor_ids_sim),
                len(object_actor_ids_sim),
            )

            # simulate
            self.env.simulate_and_refresh()

            self.gym.simulate(self.sim)
            if self.device != "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)

            self.gym.start_access_image_tensors(self.sim)
            wrist_seg_image_buf_tensor = torch.stack(self.env.camera_tensors, dim=0).reshape(self.num_envs, -1)
            num_pixels = wrist_seg_image_buf_tensor.shape[-1]
            num_object_pixels = (wrist_seg_image_buf_tensor == self.env.object_actor_id_env + 1).sum(dim=-1)
            self.gym.end_access_image_tensors(self.sim)

            # filter if the object is not visible: ratio of pixels < `filter_vision_threshold`
            filtered[start:end] = num_object_pixels < max(5, int(filter_vision_threshold * num_pixels))

            start += self.num_envs

            print(
                f"Filtered {filtered.sum()} out of {start} samples (total {num_samples} samples)."
            )

        for key in init_states:
            init_states[key] = init_states[key][~filtered].cpu()

        return init_states
