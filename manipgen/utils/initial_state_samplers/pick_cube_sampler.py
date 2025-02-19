from isaacgym import gymtorch
from isaacgymenvs.utils.torch_jit_utils import (
    tensor_clamp, 
    quat_from_angle_axis, 
    quat_conjugate, 
    quat_mul, 
    quat_rotate,
    to_torch, 
)

import numpy as np
import torch
import math
import time

from manipgen.utils.initial_state_samplers.base_sampler import BaseSampler

@torch.jit.script
def find_nearest_point(object_points, eef_pos_local):
    # type: (Tensor, Tensor) -> Tensor

    num_create = eef_pos_local.shape[0]
    num_points = object_points.shape[0]
    dist = torch.norm(
        object_points.expand(num_create, num_points, 3) - eef_pos_local.unsqueeze(1),
        dim=-1,
    )
    _, min_idx = torch.min(dist, dim=-1)

    return min_idx

class FrankaPickCubePoseSampler(BaseSampler):
    """Sample initial poses for franka pick tasks using inverse kinematics
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
            object_pos_noise: noise for sampling object position
        """
        super().__init__(env, vision_based, cfg)
        self.damping = self.cfg.damping  # inverse kinematics coefficient
        self.max_ik_iters = self.cfg.max_ik_iters
        self.max_backtrack_iters = self.cfg.max_backtrack_iters
        self.ik_solver_pos_tolerance = self.cfg.ik_solver_pos_tolerance
        self.ik_solver_rot_tolerance = self.cfg.ik_solver_rot_tolerance

        self.target_eef_radius = self.cfg.target_eef_radius
        self.target_eef_z_offset = self.cfg.target_eef_z_offset
        self.target_eef_rotation_mode = 2   # 0: fully random, 1: more limited, 2: point towards the object
        self.object_pos_noise = self.cfg.object_pos_noise
        self.object_pos_center = [
            self.env.table_pose.p.x,
            self.env.table_pose.p.y,
            self.env.cfg_base.env.table_height + 0.5 * self.env.box_size,
        ]

        self.object_points_local = self._sample_object_points(cube_size=self.env.box_size)
        
        # for backtracking
        self.franka_default_arm_dof_pos = torch.tensor(
            self.env.cfg_task.randomize.franka_arm_initial_dof_pos,
            device=self.device,
        )

    def _sample_object_points(self, cube_size, num_samples=1024):
        """Sample points in the cube. 
        If vision_based=True, the end-effector will point towards one of these points instead of the mesh center.
        """
        points = torch.rand((num_samples, 3), device=self.device) * 2 - 1
        points *= cube_size / 2
        return points
    
    def _fetch_object_points(self, num_create, object_pose, nearest=False, eef_pos=None):
        object_pos, object_rot = object_pose[:, :3], object_pose[:, 3:]

        if nearest:
            assert eef_pos is not None
            target_pos_local = quat_rotate(
                quat_conjugate(object_rot), eef_pos - object_pos
            )
            idx = find_nearest_point(
                self.object_points_local, target_pos_local
            )
        else:
            idx = torch.randint(self.object_points_local.shape[0], (num_create,), device=self.device)

        object_points = quat_rotate(
            object_rot, self.object_points_local[idx]
        ) + object_pos

        return object_points

    def _sample_object_pose(self, num_create):
        init_pos = 2 * torch.rand((num_create, 3), device=self.device) - 1.0  # position
        init_pos[:, 0] = (
            self.object_pos_center[0] + init_pos[:, 0] * self.object_pos_noise
        )
        init_pos[:, 1] = (
            self.object_pos_center[1] + init_pos[:, 1] * self.object_pos_noise
        )
        init_pos[:, 2] = self.object_pos_center[2]

        theta_half = (
            (2 * torch.rand(num_create, device=self.device) - 1) * math.pi / 8
        )  # rotation along z-axis
        init_rot = torch.zeros((num_create, 4), device=self.device)
        init_rot[:, 2] = torch.sin(theta_half)
        init_rot[:, 3] = torch.cos(theta_half)

        return torch.cat([init_pos, init_rot], dim=-1)

    def _sample_target_eef_pose(self, num_create, object_pose):
        # sample target pos
        p = torch.randn((num_create, 3), device=self.device)
        p = p / p.norm(dim=-1, keepdim=True)
        p[:, -1] = torch.abs(p[:, -1])
        u = torch.rand(num_create, device=self.device).unsqueeze(-1) ** (1.0 / 3)
        object_points = self._fetch_object_points(num_create, object_pose)
        target_pos = p * u * self.target_eef_radius + object_points
        target_pos[:, -1] += self.target_eef_z_offset

        # sample target rot
        down_q = to_torch(num_create * [1.0, 0., 0., 0.], device=self.device).reshape(num_create, 4)
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
            # the gripper points downwards at the beginning
            # step 1: rotate along z-axis by an angle in [-pi/2, pi/2]
            # step 2: rotate along a random axis by an angle in [-pi/2, pi/2]
            rot1 = torch.zeros((num_create, 4), device=self.device)
            theta_half = (
                (2 * torch.rand(num_create, device=self.device) - 1) * math.pi * 0.25
            )
            rot1[:, 2] = torch.sin(theta_half)
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
            rot1 = torch.zeros((num_create, 4), device=self.device)
            theta_half = (
                (2 * torch.rand(num_create, device=self.device) - 1) * math.pi * 0.5
            )
            rot1[:, 2] = torch.sin(theta_half)
            rot1[:, 3] = torch.cos(theta_half)

            v1 = to_torch(num_create * [0., 0., 1.], device=self.device).reshape(num_create, 3)
            v2 = object_points - target_pos
            v2[:, -1] = torch.clamp(v2[:, -1], max=0.0)
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

        return torch.cat([target_pos, target_rot], dim=-1)

    def _sample_states(self, stage, steps, object_pose, target_eef_pose):
        resample = (stage == 0) & (steps == 0)
        num_resample = resample.sum()

        if num_resample != 0:
            object_pose[resample] = self._sample_object_pose(num_resample)
            target_eef_pose[resample] = self._sample_target_eef_pose(
                num_resample, object_pose[resample]
            )

        return object_pose, target_eef_pose

    def _generation_step(self, stage, steps, object_pose, target_eef_pose):
        # set franka states
        ik = stage == 0
        backtrack = stage == 1
        reset = stage == 2

        franka_arm_dof_pos = self.env.arm_dof_pos

        if ik.any():
            eef_pos = self.env.fingertip_centered_pos
            eef_rot = self.env.fingertip_centered_quat
            target_pos = target_eef_pose[:, :3]
            target_rot = target_eef_pose[:, 3:]

            pos_err = target_pos - eef_pos
            orn_err = self._orientation_error(target_rot, eef_rot)
            dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
            franka_arm_dof_pos[ik] += self._control_ik(dpose)[ik]

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
            * self.env.asset_info_franka_table.franka_gripper_width_max / 2.0

        franka_dof_pos = torch.cat([franka_arm_dof_pos, franka_gripper_pos], dim=-1)
        franka_dof_pos = tensor_clamp(
            franka_dof_pos,
            self.env.franka_dof_lower_limits,
            self.env.franka_dof_upper_limits,
        )

        self.env.dof_pos[:, :self.env.franka_num_dofs] = franka_dof_pos
        self.env.dof_vel[:] = 0.0
        self.env.dof_torque[:] = 0.0

        franka_actor_ids_sim = self.env.franka_actor_ids_sim.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.env.dof_state),
            gymtorch.unwrap_tensor(franka_actor_ids_sim),
            len(franka_actor_ids_sim),
        )

        self.gym.set_dof_actuation_force_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(torch.zeros_like(self.env.dof_torque)),
            gymtorch.unwrap_tensor(franka_actor_ids_sim),
            len(franka_actor_ids_sim),
        )

        # set object states
        self.env.object_pos[:, :] = object_pose[:, :3]
        self.env.object_quat[:, :] = object_pose[:, 3:]
        self.env.object_linvel[:, :] = 0.0
        self.env.object_angvel[:, :] = 0.0

        object_actor_ids_sim = self.env.object_actor_ids_sim.to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.env.root_state),
            gymtorch.unwrap_tensor(object_actor_ids_sim),
            len(object_actor_ids_sim),
        )

        # simulate
        self.env.simulate_and_refresh()

    def _check_state(self, stage, steps, target_eef_pose):
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
        
        # check collision
        contact = self.env.check_contact(self.env.franka_rigid_body_ids_sim)

        # check if the object is in the camera view
        visible = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        if self.vision_based:
            camera_pose = self._get_camera_pose()
            object_pose = torch.cat([self.env.object_pos, self.env.object_quat], dim=-1)
            visible = self._check_object_in_camera_view(
                camera_pose, object_pose, self.object_points_local,
                fov=self.env.cfg_task.env.local_obs.horizontal_fov, clamp_dist=0.28
            )
        
        # backtrack found
        found = (~contact) & (stage == 1) & visible
        # backtrack timeout
        backtrack_timeout = (
            (steps == self.max_backtrack_iters) & (stage == 1) & (~found)
        )
        # assert backtrack_timeout.sum() == 0     # this should not happen

        fail = ik_timeout | backtrack_timeout

        return ik_solved, found, fail

    def generate(self, num_samples):
        """Sample initial states for franka pick tasks
        Args:
            num_samples: number of samples
        Returns:
            init_states: a dict containing object pose and franka dof pos
        """
        franka_dof_pos_list = []
        object_pose_list = []
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
        object_pose = self._sample_object_pose(self.env.num_envs)
        target_eef_pose = torch.zeros(
            (self.env.num_envs, 7), dtype=torch.float32, device=self.env.device
        )

        while total_samples < num_samples:
            object_pose, target_eef_pose = self._sample_states(
                stage, steps, object_pose, target_eef_pose
            )
            self._generation_step(stage, steps, object_pose, target_eef_pose)

            steps += 1
            ik_solved, found, fail = self._check_state(stage, steps, target_eef_pose)

            franka_dof_pos_list.append(self.env.dof_pos[found, :self.env.franka_num_dofs])
            object_pose_list.append(object_pose[found])
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
        }
        return init_states

    def filter_vision(self, init_states, filter_vision_threshold=0.001):
        """Filter initial states where the object takes up less than `filter_vision_threshold` of the camera input
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

            self.env.object_pos[:, :] = init_states["object_pose"][start:end, 0:3]
            self.env.object_quat[:, :] = init_states["object_pose"][start:end, 3:7]
            self.env.object_linvel[:, :] = 0.0
            self.env.object_angvel[:, :] = 0.0

            object_actor_ids_sim = self.env.object_actor_ids_sim.to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.env.root_state),
                gymtorch.unwrap_tensor(object_actor_ids_sim),
                len(object_actor_ids_sim),
            )

            self.env.dof_pos[:, :self.env.franka_num_dofs] = init_states["franka_dof_pos"][start:end]
            self.env.dof_vel[:] = 0.0
            self.env.dof_torque[:] = 0.0

            franka_actor_ids_sim = self.env.franka_actor_ids_sim.to(dtype=torch.int32)
            self.gym.set_dof_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.env.dof_state),
                gymtorch.unwrap_tensor(franka_actor_ids_sim),
                len(franka_actor_ids_sim),
            )

            self.gym.set_dof_actuation_force_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(torch.zeros_like(self.env.dof_torque)),
                gymtorch.unwrap_tensor(franka_actor_ids_sim),
                len(franka_actor_ids_sim),
            )

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
