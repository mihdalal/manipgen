from isaacgym import gymtorch
from isaacgymenvs.utils.torch_jit_utils import quat_mul, quat_from_euler_xyz

import numpy as np
import torch
import math
import time
import os

from manipgen.utils.initial_state_samplers.base_sampler import BaseSampler
from manipgen.utils.initial_state_samplers.pick_cube_sampler import (
    FrankaPickCubePoseSampler,
)
from manipgen.utils.geometry_utils import sample_mesh_points


class FrankaPickPoseSampler(BaseSampler):
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
            self.env.cfg_base.env.table_height + 0.3,
        ]

        self.object_points_local = self._sample_object_points()

        # for backtracking
        self.franka_default_arm_dof_pos = torch.tensor(
            self.env.cfg_task.randomize.franka_arm_initial_dof_pos,
            device=self.device,
        )

    def _sample_object_points(self, num_samples=1024):
        """Sample points on the object mesh. 
        If vision_based=True, the end-effector will point towards one of these points instead of the mesh center.
        """
        mesh = self.env.object_mesh
        points = sample_mesh_points(mesh, num_samples)
        points = torch.from_numpy(points).float().to(self.device)

        return points

    def _generate_rest_pose(self):
        object_euler_xy = self.env.grasp_data[self.env.object_code][self.env.object_scale]["object_euler_xy"]
        object_z = self.env.grasp_data[self.env.object_code][self.env.object_scale]["object_init_z"]

        self.num_rest_samples = len(object_euler_xy)
        self.object_rest_pos = torch.zeros(
            (self.num_rest_samples, 3), device=self.device
        )
        self.object_rest_pos[:, 2] = (
            object_z + self.env.cfg_base.env.table_height
        )
        self.object_rest_rot = quat_from_euler_xyz(
            object_euler_xy[:, 0],
            object_euler_xy[:, 1],
            torch.zeros(self.num_rest_samples, device=self.device),
        )

        print(f"Loaded {self.num_rest_samples} rest poses.")

    def _sample_object_pose(self, num_create):
        select_rest_pose = torch.randint(
            self.num_rest_samples, (num_create,), device=self.device
        )

        init_pos = 2 * torch.rand((num_create, 3), device=self.device) - 1.0  # position
        init_pos[:, 0] = (
            self.object_pos_center[0] + init_pos[:, 0] * self.object_pos_noise
        )
        init_pos[:, 1] = (
            self.object_pos_center[1] + init_pos[:, 1] * self.object_pos_noise
        )
        init_pos[:, 2] = self.object_rest_pos[select_rest_pose, 2] + 0.001

        rest_rot = self.object_rest_rot[select_rest_pose]
        theta_half = (
            torch.rand(num_create, device=self.device) * math.pi
        )  # rotation along z-axis
        rot_z = torch.zeros((num_create, 4), device=self.device)
        rot_z[:, 2] = torch.sin(theta_half)
        rot_z[:, 3] = torch.cos(theta_half)
        init_rot = quat_mul(rot_z, rest_rot)

        return torch.cat([init_pos, init_rot], dim=-1)

    _fetch_object_points = FrankaPickCubePoseSampler._fetch_object_points
    _sample_target_eef_pose = FrankaPickCubePoseSampler._sample_target_eef_pose
    _sample_states = FrankaPickCubePoseSampler._sample_states
    _generation_step = FrankaPickCubePoseSampler._generation_step
    _check_state = FrankaPickCubePoseSampler._check_state
    generate_inner = FrankaPickCubePoseSampler.generate
    filter_vision = FrankaPickCubePoseSampler.filter_vision

    def generate(self, num_samples):
        """Sample initial states for franka pick tasks
        Args:
            num_samples: number of samples
        Returns:
            init_states: a dict containing object pose and franka dof pos
        """
        self._generate_rest_pose()

        init_states = self.generate_inner(num_samples)

        return init_states