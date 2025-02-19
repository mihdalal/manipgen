from isaacgym import gymtorch

import torch

from manipgen.utils.rlgames_utils import get_actions
from manipgen.utils.policy_testers.base_tester import BaseTester


class FrankaOpenTester(BaseTester):
    def __init__(self, env, policy, num_steps, task_name):
        super().__init__(env, policy, num_steps, task_name)

    def pre_steps(self, obs, keep_runs):
        assert False, "`pre_steps` is not implemented for this testers. Please specify initial states."

    def run_steps(self, run_pre_steps=True, is_deterministic=True):
        """
        Args:
            run_pre_steps: if true, teleport franka to easy initial poses or take actions with hand-engineering code before running the test
        Returns:
            keep_runs: a tensor of shape (num_envs,) indicating whether the run is kept. A run might fail in pre_steps, which should not be counted
            success: a tensor of shape (num_envs,) indicating whether the run is successful
            images: a list of lists of images, each list of images is for one environment
        """
        obs = self.env.reset()
        keep_runs = torch.ones(self.num_envs, device=self.device).bool()
        if run_pre_steps:
            obs, keep_runs = self.pre_steps(obs, keep_runs)
            self.env.progress_buf[:] = 0

        # check whether the initial states are valid
        self.env.ctrl_target_fingertip_midpoint_pos[:] = self.env.fingertip_centered_pos
        self.env.ctrl_target_fingertip_midpoint_quat[:] = self.env.fingertip_centered_quat
        self.env.move_gripper_to_target_pose(gripper_dof_pos=-0.04, sim_steps=60)
        keep_runs &= self.env.gripper_dof_pos.sum(dim=1) > 0.002

        images = []
        for step in range(self.num_steps):
            actions = get_actions(obs, self.policy, is_deterministic=is_deterministic)
            obs, reward, done, info = self.env.step(actions)

            if step % 3 == 2:
                images.append(info["camera_images"])

        success = torch.clamp(self.env.relative_dof_completeness, max=1.0)

        self.env.reset_idx(torch.arange(self.num_envs, device=self.device))

        return keep_runs, success, images
