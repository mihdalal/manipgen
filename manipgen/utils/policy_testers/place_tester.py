from isaacgym import gymtorch

import torch

from manipgen.utils.policy_testers.base_tester import BaseTester
from manipgen.utils.rlgames_utils import get_actions

class FrankaPlaceTester(BaseTester):
    def __init__(self, env, policy, num_steps, task_name):
        super().__init__(env, policy, num_steps, task_name)
    
    def pre_steps(self, obs, keep_runs):
        self.env.pre_steps(obs, keep_runs)
        return obs, keep_runs
    
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
            # obs, keep_runs = self.pre_steps(obs, keep_runs)
            # self.env.progress_buf[:] = 0
            assert "We do not have a pre_steps for place. Please specify the initial states"

        obs = self.env.reset()

        images = []
        for step in range(self.num_steps):
            actions = get_actions(obs, self.policy, is_deterministic=is_deterministic)
            obs, reward, done, info = self.env.step(actions)

            if step % 3 == 2:
                images.append(info["camera_images"])

        hit_joint_limits = info["hit_joint_limits"]
        keep_runs = keep_runs & ~hit_joint_limits

        teleportation_images = []
        # open gripper
        self.env.ctrl_target_fingertip_midpoint_pos[:] = self.env.fingertip_centered_pos
        self.env.ctrl_target_fingertip_midpoint_quat[:] = self.env.fingertip_centered_quat
        target_gripper_dof_pos = self.env.gripper_dof_pos[:] + 0.02
        camera_images = self.env.move_gripper_to_target_pose(gripper_dof_pos=target_gripper_dof_pos, sim_steps=40, get_camera_images=True)
        teleportation_images.extend(camera_images)
        # pull out the arm
        self.env.ctrl_target_fingertip_midpoint_pos[:] -= 0.15 * self.env.gripper_direction
        camera_images = self.env.move_gripper_to_target_pose(gripper_dof_pos=target_gripper_dof_pos, sim_steps=30, get_camera_images=True)
        teleportation_images.extend(camera_images)
        teleportation_images = [teleportation_images[i] for i in range(0, len(teleportation_images), 3)]
        images.extend(teleportation_images)

        success = self.env._check_place_success()

        self.env.reset_idx(torch.arange(self.num_envs, device=self.device))

        return keep_runs, success, images
