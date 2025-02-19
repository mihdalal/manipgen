from isaacgym import gymtorch

import torch

from manipgen.utils.rlgames_utils import get_actions
from manipgen.utils.media_utils import make_gif_from_numpy

import time
import multiprocessing
from copy import deepcopy


class BaseTester:
    def __init__(self, env, policy, num_steps, task_name):
        """
        Args:
            env: isaacgym environment
            policy: policy for the task
            num_steps: number of steps in each test
            task_name: name of the task
        """
        self.env = env
        self.policy = policy
        self.num_steps = num_steps
        self.task_name = task_name
        self.device = env.device
        self.num_envs = env.num_envs

        self.env.disable_automatic_reset = True
        self.env.disable_hardcode_control = True

    def pre_steps(self, obs, keep_runs):
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
            obs, keep_runs = self.pre_steps(obs, keep_runs)
            self.env.progress_buf[:] = 0

        images = []
        for step in range(self.num_steps):
            actions = get_actions(obs, self.policy, is_deterministic=is_deterministic)
            obs, reward, done, info = self.env.step(actions)

            if step % 3 == 2:
                images.append(info["camera_images"])

        success = info["success"].bool()

        self.env.reset_idx(torch.arange(self.num_envs, device=self.device))

        return keep_runs, success, images

    def run(
        self,
        num_iterations=-1,
        collect_failed_runs=False,
        save_success_rate="",
        is_deterministic=True,
    ):
        """
        Args:
            num_iterations: number of iterations to run, -1 means infinite
            collect_failure_runs: if true, only collect the videos of failed runs
            save_success_rate: if not empty, save success rate to the file
            is_deterministic: if true, use deterministic policy
        """

        total_iters = 0
        total_success = 0
        total_runs = 0
        gif_processes = []

        while num_iterations == -1 or total_iters < num_iterations:
            run_pre_steps = self.env.init_states is None
            keep_runs, success, images = self.run_steps(run_pre_steps=run_pre_steps, is_deterministic=is_deterministic)

            success *= keep_runs
            if keep_runs.sum().item() == 0:
                keep_runs[0] = True
            success_num = success.sum().item()
            run_num = keep_runs.sum().item()

            total_iters += 1
            total_success += success_num
            total_runs += run_num
            print(
                f"Iter: {total_iters} | Iter success: {100 * success_num / run_num:.2f} "
                f"| Overall success: {100 * total_success / total_runs:.2f} | "
                f"Num fail: {int(run_num - success_num)} | Filtered: {self.num_envs - run_num}"
            )

            if images[0] is not None:
                # process images
                if self.env.capture_obs_camera:
                    images = [
                        [env_images[0] for env_images in step_images]
                        for step_images in images
                    ]
                else:
                    images = [
                        [env_images[0][100:700, 320:1120] for env_images in step_images]
                        for step_images in images
                    ]  # crop images for better visualization
                images = [
                    [step_images[env_id] for step_images in images]
                    for env_id in range(self.env.capture_envs)
                ]

                # save images to gif
                if len(gif_processes) > 0:
                    for p in gif_processes:
                        p.join()
                    gif_processes = []

                success_buffer = success.clone()
                valid_buffer = keep_runs.clone()
                image_buffer = deepcopy(images)
                iter_id = total_iters

                def save_gif(end_id):
                    gif_name = f"{self.task_name}_{iter_id}_{end_id}"
                    if self.env.capture_depth:
                        gif_name += "_depth"
                    make_gif_from_numpy(
                        "media/",
                        image_buffer[env_id],
                        name=gif_name,
                    )

                for env_id in range(self.env.capture_envs):
                    if collect_failed_runs and (success_buffer[env_id] or ~valid_buffer[env_id]):
                        continue
                    p = multiprocessing.Process(target=save_gif, args=(env_id,))
                    p.start()
                    gif_processes.append(p)

        if save_success_rate != "":
            with open(save_success_rate, "a") as f:
                if hasattr(self.env, "object_code") and hasattr(self.env, "object_scale"):
                    f.write(
                        f"{self.env.object_code} {self.env.object_scale:.2f} {100 * total_success / total_runs:.4f}\n"
                    )
                elif hasattr(self.env, "object_code"):
                    f.write(
                        f"{self.env.object_code} {100 * total_success / total_runs:.4f}\n"
                    )
                else:
                    f.write(
                        f"{100 * total_success / total_runs:.4f}\n"
                    )
