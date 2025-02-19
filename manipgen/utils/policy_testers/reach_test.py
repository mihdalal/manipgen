from isaacgym import gymtorch

import torch

from manipgen.utils.policy_testers.base_tester import BaseTester


class FrankaReachTester(BaseTester):
    def __init__(self, env, policy, num_steps, task_name):
        env.init_states = None
        super().__init__(env, policy, num_steps, task_name)
        self.num_pre_steps = 0
        
    def pre_steps(self, obs, keep_runs):
        self.env.pre_steps(obs, keep_runs, self.num_pre_steps)
        return obs, keep_runs
