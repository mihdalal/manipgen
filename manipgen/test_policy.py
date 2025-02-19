from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch

import torch
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from isaacgymenvs.utils.reformat import omegaconf_to_dict
from isaacgymenvs.utils.utils import set_seed
from manipgen.envs import environments
from manipgen.utils.policy_testers import testers
from manipgen.utils.rlgames_utils import load_model


@hydra.main(version_base="1.1", config_path="./config", config_name="config_test")
def test_policy(cfg: DictConfig):
    set_seed(cfg.seed)

    # set up environment
    use_init_states = cfg.init_states != ""
    init_states = None
    if use_init_states:
        init_states = torch.load(cfg.init_states)
        # reverse the order of the states
        for k, v in init_states.items():
            init_states[k] = torch.flip(v, [0])
            
    env = environments[cfg.task_name](
        cfg,
        init_states=init_states
    )
    env.disable_automatic_reset = True

    # set up policy
    policy = load_model(
        actions_num=env.num_actions,
        obs_shape=(env.num_obs,),
        device=env.device,
        checkpoint_path=cfg.checkpoint,
        rl_config=omegaconf_to_dict(cfg["train"]),
    )

    # set up tester
    task_name = cfg.task_name
    if task_name in ("pick", "place"):
        code = cfg.task.env.object_code.replace("/", "-")
        scale = int(100 * cfg.task.env.object_scale)
        task_name = task_name + "_" + code + "_" + f"{scale:03d}"
    elif task_name in ("grasp_handle", "open", "close", "open_nograsp", "close_nograsp"):
        task_name = task_name + "_" + cfg.task.env.object_code
    tester = testers[cfg.task_name](
        env=env,
        policy=policy,
        num_steps=cfg.num_steps,
        task_name=task_name,
    )

    # generate initial states
    tester.run(
        num_iterations=cfg.num_iterations,
        collect_failed_runs=cfg.collect_failed_runs,
        save_success_rate=cfg.save_success_rate,
        is_deterministic=cfg.deterministic_policy,
    )


if __name__ == "__main__":
    test_policy()
