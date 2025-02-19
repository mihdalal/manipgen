from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch

import torch
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner, _restore, _override_sigma
from rl_games.algos_torch import players
from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, MultiObserver
from isaacgymenvs.utils.utils import set_seed
from isaacgymenvs.utils.reformat import omegaconf_to_dict

import hydra
from omegaconf import DictConfig, OmegaConf
import time
import signal
import os

from manipgen.envs import environments
from manipgen.utils.rlgames_utils import (
    PSLAlgoObserver,
    WandbAlgoObserver,
    A2CAgent_ManipGen,
)


@hydra.main(version_base="1.1", config_path="./config", config_name="config_train")
def train(cfg: DictConfig):

    # ===================== Env =====================
    def create_isaacgym_env(**kwargs):
        if cfg.task.use_init_states:
            if cfg.init_states != "":
                init_states = torch.load(cfg.init_states)
            else:
                task_name = cfg.task_name
                if task_name in ("pick", "place"):
                    code = cfg.task.env.object_code.replace("/", "-")
                    scale = int(100 * cfg.task.env.object_scale)
                    task_name = task_name + "_" + code + "_" + f"{scale:03d}"
                elif task_name in ("grasp_handle", "open", "close", "open_nograsp", "close_nograsp"):
                    task_name = task_name + "_" + cfg.task.env.object_code
                init_states = torch.load(f"init_states/franka_{task_name}_init_states.pt")

            idx = torch.randperm(len(list(init_states.values())[0]))
            for key in init_states.keys():
                init_states[key] = init_states[key][idx]
        else:
            init_states = None
        envs = environments[cfg.task_name](
            cfg,
            init_states
        )
        return envs

    set_seed(cfg.seed)
    env_configurations.register(
        "rlgpu",
        {
            "vecenv_type": "RLGPU",
            "env_creator": lambda **kwargs: create_isaacgym_env(**kwargs),
        },
    )
    
    vecenv.register(
        "RLGPU",
        lambda config_name, num_actors, **kwargs: RLGPUEnv(
            config_name, num_actors, **kwargs
        ),
    )

    # ===================== RL =====================
    rlgames_config = omegaconf_to_dict(cfg["train"])
    if cfg.exp_name != "":
        rlgames_config["params"]["config"]["full_experiment_name"] = cfg.exp_name
    rlgames_config["params"]["seed"] = cfg.seed
    rlgames_config["params"]["config"]["device"] = cfg.device
    rlgames_config["params"]["config"]["train_dir"] = cfg.train_dir

    observers = [
        PSLAlgoObserver(),
    ]
    if cfg.wandb_activate:
        observers.append(WandbAlgoObserver(cfg))
    runner = Runner(MultiObserver(observers))
    runner.algo_factory.register_builder(
        "a2c_continuous_sp", lambda **kwargs: A2CAgent_ManipGen(**kwargs)
    )
    runner.player_factory.register_builder(
        "a2c_continuous_sp", lambda **kwargs: players.PpoPlayerContinuous(**kwargs)
    )
    runner.load(rlgames_config)
    runner.reset()
    
    # set up rl agent
    agent = runner.algo_factory.create(runner.algo_name, base_name='run', params=runner.params)
    _restore(agent, {"checkpoint": cfg.checkpoint})
    _override_sigma(agent, {})
    def handle(signum, frame):
        print("Signal handler called with signal", signum)
        print("Saving checkpoint before exiting...")
        agent.save(os.path.join(agent.nn_dir, "checkpoint_latest"))
        exit()
    signal.signal(signal.SIGUSR1, handle)

    tik = time.time()
    agent.train()
    tok = time.time()
    print(
        "{} finished in {:.2f} seconds".format(
            "Training" if not cfg.test else "Testing", tok - tik
        )
    )


if __name__ == "__main__":
    train()
