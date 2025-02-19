import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base="1.1", config_path="./config", config_name="config_sample")
def sample(cfg: DictConfig):
    from isaacgym import gymapi
    from isaacgym import gymutil
    from isaacgym import gymtorch

    import torch
    import time
    from pathlib import Path
    from copy import deepcopy
    
    from isaacgymenvs.utils.utils import set_seed
    from manipgen.envs import environments
    from manipgen.utils.initial_state_samplers import samplers
  
    set_seed(cfg.seed)
    
    # basic setup
    cfg.sample_mode = True
    if cfg.task_name == "place_cube" and cfg.easy_mode:
        # place_cube env disables contact between table and receptable in standard mode
        # but we need to enable it for sampling in easy mode
        cfg.sample_mode = False
    init_state_dir = Path("init_states") if not cfg.init_state_dir else Path(cfg.init_state_dir)
    init_state_dir.mkdir(exist_ok=True)

    # sampler config
    sampler_args = {"env": None, "vision_based": cfg.vision_based, "cfg": cfg.task.sampler}
    task_name = cfg.task_name
    if task_name in ("pick", "place"):
        code = cfg.task.env.object_code.replace("/", "-")
        scale = int(100 * cfg.task.env.object_scale)
        task_name = task_name + "_" + code + "_" + f"{scale:03d}"

        if cfg.task_name == "place":
            if sampler_args["cfg"].policy_init_states_path == "":
                sampler_args["cfg"].policy_init_states_path = f"init_states/franka_pick_{code}_{scale:03d}_init_states.pt"
            sampler_args["cfg"].policy_checkpoint_path = cfg.checkpoint
            if cfg.checkpoint == "":
                sampler_args["cfg"].policy_checkpoint_path = f"runs/pick/franka_pick_{code}_{scale:03d}/nn/franka_pick.pth"

    elif task_name in ("grasp_handle", "open", "close", "open_nograsp", "close_nograsp"):
        code = cfg.task.env.object_code
        task_name = task_name + "_" + code

        if cfg.task_name in ("open", "close"):
            if sampler_args["cfg"].policy_init_states_path == "":
                sampler_args["cfg"].policy_init_states_path = f"init_states/franka_grasp_handle_{code}_init_states.pt"
            sampler_args["cfg"].policy_checkpoint_path = cfg.checkpoint
            if cfg.checkpoint == "":
                sampler_args["cfg"].policy_checkpoint_path = f"runs/grasp_handle/franka_grasp_handle_{code}/nn/franka_grasp_handle.pth"

    init_states_all = None
    num_init_states = 0

    for iteration in range(cfg.max_iters):
        if num_init_states >= cfg.num_samples:
            break

        if iteration == 0:
            target_num_samples = cfg.num_samples
        else:
            target_num_samples = (cfg.num_samples - num_init_states) * (iteration + 1)
        print(f"Start iteration {iteration}.")
        print(f"Need to generate {target_num_samples} samples ({num_init_states} already generated).")
        
        # set up env
        if iteration == 0 or cfg.filter_vision:
            env = environments[cfg.task_name](cfg=cfg)
        
        # set up sampler
        sampler_args["env"] = env
        sampler = samplers[cfg.task_name](**sampler_args)

        # generate initial states
        tik = time.time()
        if cfg.easy_mode:
            init_states = sampler.generate_easy(target_num_samples)
        else:
            init_states = sampler.generate(target_num_samples)
        tok = time.time()
        num_samples = init_states["franka_dof_pos"].shape[0]
        print("Generate {} samples in {:.4f} seconds.".format(num_samples, tok - tik))

        # shuffle initial states
        idx = torch.randperm(num_samples)
        for key in init_states.keys():
            init_states[key] = init_states[key][idx]

        # filter initial states
        if cfg.filter:
            tik = time.time()
            init_states = sampler.filter(init_states)
            tok = time.time()
            print("Filter in {:.4f} seconds.".format(tok - tik))

        # filter initial states based on vision
        if cfg.filter_vision:
            # reload environment
            env.destroy()
            cfg_vision = deepcopy(cfg)
            cfg_vision.sample_mode = False
            cfg_vision.num_envs = cfg_vision.task.env.numEnvs = 256
            cfg_vision.render = False
            cfg_vision.capture_video = False
            cfg_vision.local_obs = True
            env = environments[cfg_vision.task_name](
                cfg=cfg_vision,
            )
            sampler_args["env"] = env
            sampler = samplers[cfg.task_name](**sampler_args)
            tik = time.time()
            init_states = sampler.filter_vision(init_states, cfg.filter_vision_threshold)
            tok = time.time()
            print("Filter vision in {:.4f} seconds.".format(tok - tik))
            env.destroy()

        # merge initial states
        if init_states_all is None:
            init_states_all = init_states
        else:
            for key in init_states_all.keys():
                init_states_all[key] = torch.cat([init_states_all[key], init_states[key]], dim=0)
        first_key = list(init_states_all.keys())[0]
        num_init_states = len(init_states_all[first_key])

    print("Generated {} samples.".format(num_init_states))
    if num_init_states < cfg.num_samples:
        print("Warning: not enough samples generated. Please check the sampler or increase cfg.max_iters.")

    # save initial states
    target_path = (
        cfg.init_states
        if cfg.init_states
        else init_state_dir / f"franka_{task_name}_init_states.pt"
    )
    torch.save(init_states_all, target_path)
    print("Saved init states to {}.".format(str(target_path)))


if __name__ == "__main__":
    sample()

