import argparse
import hydra
from manipgen.real_world.neural_mp import NeuralMP
from manipgen.real_world.manipgen import ManipGen
from manipgen.real_world.utils.neural_mp_env_wrapper import IndustrealEnvWrapper
from manipgen.real_world.industreal_psl_env import FrankaRealPSLEnv
from manipgen.real_world.vlm_planner import VLMPlanner

def get_args():
    """Gets arguments from the command line."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--debug_mode",
        action="store_true",
        required=False,
        help="Enable output for debugging",
    )
    parser.add_argument(
        "-c",
        "--vlm_cam_idx",
        required=True,
        type=int,
        help="Choose one global camera for VLM planning",
    )
    parser.add_argument(
        "--prompt",
        type=str, default="put everything on the table in the black box",
    )
    parser.add_argument(
        "--pick_checkpoint",
        type=str, default="real_world/checkpoints/pick.pth",
    )
    parser.add_argument(
        "--place_checkpoint",
        type=str, default="real_world/checkpoints/place.pth",
    )
    parser.add_argument(
        "--grasp_handle_checkpoint",
        type=str, default="real_world/checkpoints/grasp_handle.pth",
    )
    parser.add_argument(
        "--open_checkpoint",
        type=str, default="real_world/checkpoints/open.pth",
    )
    parser.add_argument(
        "--close_checkpoint",
        type=str, default="real_world/checkpoints/close.pth",
    )
    parser.add_argument(
        "--neural_mp_checkpoint",
        type=str, default="real_world/checkpoints/neural_mp.pth",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()
    hydra.initialize(config_path="../../config/real/", job_name="run_manipgen")
    cfg = hydra.compose(config_name='base')

    # load local policies
    env = FrankaRealPSLEnv(args, cfg)
    env.load_policy(task="pick", checkpoint_path=args.pick_checkpoint)
    env.load_policy(task="place", checkpoint_path=args.place_checkpoint)
    env.load_policy(task="grasp_handle", checkpoint_path=args.grasp_handle_checkpoint)
    env.load_policy(task="open", checkpoint_path=args.open_checkpoint)
    env.load_policy(task="close", checkpoint_path=args.close_checkpoint)
    
    # motion planning
    neural_mp_env = IndustrealEnvWrapper(env)
    motion_planner = NeuralMP(
        neural_mp_env,
        model_path=args.neural_mp_checkpoint, 
        train_mode=True,
        tto=True, 
        in_hand=False, 
        max_neural_mp_rollout_length=100, 
        num_robot_points=2048, 
        num_obstacle_points=4096,
        clamp_joint_limit=True,
        visualize=False,
    )

    # VLM planning
    vlm_planner = VLMPlanner(env, cam_ids=[args.vlm_cam_idx])
    
    manipgen = ManipGen(env, vlm_planner, motion_planner)
    manipgen.solve_task(args.prompt, args.debug_mode)
