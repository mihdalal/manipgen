import subprocess
import os
import argparse
if __name__ == "__main__":
    # argparse the following args:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default='pick')
    parser.add_argument("--model", type=str, default='mlp')
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--not_multitask", action='store_true')
    parser.add_argument("--num_objects_per_iteration", type=int, default=100)
    parser.add_argument("--max_epochs_per_iteration", type=int, default=100)
    args = parser.parse_args()    
    
    task = args.task
    model = args.model
    gpu_id = args.gpu_id
    num_objects_per_iteration = args.num_objects_per_iteration
    max_epochs_per_iteration = args.max_epochs_per_iteration
    gpu_cmd = {
        "DISABLE_LAYER_NV_OPTIMUS_1": "1",
        "DISABLE_LAYER_AMD_SWITCHABLE_GRAPHICS_1": "1",
        "MESA_VK_DEVICE_SELECT": "10de:2230",
        "DISPLAY": ":1",
        "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json",
        "ENABLE_DEVICE_CHOOSER_LAYER": "1",
        "VULKAN_DEVICE_INDEX": f"{gpu_id}",
        "CUDA_VISIBLE_DEVICES": f"{gpu_id}",
    }
    iters = len(os.listdir(f"object_lists/{task}/")) - 1
    max_epochs = max_epochs_per_iteration
    if args.not_multitask:
        iters = 1
        max_epochs = 1000000
    for outer_epoch in range(1000):
        for idx in range(iters):
            command_list =[
                "python", "dagger.py", f"task={task}", f"exp_name=dagger_{task}_{model}", f"wandb_run_name=dagger_{task}_{model}",
                f"checkpoint=multitask/{task}/policies",
                f"init_states=multitask/{task}/init_states",
                "num_envs=32", "wandb_activate=True", "wandb_project=dagger-multitask",
                "dagger.buffer_size=100", "dagger.batch_size=2048",
                "dagger.lr=0.0001", "dagger.num_transitions_per_iter=120", "dagger.num_learning_epochs=1",
                f"dagger.student_cfg_path=config/dagger/robomimic/bc_{model}.json",
                f"object_list=object_lists/unidexgrasp/{idx}.txt",
                "dagger.multitask=True", "local_obs=True", "global_obs=False", "dagger.visual_obs_type=depth",
                "test_episodes=100", "capture_local_obs=False", "capture_video=False", "test_frequency=1000",
                f"dagger.num_learning_iterations={max_epochs}",
                "task.sim.physx.max_gpu_contact_pairs=16777216",
                "train_dir=runs/multitask/"
            ]
            # check if runs_dagger/dagger_{task}_{model}/nn/checkpoint_latest.pth exists, if so resume from that path
            if os.path.exists(f"runs/multitask/dagger_{task}_{model}/nn/checkpoint_latest.pth"):
                command_list.append(f"resume=runs/multitask/dagger_{task}_{model}/nn/checkpoint_latest.pth")
            print("Running command:", " ".join(command_list))
            subprocess.run(command_list, env={**os.environ, **gpu_cmd})
