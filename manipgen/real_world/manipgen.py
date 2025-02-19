import time
import numpy as np
from manipgen.real_world.industreal_psl_env import FrankaRealPSLEnv
from manipgen.real_world.neural_mp import NeuralMP
from manipgen.real_world.vlm_planner import VLMPlanner

from manipgen.real_world.vlm_planning.target_pose_utils import backtrack_eef_position

def retract_eef(env, offset=-0.06, duration=5.0, block=True, use_libfranka_controller=True):
    """Retracts the end-effector backward by a specified offset.
    
    Args:
        env (FrankaRealPSLEnv): The environment object
        offset (float, optional): The offset to pull back the end-effector. Defaults to -0.06.
        duration (float, optional): The duration for which to execute the retraction. Defaults to 5.0.
        block (bool, optional): If True, the function will block until the retraction is completed. Defaults to True.
        use_libfranka_controller (bool, optional): If True, use the libfranka controller to execute the retraction. Defaults to True.
    """
    curr_pos = env.get_obs(trans_quat=False)['eef_pos']
    curr_quat = env.get_obs(trans_quat=False)['eef_quat']
    target_pos = backtrack_eef_position(curr_pos, curr_quat, offset)
    eef_target = np.concatenate([target_pos, curr_quat])
    env.execute_frankapy(eef_target, duration=duration, block=block, use_libfranka_controller=use_libfranka_controller)

class ManipGen:
    def __init__(self, env:FrankaRealPSLEnv, vlm_planner: VLMPlanner, motion_planner: NeuralMP):
        """Initializes the ManipGen class with the environment and planners

        Args:
            vlm_planner (VLMPlanner): A planner that generates a plan of the form (object, skill)
            motion_planner (MotionPlanner): A planner that generates a plan to move to a target pose
        """
        self.env = env
        self.vlm_planner = vlm_planner
        self.motion_planner = motion_planner
        
    def solve_task(self, text_prompt, debug=False):
        """Solves a task given a text prompt"""

        self.env.franka_arm.open_gripper(block=False)
        # move robot to home position with motion planner
        self.motion_planner.motion_plan_to_target_joint_angles_and_execute(
            np.array([0.0, -1.01627597, 0.0, -2.21283626, 0.0, 1.20934282, np.pi / 4]),
        )
        self.env.reset()
        
        start_time = time.time()

        # The skills planner uses the VLM planner to generate a plan
        vlm_plan, tags = self.vlm_planner.plan(text_prompt)
        after_plan_start_time = time.time()
        print("VLM Plan: ", vlm_plan)
                                
        skill_attempts = 1

        # The vlm plan is of the form (object, skill, is_tabletop). 
        # `is_table_top` is a boolean that indicates if there is obstacle above the object.
        # This changes the way we sample target orientations for reaching the object.
        # We motion plan to each object by estimating its position, performing IK, and executing the motion
        while vlm_plan:
            obj, skill, is_table_top = vlm_plan.pop(0)
            skill_completed = False
            for attempt_id in range(skill_attempts):
                # Step 1: Target Pose Estimation
                target_pose, pcd = self.vlm_planner.estimate_target_pose(obj, skill, is_table_top, visualize=False, debug=debug)
                print(f"Object: {obj}, Skill: {skill}, Attempt: {attempt_id + 1} / {skill_attempts}")
                print(f"Target Pos: {target_pose[:3]}, Target Quat: {target_pose[3:]}")
                
                # Step 2: Motion Planning
                motion_plan = self.motion_planner.motion_plan_to_target_ee_pose(target_pose, pcd=pcd, batch_size=64, in_hand=True)
                self.motion_planner.execute_motion_plan(motion_plan)
                
                # if Neural MP fails to reach the target, run the default end-effector controller to move the robot to the target pose
                mp_target_eef_pose = self.vlm_planner.add_umi_finger_length_in_target_pose(target_pose)
                curr_pos = self.env.get_obs(trans_quat=False)['eef_pos']
                target_pos = mp_target_eef_pose[:3]
                dist = np.linalg.norm(curr_pos - target_pos)
                if dist > 0.30:
                    raise Exception("Neural MP failed to reach the target pose. Please check if the motion planner is working correctly.")
                elif dist > 0.06:
                    self.env.execute_frankapy(mp_target_eef_pose, duration=3.0, block=True, use_libfranka_controller=True)
                
                # Step 3: Local Policies
                print("Executing Skill: ", skill)
                if skill in ("pick", "place"):
                    # get local segmentation mask (only for pick)
                    local_seg_mask = self.vlm_planner.get_local_seg_mask(obj, skill=skill) if skill == "pick" else None
                    self.env.execute_local_policy(task=skill, local_seg_mask=local_seg_mask, debug=debug)
                    retract_eef(self.env, offset=0.10, duration=1.0, block=False, use_libfranka_controller=False)

                elif skill in ("open", "close"):
                    # execute grasp handle policy before running open/close
                    self.env.execute_local_policy(task="grasp_handle", debug=debug)
                    self.env.execute_local_policy(task=skill, debug=debug)
                    retract_eef(self.env, offset=0.10, duration=1.0, block=False, use_libfranka_controller=False)
                                
                # TODO: Check if skill is completed   
                skill_completed = True
                if skill_completed:
                    break

            self.update_execution_log(skill, obj)

        # print time logs
        if debug:
            print("Total Execution Time: ", time.time() - start_time)
            print("Time after planning: ", time.time() - after_plan_start_time)
            
        return True
