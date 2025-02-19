import copy
import time
import os

from frankapy.franka_constants import FrankaConstants
from franka_interface_msgs.msg import SensorDataGroup

from manipgen.utils.dagger_utils import get_obs_shape_meta, get_rollout_action, make_video
from manipgen.real_world.utils.constants import FRANKA_LOWER_LIMITS, FRANKA_UPPER_LIMITS
from manipgen.real_world.utils.homography_utils import HomographyTransform
from manipgen.real_world.utils.realsense_camera import Rate, RealSenseCam
from manipgen.real_world.utils.rl_utils import compute_policy_observations, transform_quat
from manipgen.real_world.utils.vis_utils import display_raw_depth, crop_depth_to_policy_view, crop_rgb_to_policy_view
import manipgen.utils.robomimic_utils as RMUtils
from industreallib.tasks.classes.industreal_task_base import IndustRealTaskBase
import industreallib.control.scripts.control_utils as control_utils

from scipy.spatial.transform import Rotation
import time
import rospy
import torch
import numpy as np
import hydra
import open3d as o3d

class FrankaRealPSLEnv(IndustRealTaskBase):
    """Environment class for the Franka robot in the real world."""
    def __init__(self, args, cfg):
        super().__init__(args, cfg.task_instance_config, in_sequence=False)
        self.setup_sensors(sensors_cfg=cfg.sensors, hz=cfg.hz)
        self.cfg = cfg
        self._ros_rate_joint = rospy.Rate(60)

        self.log_dir = os.path.join(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."), 
            "manipgen/real_world/manipgen_logs"
        )
    
    def load_policy(self, task, checkpoint_path):
        """Load RL policies.

        Args:
            task (str): task name
            checkpoint_path (str): path to the checkpoint
        """

        if getattr(self, "policies", None) is None:
            self.policies = {}
        else:
            assert type(self.policies) is dict
        
        print(f"Loading an RL policy for {task}...")
        
        # setup policy
        self.cfg = hydra.compose(config_name=task)
        config = self.cfg.policy_config
        robomimic_cfg = RMUtils.load_config(
            algo_cfg_path=config.dagger.student_cfg_path,
            override_cfg=config,
        )
        RMUtils.initialize(robomimic_cfg)
        obs_shape_meta = get_obs_shape_meta(config)
        
        # build policy
        policy = RMUtils.build_model(
            config=robomimic_cfg,
            shape_meta=obs_shape_meta,
            device="cuda",
        )
        checkpoint = torch.load(checkpoint_path)
        policy.deserialize(checkpoint["student_state_dict"])
        policy.set_eval()

        # If RNN policy, reset RNN states
        policy.reset()
    
        self.policies[task] = policy
            
    def setup_sensors(self, sensors_cfg, hz):
        """Setup sensors."""
        self.sensors = []
        if sensors_cfg:
            self.sensors = [
                RealSenseCam(sensors_cfg[sensor].camera_cfg)
                for sensor in sensors_cfg if sensors_cfg[sensor].enabled
            ]

        self.rate = Rate(hz)
        
        self.hom = {}
        
        for cam_key in sensors_cfg:
            if cam_key != 'cam1' and sensors_cfg[cam_key].enabled:
                sensors_cfg[cam_key]["camera_cfg"]["img_width"] = 640
                sensors_cfg[cam_key]["camera_cfg"]["img_height"] = 480
                cam_idx = int(cam_key[-1])
                try:
                    self.hom[cam_key] = HomographyTransform(
                        f"img{cam_idx}",
                        transform_file="hom",
                        cam_cfg=sensors_cfg[cam_key]["camera_cfg"],
                    )
                except:
                    print(f"Failed to load homography for {cam_key}")
        
    def get_obs(self, wrist_cam_only=False, trans_quat=True):
        """Get observations from the environment."""
        observations = self.get_proprio_obs(trans_quat=trans_quat)\
        
        # Aggregate observations from the sensors
        for sensor in self.sensors:
            observations.update(sensor.get_obs())
            if wrist_cam_only:
                break

        return observations
    
    def get_proprio_obs(self, trans_quat=True):
        """Get proprioceptive observations."""
        franka_arm = self.franka_arm
        curr_state = franka_arm.get_robot_state()
        # For list of all keys in state dict, see
        # https://github.com/iamlab-cmu/frankapy/blob/master/frankapy/franka_arm_state_client.py

        curr_joint_angles = curr_state["joints"]
        curr_pos = curr_state["pose"].translation
        curr_ori_mat = curr_state["pose"].rotation
        curr_gripper_width = self.franka_arm.get_gripper_width()
        
        # proprioceptive observations
        observations = dict(
            q_pos=curr_joint_angles,
            q_vel=curr_state["joint_velocities"],
            eef_pos=curr_pos,
            eef_quat=Rotation.from_matrix(curr_ori_mat).as_quat(),  # (x, y, z, w)
            eef_mat=curr_ori_mat,
            gripper_width=curr_gripper_width,
        )
        if trans_quat:
            observations["eef_quat"] = transform_quat(observations["eef_quat"])
        return observations
    
    def _enforce_workspace_constraints(self, target_pos, target_ori_mat):
        """Enforces workspace constraints."""

        WORKSPACE_Z_MIN = self.cfg.workspace_config.workspace_z_min
        WORKSPACE_Z_MAX = self.cfg.workspace_config.workspace_z_max
        
        GRIPPER_WIDTH = self.cfg.workspace_config.gripper_width

        # get gripper tip position in eef frame
        tip_pos_eef_frame = np.array([
            [0.018,     GRIPPER_WIDTH / 2.0,   0.010],
            [0.018,    -GRIPPER_WIDTH / 2.0,   0.010],
            [-0.018,    GRIPPER_WIDTH / 2.0,   0.010],
            [-0.018,   -GRIPPER_WIDTH / 2.0,   0.010],
        ])

        # convert to base frame
        tip_pos_base_frame = target_pos + (target_ori_mat @ tip_pos_eef_frame.T).T

        # enforce workspace constraints
        target_pos[2] -= min(np.min(tip_pos_base_frame[:, 2] - WORKSPACE_Z_MIN), 0)
        target_pos[2] -= max(np.max(tip_pos_base_frame[:, 2] - WORKSPACE_Z_MAX), 0)

        return target_pos, target_ori_mat

    def _send_targets(self, actions, curr_pos, curr_ori_mat, print_info=False):
        """Sends pose targets to franka-interface via frankapy."""
        # NOTE: All actions are assumed to be in the form of [delta_position; delta_orientation],
        # where delta position is in the robot base frame, delta orientation is in the end-effector
        # frame, and delta orientation is an Euler vector (i.e., 3-element axis-angle
        # representation).

        if self.task_instance_config.control.mode.type == "nominal":
            targ_pos = curr_pos + actions[:3]
            targ_ori_mat = Rotation.from_rotvec(actions[3:6]).as_matrix() @ curr_ori_mat
            
            if self.cfg.workspace_config.enable_workspace_bounds:
                targ_pos, targ_ori_mat = self._enforce_workspace_constraints(targ_pos, targ_ori_mat)

        elif self.task_instance_config.control.mode.type in ["plai", "leaky_plai"]:
            if self._prev_targ_pos is None:
                self._prev_targ_pos = curr_pos.copy()
            if self._prev_targ_ori_mat is None:
                self._prev_targ_ori_mat = curr_ori_mat.copy()

            targ_pos = self._prev_targ_pos + actions[:3]
            targ_ori_mat = Rotation.from_rotvec(actions[3:6]).as_matrix() @ self._prev_targ_ori_mat

            if self.task_instance_config.control.mode.type == "leaky_plai":
                # leaky PLAI for position
                pos_err = targ_pos - curr_pos
                pos_err_clip = np.clip(
                    pos_err,
                    a_min=-np.asarray(
                        self.task_instance_config.control.mode.leaky_plai.pos_err_thresh
                    ),
                    a_max=np.asarray(
                        self.task_instance_config.control.mode.leaky_plai.pos_err_thresh
                    ),
                )
                targ_pos = curr_pos + pos_err_clip

                # leaky PLAI for rotation
                rot1 = Rotation.from_matrix(curr_ori_mat)
                rot2 = Rotation.from_matrix(targ_ori_mat)
                relative_rotation = rot2 * rot1.inv()
                angle_degrees = np.rad2deg(relative_rotation.magnitude())
                axis = relative_rotation.as_rotvec() / relative_rotation.magnitude()
                
                if angle_degrees > self.task_instance_config.control.mode.leaky_plai.rot_err_thresh:
                    clamped_angle_radians = np.deg2rad(self.task_instance_config.control.mode.leaky_plai.rot_err_thresh)
                    clamped_relative_rotation = Rotation.from_rotvec(clamped_angle_radians * axis)
                    clamped_rot2 = clamped_relative_rotation * rot1
                    targ_ori_mat = clamped_rot2.as_matrix()
                    
                if print_info:
                    print("rot", angle_degrees, axis, actions[3:6] / np.array(self.task_instance_config.control.mode.leaky_plai.action_scale[3:6]))
                    print("pos", pos_err, actions[:3] / np.array(self.task_instance_config.control.mode.leaky_plai.action_scale[:3]))
            
            if self.cfg.workspace_config.enable_workspace_bounds:
                targ_pos, targ_ori_mat = self._enforce_workspace_constraints(targ_pos, targ_ori_mat)
            
            self._prev_targ_pos = targ_pos.copy()
            self._prev_targ_ori_mat = targ_ori_mat.copy()

        else:
            raise ValueError("Invalid control mode.")

        ros_msg = control_utils.compose_ros_msg(
            targ_pos=targ_pos,
            targ_ori_quat=np.roll(
                Rotation.from_matrix(targ_ori_mat).as_quat(), shift=1
            ),  # (w, x, y, z)
            prop_gains=self.task_instance_config.control.prop_gains,
            msg_count=self._ros_msg_count,
        )

        self._ros_publisher.publish(ros_msg)
        self._ros_msg_count += 1
    
    def execute_local_policy(self, task="pick_cube", duration=None, local_seg_mask=None, debug=False):
        """Execute local policy
        
        Args:
            task (str): task name
            duration (float): duration of the execution
            local_seg_mask (np.ndarray): local segmentation mask
            debug (bool): debug flag
        """
        # reload cfg
        self.cfg = hydra.compose(config_name=task)
        self.task_instance_config = self.cfg.task_instance_config
        
        # reset policy
        self.policies[task].reset()
        
        self._start_target_stream(franka_arm=self.franka_arm)

        print(f"\nExecuting RL policy for {task}...")
        
        timing_stats = dict(obs=0.0, action=0.0)
        policy_start_time = rospy.get_time()
        num_steps = 0
        
        prev_obs = None
        frame0_obs = None
        
        depth_imgs = []
        rgb_imgs = []
        
        if duration is None:
            duration = self.task_instance_config.motion.duration

        while rospy.get_time() - policy_start_time < duration:
            num_steps += 1

            # compute observation
            tik = time.time()
            obs = self.get_obs(wrist_cam_only=True)
            if prev_obs is None:
                # first frame
                prev_obs = copy.deepcopy(obs)
                frame0_obs = copy.deepcopy(obs)
                if self.cfg.policy_config.dagger.use_seg_obs:
                    mask = local_seg_mask
                else:
                    mask = None
                    
            state_obs, visual_obs = compute_policy_observations(
                obs=obs,
                prev_obs=prev_obs,
                frame0_obs=frame0_obs,
                device=self._device,
                seg_mask=mask,
                cfg=self.cfg.perception_config,
            )
            timing_stats["obs"] = time.time() - tik

            # process inputs for the policy
            tik = time.time()
            actions = get_rollout_action(
                model=self.policies[task],
                state_obs=state_obs, 
                visual_obs=visual_obs,
            )[0].detach().cpu().numpy() 
            
            actions = actions * np.array(
                self.task_instance_config.control.mode[
                    self.task_instance_config.control.mode.type
                ].action_scale,
                dtype=np.float32,
            )
            timing_stats["action"] = time.time() - tik
            
            curr_state = self.franka_arm.get_robot_state()

            self._send_targets(
                actions=actions,
                curr_pos=curr_state["pose"].translation,
                curr_ori_mat=curr_state["pose"].rotation,
            )
            
            prev_obs = copy.deepcopy(obs)
        
            if debug:
                depth_imgs.append(obs["cam1_depth"][0].copy())
                rgb_imgs.append(obs["cam1"][0].copy())

            self._ros_rate.sleep()
            
        self.franka_arm.stop_skill()
        self._prev_targ_pos, self._prev_targ_ori_mat = None, None
        if task == 'pick' or task == 'grasp_handle':
            self.franka_arm.close_gripper(block=(task == "grasp_handle"))
        elif task == 'place' or task == 'close' or task == 'open':
            self.franka_arm.open_gripper(block=False)
        if debug:
            # save depth images as video to logs/depth_{date}.mp4
            for i in range(len(depth_imgs)):
                depth = crop_depth_to_policy_view(depth_imgs[i].copy(), self.cfg.perception_config)
                depth_imgs[i] = display_raw_depth(depth, display=False)
                
                rgb = crop_rgb_to_policy_view(rgb_imgs[i].copy(), self.cfg.perception_config)
                rgb_imgs[i] = rgb
            time_stamp = time.strftime('%Y%m%d-%H%M%S')
            make_video(depth_imgs, self.log_dir, 0, name=f"{task}_depth_inputs_{time_stamp}.mp4")
            make_video(rgb_imgs, self.log_dir, 0, name=f"{task}_rgb_inputs_{time_stamp}.mp4")

    def open_gripper(self):
        """Open gripper."""
        self.do_simple_procedure(procedure=['open_gripper'], franka_arm=self.franka_arm)
        
    def execute_frankapy(self, target_eef_pose, duration=5, block=True, use_libfranka_controller=False):
        """
        Move to target pose using frankapy controller.
        Args:
            target_eef_pose (np.ndarray): target pose of end effector
        """
        goal_pos = target_eef_pose[:3]
        goal_ori_mat = Rotation.from_quat(target_eef_pose[3:]).as_matrix()  # intrinsic rotations

        # First use frankapy controller (better IK) to go to pose,
        # then use libfranka controller (better accuracy) to go to same pose
        control_utils.go_to_pose(
            franka_arm=self.franka_arm, pos=goal_pos, ori_mat=goal_ori_mat, duration=duration, use_impedance=True, block=block
        )
        if use_libfranka_controller:
            control_utils.go_to_pose(
                franka_arm=self.franka_arm, pos=goal_pos, ori_mat=goal_ori_mat, duration=duration, use_impedance=False, block=block
            )
        # print error:
        achieved_pos = self.get_obs()["eef_pos"]
        achieved_ori_mat = self.get_obs()["eef_mat"]
        pos_error = np.linalg.norm(achieved_pos - goal_pos)
        achieved_ori_mat = Rotation.from_matrix(achieved_ori_mat)
        goal_ori_mat = Rotation.from_matrix(goal_ori_mat)
        ori_error = np.linalg.norm((achieved_ori_mat.inv() * goal_ori_mat).as_rotvec()) * 180 / np.pi
        print(f"Execute frankapy: pos_error: {pos_error:.6f}" + f" ori_error: {ori_error:.6f}")
    
    def spin_end_effector_180(self):
        """ Util function for global camera calibration. """
        curr_joint_angles = self.franka_arm.get_joints()
        targ_joint_angles = curr_joint_angles.copy()
        targ_joint_angles[-1] = curr_joint_angles[-1] - np.pi
        # project back into -np.pi to np.pi range
        targ_joint_angles[-1] = (targ_joint_angles[-1] + np.pi) % (2 * np.pi) - np.pi

        print("\nPerturbing yaw...")
        self.franka_arm.goto_joints(joints=targ_joint_angles, use_impedance=False, ignore_virtual_walls=True)
        print("Finished perturbing yaw.")

    def reset(self):
        """
        Reset environment.

        Returns:
            observation (dict): initial observation dictionary.
        """
        self.do_simple_procedure(procedure=["open_gripper", "go_home"], franka_arm=self.franka_arm)
        return self.get_obs()

    def get_fixed_depth(self, cam_name="cam3"):
        """
        Get the depth image from the environment
        """
        return self.get_obs()[f"{cam_name}_depth"][0]

    def get_fixed_rgb(self, cam_name="cam3"):
        """
        Get the rgb image from the environment
        """
        return self.get_obs()[f"{cam_name}"][0]
    
    def visualize_pcd(self, points, colors):
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            o3d_pcd.colors = o3d.utility.Vector3dVector(colors[:, ::-1])
        o3d.visualization.draw_geometries([o3d_pcd])
        
    
    def get_raw_pcd_single_cam(self, cam_idx=1, filter_pcd=True, denoise=False, debug=False):
        """
        Get the raw point cloud from a single camera.

        Args:
            cam_idx (int): Camera index.
            filter (bool): Whether to filter the point cloud.
            denoise (bool): Whether to denoise the point cloud.
            debug (bool): Whether to save and visualize the point cloud for debugging.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Points and colors of the point cloud.
        """
        img_numpy = self.get_fixed_rgb(f"cam{cam_idx}")
        depth_numpy = self.get_fixed_depth(f"cam{cam_idx}")

        raw_pc = self.hom[f"cam{cam_idx}"].get_pointcloud(depth_numpy).reshape(-1, 3)
        raw_img = img_numpy[:, :, ::-1].reshape(-1, 3) / 255.0

        if debug:
            self.visualize_pcd(raw_pc, raw_img)
        if filter_pcd:
            points, colors = self.hom[f"cam{cam_idx}"].get_filtered_pc(
                raw_pc.reshape(-1, 3), raw_img.reshape(-1, 3)
            )
            if denoise:
                points, colors = self.hom[f"cam{cam_idx}"].denoise_pc(points, colors)
            if debug:
                self.visualize_pcd(points, colors)
            return points, colors

        return raw_pc, raw_img

    def get_multi_cam_pcd(
        self,
        debug_raw_pcd=False,
        debug_combined_pcd=False,
        filter_pcd=True,
        denoise=False,
    ):
        """
        Get the combined point cloud from multiple cameras. Processing depth and RGB at the same time. RGB info gives better visualization for debugging supports but will slow down the process.

        Args:
            debug_raw_pcd (bool): Whether to debug the raw point cloud.
            debug_combined_pcd (bool): Whether to debug the combined point cloud.
            save_pcd (bool): Whether to save the combined point cloud.
            save_file_name (str): File name for saving the combined point cloud.
            filter (bool): Whether to filter the point cloud.
            denoise (bool): Whether to denoise the point cloud.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Points and colors of the combined point cloud.
        """
        combined_pcd = None
        combined_rgb = None
        for cam_key in self.hom.keys():
            cam_idx = int(cam_key[-1])
            pcd, rgb = self.get_raw_pcd_single_cam(
                cam_idx=cam_idx,
                filter_pcd=filter_pcd,
                denoise=denoise,
                debug=debug_raw_pcd,
            )
            if combined_pcd is None:
                combined_pcd = pcd
                combined_rgb = rgb
            else:
                combined_pcd = np.concatenate((combined_pcd, pcd), axis=0)
                combined_rgb = np.concatenate((combined_rgb, rgb), axis=0)
        if debug_combined_pcd:
            self.visualize_pcd(combined_pcd, combined_rgb)
        return combined_pcd, combined_rgb

    def get_multi_cam_pcd_fast(
        self, debug=False, downsample=50000,
    ):
        """
        Get the combined point cloud from multiple cameras quickly by only processing depth information.

        Args:
            debug_raw_pcd (bool): Whether to debug the raw point cloud.
            debug_combined_pcd (bool): Whether to debug the combined point cloud.
            save_pcd (bool): Whether to save the combined point cloud.
            filter (bool): Whether to filter the point cloud.
            denoise (bool): Whether to denoise the point cloud.
            downsample (int): Number of points to downsample.

        Returns:
            np.ndarray: Masked point cloud.
        """
        pcds = []
        for cam_key in self.hom.keys():
            depth_numpy = self.get_fixed_depth(f"{cam_key}")
            raw_pcd = self.hom[cam_key].get_pointcloud(depth_numpy).reshape(-1, 3)
            pcds.append(raw_pcd)
        raw_combined_pcd = np.concatenate(pcds, axis=0)
        
        random_obstacle_indices = np.random.choice(
            len(raw_combined_pcd), size=downsample, replace=False
        )
        downsampled_pcd = raw_combined_pcd[random_obstacle_indices]

        filtered_pcd = self.hom[list(self.hom.keys())[0]].get_filtered_pc(downsampled_pcd)
        if debug:
            self.visualize_pcd(filtered_pcd, None)
        return filtered_pcd, None
    
    def _send_targets_joint(self, target_joint_angles):
        """Sends pose targets to franka-interface via frankapy."""
    
        ros_msg = control_utils.compose_ros_msg_joints(
            targ_joint_angles = target_joint_angles,
            msg_count=self._ros_msg_count,
        )

        self._ros_publisher.publish(ros_msg)
        self._ros_msg_count += 1
    
    def _start_target_stream_joints(self, franka_arm, joint_angles):
        """Starts streaming targets to franka-interface via frankapy."""
        self._ros_publisher = rospy.Publisher(
            FrankaConstants.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000
        )

        # Initiate streaming with dummy command to go to current pose
        # NOTE: Closely adapted from
        # https://github.com/iamlab-cmu/frankapy/blob/master/examples/run_dynamic_pose.py
        franka_arm.goto_joints(
            joint_angles,
            duration=5,
            dynamic=True,
            ignore_virtual_walls=True,
            buffer_time=10,
        )
    
    def execute_joint_action(
        self,
        action_target: np.ndarray,
        speed: float = 0.1,
    ):
        """
        Execute joint control action on the robot. Be careful with the manually set start_angles.

        Args:
            action_target (np.ndarray): Target joint angles.
            start_angles (np.ndarray, optional): Start joint angles.
            speed (float, optional): Speed for execution (rad/s).
            set_intermediate_states (bool, optional): Whether to set intermediate states.
        """
        ctrl_hz = 60
        obs = self.get_proprio_obs()
        start_angles = obs['q_pos']

        max_action = max(abs(action_target - start_angles))
        num_steps = max(int(max_action / speed * ctrl_hz), 10)

        for step in range(num_steps):
            action_processed = (
                start_angles + (action_target - start_angles) * (step + 1) / num_steps
            )
            action_processed = np.clip(
                action_processed, FRANKA_LOWER_LIMITS, FRANKA_UPPER_LIMITS
            )
            self._send_targets_joint(target_joint_angles=action_processed)
            self._ros_rate_joint.sleep()

    def close(self):	
        print("Closing...")	
        for sensor in self.sensors:	
            sensor.close()	
        time.sleep(1)	
        print("Environment Closed")
