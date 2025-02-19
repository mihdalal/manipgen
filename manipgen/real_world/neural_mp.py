from collections import OrderedDict
import json
import time
from typing import Tuple
from frankapy.franka_interface_common_definitions import SensorDataMessageType
from frankapy.proto_utils import make_sensor_group_msg, sensor_proto2ros_msg
from frankapy.utils import min_jerk
import numpy as np
import torch
import urchin
from robofin.pointcloud.torch import FrankaSampler
from robofin.robots import FrankaRobot
import manipgen_robomimic.utils.file_utils as FileUtils
import manipgen_robomimic.utils.torch_utils as TorchUtils

import rospy
from manipgen.real_world.utils.geometry import TorchCuboids, TorchCylinders, TorchSpheres, vectorized_subsample
from manipgen.real_world.utils.neural_mp_env_wrapper import IndustrealEnvWrapper
from manipgen.real_world.utils.real_world_collision_checker import FrankaCollisionChecker
import meshcat
from scipy.spatial.transform import Rotation as R_scipy
from frankapy import FrankaConstants as FC
from frankapy.proto import JointPositionSensorMessage, ShouldTerminateSensorMessage
from franka_interface_msgs.msg import SensorDataGroup

class NeuralMP:
    def __init__(self, cfg, env:IndustrealEnvWrapper, model_path, train_mode, tto, in_hand,
                 max_neural_mp_rollout_length=100, num_robot_points=2048, clamp_joint_limit=False, num_obstacle_points=4096, 
                 visualize=False):
        import torch
        import torch._dynamo

        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("medium")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch._dynamo.config.suppress_errors = True
        
        self.cfg = cfg.neural_mp_config
        self.device = TorchUtils.get_torch_device(try_to_use_cuda=True)
        policy, _ = FileUtils.policy_from_checkpoint(
            ckpt_path=model_path, device=self.device, verbose=False
        )
        self.policy = policy
        self.train_mode = train_mode
        self.tto = tto
        self.in_hand = in_hand
        # TODO: double dim 1-3 when grasping very large objects
        # Refer to Neural MP (https://github.com/mihdalal/neuralmotionplanner) for more details
        self.in_hand_params = np.array(self.cfg.in_hand_params)
        self.max_neural_mp_rollout_length = max_neural_mp_rollout_length
        self.num_robot_points = num_robot_points
        self.num_obstacle_points = num_obstacle_points
        self.gpu_fk_sampler = FrankaSampler("cuda", use_cache=False)
        self.collision_checker = FrankaCollisionChecker()
        self.env = env
        self.clamp_joint_limit = clamp_joint_limit
        self.visualize = visualize
        if self.visualize:
            self.viz = meshcat.Visualizer()
            # Load the FK module
            self.urdf = urchin.URDF.load(FrankaRobot.urdf)
            # Preload the robot meshes in meshcat at a neutral position
            for idx, (k, v) in enumerate(self.urdf.visual_trimesh_fk(np.zeros(8)).items()):
                self.viz[f"robot/{idx}"].set_object(
                    meshcat.geometry.TriangularMeshGeometry(k.vertices, k.faces),
                    meshcat.geometry.MeshLambertMaterial(wireframe=False),
                )
                self.viz[f"robot/{idx}"].set_transform(v)
        print("Neural MP initialized")

    
    def exclude_robot_pcd(self, points, colors=None, thred=0.03):
        """
        Exclude points belonging to the robot from the point cloud.

        Args:
            points (np.ndarray): Point cloud points.
            colors (np.ndarray, optional): Point cloud colors.
            thred (float): Threshold for excluding points.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Filtered points and colors.
        """
        config = self.env.get_joint_angles()
        centers, radii = self.collision_checker.spheres_cr(config)
        points = np.expand_dims(points, axis=2)

        centers = np.repeat(centers, points.shape[0], axis=0)
        sdf = np.linalg.norm((points - centers), axis=1) - radii
        is_robot_pcd = np.sum(sdf < thred, axis=1)
        scene_pcd_idx = np.where(is_robot_pcd == 0)[0]
        if colors is None:
            return points[scene_pcd_idx, :, 0], None
        else:
            return points[scene_pcd_idx, :, 0], colors[scene_pcd_idx, :]
    
    def fk_batched(self, configs: torch.Tensor):
        """
        Perform forward kinematics on batched configurations.

        Args:
            configs (torch.Tensor): Batched joint configurations.

        Returns:
            torch.Tensor: Transformations for the end effector.
        """
        ret = self.env.chain_gpu.forward_kinematics(configs.float(), end_only=True)
        m = ret.get_matrix()
        return m
        
    def transform_in_hand_obj_batched(self, joint_angles, trans_in_hand2eef, fk=None):
        """
        Transform an in-hand object given batched joint angles.

        Args:
            joint_angles (np.ndarray): Batched joint angles.
            trans_in_hand2eef (torch.Tensor): Transformation from in-hand object to end effector.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Global positions, orientations, and transformations.
        """

        def rotation_matrix_to_quaternion(matrix):
            m = matrix
            qw = torch.sqrt(1 + m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]) / 2
            qx = (m[:, 2, 1] - m[:, 1, 2]) / (4 * qw)
            qy = (m[:, 0, 2] - m[:, 2, 0]) / (4 * qw)
            qz = (m[:, 1, 0] - m[:, 0, 1]) / (4 * qw)
            return torch.cat(
                [qw.unsqueeze(1), qx.unsqueeze(1), qy.unsqueeze(1), qz.unsqueeze(1)], dim=1
            )

        if fk is None:
            panda_hand = self.fk_batched(joint_angles)  # Bx4x4
        else:
            panda_hand = fk

        final = panda_hand @ trans_in_hand2eef
        self.final_trans = final.cpu().numpy()
        global_pos = final[:, :3, 3]
        global_ori = rotation_matrix_to_quaternion(final[:, :3, :3])
        final = final
        return global_pos, global_ori, final
    
    def compute_in_hand_pcd(self, joint_angles, num_points, in_hand_params, fk=None):
        """
        Compute the in-hand point cloud.

        Args:
            joint_angles (np.ndarray): Joint angles.
            num_points (int): Number of points in the point cloud.
            in_hand_params (np.ndarray): In-hand object parameters (type, size, position, orientation).

        Returns:
            torch.Tensor: In-hand point cloud.
        """
        in_hand_type = ["box", "cylinder", "sphere"][int(in_hand_params[0])]
        in_hand_size = in_hand_params[1:4]
        in_hand_pos = in_hand_params[4:7]
        in_hand_ori = in_hand_params[7:11]

        r = R_scipy.from_quat(in_hand_ori)
        in_hand_rotation_matrix = r.as_matrix()

        trans_in_hand2eef = torch.eye(4, device=torch.device("cuda"))
        trans_in_hand2eef[:3, 3] = torch.from_numpy(in_hand_pos).cuda()
        trans_in_hand2eef[:3, :3] = torch.from_numpy(in_hand_rotation_matrix).cuda()

        joint_angles_batched = torch.from_numpy(joint_angles).cuda()  # Bx7
        global_pos, global_ori, _ = self.transform_in_hand_obj_batched(
            joint_angles_batched, trans_in_hand2eef, fk=fk
        )

        if in_hand_type == "box":
            cuboid_centers = global_pos.float()
            cuboid_quaternions = global_ori.float()
            cuboid_dims = (
                torch.from_numpy(in_hand_size)
                .cuda()
                .unsqueeze(0)
                .repeat(joint_angles.shape[0], 1)
                .float()
            )
            in_hand_obj = TorchCuboids(
                cuboid_centers.unsqueeze(1),
                cuboid_dims.unsqueeze(1),
                cuboid_quaternions.unsqueeze(1),
            )
        elif in_hand_type == "cylinder":
            cylinder_centers = global_pos.float()
            cylinder_quaternions = global_ori.float()
            cylinder_dims = (
                torch.from_numpy(in_hand_size)
                .cuda()
                .unsqueeze(0)
                .repeat(joint_angles.shape[0], 1)
                .float()
            )
            in_hand_obj = TorchCylinders(
                cylinder_centers.unsqueeze(1), cylinder_dims, cylinder_quaternions.unsqueeze(1)
            )
        elif in_hand_type == "sphere":
            sphere_centers = global_pos.float()
            sphere_dims = (
                torch.from_numpy(in_hand_size)
                .cuda()
                .unsqueeze(0)
                .repeat(joint_angles.shape[0], 1)
                .float()
            )
            in_hand_obj = TorchSpheres(sphere_centers.unsqueeze(1), sphere_dims)
        in_hand_pcd = in_hand_obj.sample_surface(num_points)[:, 0]
        return in_hand_pcd

    def add_in_hand_pcd(
        self,
        pcd: torch.Tensor,
        joint_angles: np.ndarray,
        in_hand_params: np.ndarray,
        num_in_hand_points=500,
        fk = None
    ):
        """
        Add the in-hand point cloud to the pre-computed point cloud.

        Args:
            pcd (torch.Tensor): Pre-computed point cloud.
            joint_angles (np.ndarray): Joint angles.
            in_hand_params (np.ndarray): In-hand object parameters (type, size, position, orientation).
            num_in_hand_points (int): Number of in-hand points.

        Returns:
            torch.Tensor: Combined point cloud.
        """
        pcd_size = pcd.shape[1]
        in_hand_pcd = self.compute_in_hand_pcd(joint_angles, num_in_hand_points, in_hand_params, fk=fk)
        combined_pcd = torch.cat((pcd, in_hand_pcd), dim=1)
        final_pcd = vectorized_subsample(combined_pcd, dim=1, num_points=pcd_size)
        return final_pcd
    
    def make_point_cloud_from_problem(
        self,
        start_config,
        goal_config,
        obstacle_points: np.ndarray,
        obstacle_colors: np.ndarray,
        in_hand: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate the point cloud from the eval task specification.

        Args:
            obstacle_points (np.ndarray): xyz point coordinates of scene obstacles.
            obstacle_colors (np.ndarray): Colors corresponding to obstacle points.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: xyz and rgb information of the combined point cloud that will be passed into the network.
        """
        gripper_width = self.env.get_gripper_width() / 2
        start_tensor_config = torch.from_numpy(np.concatenate([start_config, [gripper_width]])).to(self.device)
        goal_tensor_config = torch.from_numpy(np.concatenate([goal_config, [gripper_width]])).to(self.device)
        robot_points = self.gpu_fk_sampler.sample(
            start_tensor_config, self.num_robot_points
        )
        target_points = self.gpu_fk_sampler.sample(
            goal_tensor_config, self.num_robot_points
        )

        if in_hand:
            robot_points = self.add_in_hand_pcd(
                robot_points, start_config[np.newaxis, :], self.in_hand_params
            )
            target_points = self.add_in_hand_pcd(
                target_points, goal_config[np.newaxis, :], self.in_hand_params
            )

        xyz = torch.cat(
            (
                torch.zeros(self.num_robot_points, 4, device=self.device),
                torch.ones(self.num_obstacle_points, 4, device=self.device),
                2 * torch.ones(self.num_robot_points, 4, device=self.device),
            ),
            dim=0,
        )
        xyz[:self.num_robot_points, :3] = robot_points.float()
        random_obstacle_indices = np.random.choice(
            len(obstacle_points), size=self.num_obstacle_points, replace=False
        )
        xyz[
            self.num_robot_points : self.num_robot_points + self.num_obstacle_points,
            :3,
        ] = torch.as_tensor(obstacle_points[random_obstacle_indices, :3], device=self.device).float()
        xyz[
            self.num_robot_points + self.num_obstacle_points :,
            :3,
        ] = target_points.float()

        def sampler(config, gripper_width=gripper_width):
            gripper_cfg = gripper_width * torch.ones((config.shape[0], 1), device=config.device)
            cfg = torch.cat((config, gripper_cfg), dim=1)
            sampled_pcd, fk = self.gpu_fk_sampler.sample(cfg, self.num_robot_points, get_fk=True)
            fk = list(fk.values())[-3]
            if in_hand:
                return self.add_in_hand_pcd(
                    sampled_pcd, config.cpu().numpy(), self.in_hand_params, fk=fk
                )
            else:
                return sampled_pcd

        if self.visualize:
            len_obs = len(obstacle_colors)
            point_cloud_colors = np.zeros((3, len_obs + self.num_robot_points))
            point_cloud_colors[:, :len_obs] = obstacle_colors.T
            point_cloud_colors[0, len_obs:] = 1
            point_cloud_points = np.zeros((3, len_obs + self.num_robot_points))
            point_cloud_points[:, :len_obs] = obstacle_points.T
            point_cloud_points[:, len_obs:] = target_points[0].cpu().numpy().T
            self.viz["point_cloud"].set_object(
                # Don't visualize robot points
                meshcat.geometry.PointCloud(
                    position=point_cloud_points,
                    color=point_cloud_colors[::-1, :],
                    size=0.005,
                )
            )

            if in_hand:
                robot_pcd = (
                    sampler(torch.Tensor(start_config).cuda().unsqueeze(0)).cpu().numpy()[0]
                )
                robot_rgb = np.zeros((3, self.num_robot_points))
                robot_rgb[1, :] = 1
                self.viz["robot_point_cloud"].set_object(
                    # Don't visualize robot points
                    meshcat.geometry.PointCloud(
                        position=robot_pcd.T,
                        color=robot_rgb,
                        size=0.005,
                    )
                )
        if obstacle_colors is not None:
            obstacle_colors = obstacle_colors[random_obstacle_indices, :]
        return xyz, obstacle_colors

    @torch.inference_mode()
    def motion_plan_with_tto(self, start_config, goal_config, points, colors=None, batch_size=10, in_hand=False, max_neural_mp_rollout_length=100):
        """
        Motion plan by rolling out the policy with batched samples and perform test time optimiszation
        to select the safest path to execute on the robot.

        Args:
            points (np.ndarray): xyz information of the point cloud.
            colors (np.ndarray): rgb information of the point cloud for visualization.
            batch_size (int): size of the batch.

        Returns:
            Tuple[list, bool, float]: output trajectory, planning success flag, and average rollout time.
        """
        goal_pose = FrankaRobot.fk(goal_config, eff_frame="right_gripper")
        pset_points, colors = self.make_point_cloud_from_problem(start_config, goal_config, points, colors, in_hand=in_hand)
        point_cloud = pset_points.unsqueeze(0).repeat(batch_size, 1, 1)

        self.policy.start_episode()
        if self.train_mode:
            self.policy.policy.set_train()
        else:
            self.policy.policy.set_eval()

        q = torch.as_tensor(start_config, device=self.device).unsqueeze(0).float().repeat(batch_size, 1)
        g = torch.as_tensor(goal_config, device=self.device).unsqueeze(0).float().repeat(batch_size, 1)
        assert q.ndim == 2

        trajectory = []
        qt = q

        gripper_width = self.env.get_gripper_width() / 2
        
        obs = OrderedDict()
        obs["current_angles"] = q
        obs["goal_angles"] = g
        obs["compute_pcd_params"] = point_cloud
        
        # limit max_rollout_len up to 100, so gpu memory does not explode
        max_rollout_len = min(max_neural_mp_rollout_length, 100)
        ones_arr = torch.ones((q.shape[0], 1), device=q.device)
        gripper_cfg = gripper_width * ones_arr
        def sampler(config, gripper_cfg=None):
            if gripper_cfg is None:
                local_gripper_cfg = torch.ones((config.shape[0], 1), device=q.device) * gripper_width
            else:
                local_gripper_cfg = gripper_cfg
            cfg = torch.cat((config, local_gripper_cfg), dim=1)
            sampled_pcd, fk = self.gpu_fk_sampler.sample(cfg, self.num_robot_points, get_fk=True)
            fk = list(fk.values())[-3]
            if in_hand:
                return self.add_in_hand_pcd(
                    sampled_pcd, config.cpu().numpy(), self.in_hand_params, fk=fk
                )
            else:
                return sampled_pcd
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            for i in range(max_rollout_len):
                qt = qt + self.policy.policy.get_action(obs_dict=obs)
                # clamp qt to joint limits
                if self.clamp_joint_limit:
                    qt = torch.clamp(qt, self.env.joint_limits[0], self.env.joint_limits[1])
                trajectory.append(qt)
                samples = sampler(qt, gripper_cfg=gripper_cfg).type_as(point_cloud)
                point_cloud[:, : samples.shape[1], :3] = samples
                obs["current_angles"] = qt
                obs["compute_pcd_params"] = point_cloud

        output_traj = torch.stack(trajectory).permute(1, 0, 2)  # [batch_size, max_rollout_len, 7]
        goal_angles = g
        print("[Neural MP] minimal joint error:", torch.norm(output_traj[:, -1] - goal_angles, dim=1).min().item())
        num_valid_traj = output_traj.shape[0]
        output_traj = output_traj.reshape(-1, 7)
        if num_valid_traj == 0:
            return None

        scene_pcd = point_cloud[:, self.num_robot_points : self.num_robot_points + self.num_obstacle_points, :3].repeat(max_rollout_len, 1, 1)
        waypoint_c_num = self.collision_checker.check_scene_collision_batch(
            output_traj, scene_pcd, thred=0.03, sphere_repr_only=(not in_hand)
        )
        traj_c_num = torch.sum(waypoint_c_num.reshape(num_valid_traj, max_rollout_len), dim=1)
        best_traj_idx = torch.argmin(traj_c_num)
        output_traj = (
            output_traj.reshape(num_valid_traj, max_rollout_len, -1)[best_traj_idx]
            .detach()
            .cpu()
            .numpy()
        )
        
        # check whether goal is reached
        for i in range(len(output_traj)):
            eff_pose = FrankaRobot.fk(output_traj[i], eff_frame="right_gripper")
            pos_err = np.linalg.norm(eff_pose._xyz - goal_pose._xyz)
            ori_err = np.abs(
                np.degrees((eff_pose.so3._quat * goal_pose.so3._quat.conjugate).radians)
            )

            if (
                np.linalg.norm(eff_pose._xyz - goal_pose._xyz) < 0.01
                and np.abs(
                    np.degrees((eff_pose.so3._quat * goal_pose.so3._quat.conjugate).radians)
                )
                < 5
            ):
                planning_success = True
                output_traj = output_traj[: (i + 1)]
                break
            
        output_traj = np.concatenate((start_config.reshape(1, 7), output_traj), axis=0)
        
        self.visualize_traj(output_traj, gripper_width, in_hand, sampler)

        if output_traj.shape[0] > 3:
            # lets do EMA smoothing
            alpha = 0.9
            smoothed_traj = np.zeros_like(output_traj)
            smoothed_traj[0] = output_traj[0]
            for i in range(1, output_traj.shape[0]):
                smoothed_traj[i] = alpha * smoothed_traj[i-1] + (1-alpha) * output_traj[i]
            output_traj = smoothed_traj
            # add goal config to the end
            output_traj = np.concatenate((output_traj, goal_config.reshape(1, 7)), axis=0)

        self.visualize_traj(output_traj, gripper_width, in_hand, sampler)
                        
        return output_traj
    
    def visualize_traj(self, output_traj, gripper_width, in_hand, sampler):
        if self.visualize:
            trajc = output_traj.copy()
            while True:
                visual = input("simulate trajectory? (y/n): ")
                if visual == "n":
                    break
                elif visual == "y":
                    print("simlating")
                    for idx_traj in range(len(trajc)):
                        sim_config = np.append(trajc[idx_traj], gripper_width)
                        for idx, (k, v) in enumerate(
                            self.urdf.visual_trimesh_fk(sim_config[:8]).items()
                        ):
                            self.viz[f"robot/{idx}"].set_transform(v)
                        if in_hand:
                            # visualize robot pcd as well
                            robot_pcd = (
                                sampler(torch.Tensor(trajc[idx_traj]).cuda().unsqueeze(0))
                                .cpu()
                                .numpy()[0]
                            )
                            robot_rgb = np.zeros((3, self.num_robot_points))
                            robot_rgb[1, :] = 1
                            self.viz["robot_point_cloud"].set_object(
                                # Don't visualize robot points
                                meshcat.geometry.PointCloud(
                                    position=robot_pcd.T,
                                    color=robot_rgb,
                                    size=0.005,
                                )
                            )
                        time.sleep(0.05)
        
    def motion_plan_to_target_ee_pose(self, target_ee_pose, pcd=None, batch_size=10, in_hand: bool=False):
        start_config = self.env.get_joint_angles()
        goal_config = self.env.compute_ik(target_ee_pose)
        if pcd is None:
            points, colors = self.env.get_point_cloud()
        else:
            points, colors = pcd
        points, colors = self.exclude_robot_pcd(points, colors, thred=0.03)
        output_traj = self.motion_plan_with_tto(start_config, goal_config, points, colors=colors, batch_size=batch_size, in_hand=in_hand, max_neural_mp_rollout_length=100+i*50)
    
        if output_traj is None:
            print("Warning: Motion planner failed to find a path reaching the target region")
            output_traj = np.array([start_config]).reshape(1, -1)
        return output_traj
    
    def motion_plan_to_target_joint_angles_and_execute(self, target_joint_angles, speed=None, batch_size=10):
        start_config = self.env.get_joint_angles()
        points, colors = self.env.get_point_cloud()
        points, colors = self.exclude_robot_pcd(points, colors, thred=0.03)
        plan = self.motion_plan_with_tto(start_config, target_joint_angles, points, colors, batch_size=batch_size)
        return self.execute_motion_plan(plan, speed if speed else self.cfg.speed)
        
    def execute_motion_plan(
        self,
        plan,
        speed,
    ):
        """
        Execute a planned trajectory.

        Args:
            plan (list): List of joint angles along the path.
            init_joint_angles (np.ndarray): Initial joint angles of the robot.
            speed (float): Speed for execution (rad/s).
            set_intermediate_states (bool): Whether to set intermediate states.

        Returns:
            joint_error (float): Error in joint angles between the last state and the target state.
        """
        t1 = time.time()
        fa = self.env.env.franka_arm

        T = 5
        max_steps = 5
        motion_plan = np.concatenate((fa.get_joints().reshape(1, -1), plan))
        
        # interpolate between successive states of the motion plan
        interpolated_plan = []
        ctrl_hz = 50
        dt = 1/ctrl_hz
        for i in range(1, len(motion_plan)):
            joints_0 = motion_plan[i-1]
            joints_1 = motion_plan[i]
            max_action = max(abs(joints_1 - joints_0))
            num_steps = max(int(max_action / speed * ctrl_hz), max_steps)
            interpolated_steps = np.linspace(joints_0, joints_1, num=num_steps, endpoint=False)
            interpolated_plan.extend(interpolated_steps)
        pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)
        rate = rospy.Rate(1 / dt)
        
        print("Estimated time to complete trajectory: ", len(interpolated_plan) * dt)

        # To ensure skill doesn't end before completing trajectory, make the buffer time much longer than needed
        fa.goto_joints(interpolated_plan[1], duration=T, dynamic=True, buffer_time=60)
        init_time = rospy.Time.now().to_time()
        for i in range(2, len(interpolated_plan)):
            traj_gen_proto_msg = JointPositionSensorMessage(
                id=i, timestamp=rospy.Time.now().to_time() - init_time, 
                joints=interpolated_plan[i]
            )
            ros_msg = make_sensor_group_msg(
                trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                    traj_gen_proto_msg, SensorDataMessageType.JOINT_POSITION)
            )
            
            pub.publish(ros_msg)
            rate.sleep()

        # Stop the skill
        term_proto_msg = ShouldTerminateSensorMessage(timestamp=rospy.Time.now().to_time() - init_time, should_terminate=True)
        ros_msg = make_sensor_group_msg(
            termination_handler_sensor_msg=sensor_proto2ros_msg(
                term_proto_msg, SensorDataMessageType.SHOULD_TERMINATE)
            )
        pub.publish(ros_msg)
        fa.stop_skill()
        achieved_joint_angles = fa.get_joints()
        joint_error = np.linalg.norm(achieved_joint_angles - plan[-1])
        print("Motion Planning Execution Time: ", time.time() - t1)
        return joint_error
