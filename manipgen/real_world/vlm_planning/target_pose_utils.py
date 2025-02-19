import os
import time
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d

import pytorch_kinematics as pk
import pytorch_volumetric as pv

from robofin.robofin.pointcloud.torch import FrankaSampler
from robofin.robofin.robots import FrankaRobot
from manipgen.real_world.utils.neural_mp_env_wrapper import IndustrealEnvWrapper
from manipgen.real_world.utils.real_world_collision_checker import FrankaCollisionChecker
from scipy.spatial.transform import Rotation

def backtrack_eef_position(ref_pos, quat, offset):
    """"
    Get end effector position based on reference position and offset.
    
    Args:
        ref_pos (np.ndarray): Reference end effector position.
        quat (np.ndarray): End effector quaternion.
        offset (float): Offset in the end effector frame. offset > 0 means pulling back.

    Returns:
        np.ndarray: End effector position.
    """
    
    offset = np.array([0.0, 0.0, -offset])
    
    # batched
    add_batch = False
    if len(ref_pos.shape) == 1:
        add_batch = True
        ref_pos = ref_pos[np.newaxis, :]
        quat = quat[np.newaxis, :]
        
    offset = np.repeat(offset[np.newaxis, :], ref_pos.shape[0], axis=0)
    
    # get rotation matrix from quat
    rot = R.from_quat(quat).as_matrix()
    
    # get end effector position
    eef_pos = ref_pos + np.matmul(rot, offset[:, :, np.newaxis])[:, :, 0]
    
    return eef_pos[0] if add_batch else eef_pos

def offset_eef_position(ref_pos, quat, offset):
    """
    Get end effector position based on reference position and offset. More general version of backtrack_eef_position.
    
    Args:
        ref_pos (np.ndarray): Reference end effector position.
        quat (np.ndarray): End effector quaternion.
        offset (np.ndarray): Offset in the end effector frame.
    
    Returns:
        np.ndarray: End effector position.
    """

    add_batch = False
    if len(ref_pos.shape) == 1:
        add_batch = True
        ref_pos = ref_pos[np.newaxis, :]
        quat = quat[np.newaxis, :]
        
    offset = np.repeat(offset[np.newaxis, :], ref_pos.shape[0], axis=0)
    
    # get rotation matrix from quat
    rot = R.from_quat(quat).as_matrix()
    
    # get end effector position
    eef_pos = ref_pos + np.matmul(rot, offset[:, :, np.newaxis])[:, :, 0]
    
    return eef_pos[0] if add_batch else eef_pos

def estimate_quat_from_object_points(points):
    """
    Estimate target end effector pose based on point cloud of object.
    Task: pick (table-top)
    """

    n = len(points)
    x = points[:, 0]
    y = points[:, 1]

    # Calculate the slope using the linear regression
    m = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x ** 2) - np.sum(x) ** 2)
    
    # convert slope to angle
    angle = np.arctan(m)
    
    # get quat from euler xyz: (np.pi, 0, angle)
    quat = R.from_euler('xyz', [np.pi, 0, angle]).as_quat()
    
    return quat

def get_candidate_eef_quat_from_handle_points(points, part_masks=None, num_samples=5):
    """
    Estimate target end effector pose based on point cloud of handle.
    Task: grasp handle
    """
    
    if part_masks is None:
        z_length = np.max(points[:, 2]) - np.min(points[:, 2])
        
        points_xy = points[:, :2].copy()
        points_xy -= np.max(points_xy, axis=0) + np.min(points_xy, axis=0) / 2
        xy_length = np.max(np.linalg.norm(points_xy, axis=1))
    else:
        z_length, xy_length = 0.0, 0.0
        for i, part_mask in enumerate(part_masks):
            if part_mask.sum() == 0:
                continue
            # get max and min column and row index of mask
            row_idx, col_idx = np.where(part_mask)
            z_length += row_idx.max() - row_idx.min()
            xy_length += col_idx.max() - col_idx.min()
    
    assert num_samples % 2 == 1

    # We sample multiple orientations for the handle with different rotation around z-axis.
    # The best orientation will be selected based on the collision checking.
    print("[Target Pose Estimation] Estimating handle orientation:", xy_length, z_length)
    if z_length > xy_length:
        print("vertical handle")
        
        euler = np.array([0.0, -np.pi / 2 - np.pi / 6, 0.0])
        euler_z = np.linspace(np.pi / 2, np.pi * 3 / 2, num_samples)
        euler = np.repeat(euler[np.newaxis, :], num_samples, axis=0)
        euler[:, 2] = euler_z
        quat = R.from_euler('xyz', euler).as_quat() # (num_samples, 4)
    else:
        print("horizontal handle")
        
        euler1 = np.array([np.pi / 2, 0.0, 0.0])
        euler1_z = np.linspace(0.0, np.pi / 2, int(num_samples / 2) + 1)
        euler1 = np.repeat(euler1[np.newaxis, :], int(num_samples / 2) + 1, axis=0)
        euler1[:, 2] = euler1_z
        quat1 = R.from_euler('xyz', euler1).as_quat()
        
        euler2 = np.array([-np.pi / 2, 0.0, 0.0])
        euler2_z = np.linspace(-np.pi / 2, 0.0, int(num_samples / 2) + 1)
        euler2 = np.repeat(euler2[np.newaxis, :], int(num_samples / 2) + 1, axis=0)
        euler2[:, 2] = euler2_z
        quat2 = R.from_euler('xyz', euler2).as_quat()
        
        quat = np.concatenate([quat1, quat2[1:]], axis=0)

    return quat

def get_candidate_eef_quat_for_tightspace(num_samples=5):
    """
    Get end effector orientation for tight space.
    Task: pick and place (tight space)
    """

    # We sample multiple orientations for the handle with different rotation around z-axis.
    # The best orientation will be selected based on the collision checking.
    euler = np.array([0.0, -(np.pi / 2 + np.pi / 6), 0.0])
    euler_z = np.linspace(np.pi / 2, np.pi * 3 / 2, num_samples)
    euler = np.repeat(euler[np.newaxis, :], num_samples, axis=0)
    euler[:, 2] = euler_z
    quat = R.from_euler('xyz', euler).as_quat() # (num_samples, 4)
    
    return quat

def sample_collision_free_eef_pose(object_pos, candidate_quat, scene_pcd, scene_colors, checker, offset=0.26, diffuse_pc=False):
    """
    Sample collision free end effector pose
    Task: pick, place (tight space) and grasp handle

    Args:
        object_pos (np.ndarray): Object position.
        candidate_quat (np.ndarray): Candidate end effector quaternion.
        scene_pcd (np.ndarray): Scene point cloud.
        scene_colors (np.ndarray): Scene point cloud colors.
        checker (TargetPoseChecker): Collision checker.
        offset (float): Offset in the end effector frame.
        diffuse_pc (bool): Whether to diffuse point cloud.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Target end effector position and quaternion.
    """

    num_samples = len(candidate_quat)
    target_pos = np.repeat(object_pos[np.newaxis, :], num_samples, axis=0)
    target_quat = candidate_quat
    target_pos = backtrack_eef_position(target_pos, target_quat, offset=offset)
    # check collision
    collision_num, joint_angles = checker.check_collision_batched(
        target_pos, target_quat, scene_pcd, scene_colors, diffuse_pc=diffuse_pc, visualize=False
    )
    print("[Target Pose Estimation] Collision number", collision_num)
    
    # get target pose with least collision
    min_collision_idx = np.argmin(collision_num)
    target_pos = target_pos[min_collision_idx]
    target_quat = target_quat[min_collision_idx]
    
    return target_pos, target_quat

class TargetPoseChecker:
    def __init__(self, env, use_robot_sdf=True):
        """
        Constructor for TargetPoseChecker.

        Args:
            env (FrankaRealPSLEnv): Real world environment.
            use_robot_sdf (bool): Whether to use robot SDF for collision checking. This is more accurate than sphere representation.
        """
        self.env = IndustrealEnvWrapper(env)
        self.gpu_fk_sampler = FrankaSampler("cuda", use_cache=False)
        self.collision_checker = FrankaCollisionChecker()

        current_joint_angles = self.env.get_joint_angles()
        full_urdf = FrankaRobot.urdf
        self.chain = pk.build_serial_chain_from_urdf(open(full_urdf).read(), "panda_hand")
        self.chain = self.chain.to(torch.float32, torch.device("cuda"))
        self.ik = pk.PseudoInverseIK(
            self.chain, max_iterations=30, num_retries=10,
            joint_limits=torch.tensor(self.chain.get_joint_limits(), device=torch.device("cuda")).T,
            early_stopping_any_converged=True,
            early_stopping_no_improvement="all",
            debug=False,
            lr=0.2,
            retry_configs=torch.from_numpy(current_joint_angles).float().unsqueeze(0).cuda()
        )
        
        self.use_robot_sdf = use_robot_sdf
        if self.use_robot_sdf:
            full_urdf_sdf = full_urdf.replace(".urdf", "_sdf.urdf")
            sdf_chain = pk.build_serial_chain_from_urdf(open(full_urdf_sdf).read(), "panda_hand")
            sdf_chain = sdf_chain.to(torch.float32, torch.device("cuda"))
            self.robot_sdf = pv.RobotSDF(sdf_chain, path_prefix=os.path.dirname(full_urdf_sdf))
            
    def exclude_robot_pcd(self, points, colors=None, thred=0.01):
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
    
    def check_collision_batched(self, positions, quaternions, pc, pc_colors=None, diffuse_pc=False, visualize=False):
        """
        Check collision for a batch of end effector poses.
        
        Args:
            positions (np.ndarray): End effector positions.
            quaternions (np.ndarray): End effector quaternions.
            pc (np.ndarray): Scene point cloud.
            pc_colors (np.ndarray, optional): Scene point cloud colors.
            diffuse_pc (bool): Whether to diffuse point cloud. This can help select the safest pose from several valid poses.
            visualize (bool): Whether to visualize point cloud.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Collision number and joint angles for each candidate pose.
        """
        # exclude robot points from the point cloud
        pc, pc_colors = self.exclude_robot_pcd(pc, pc_colors, thred=0.075)      # TODO: adjust the threshold if needed
            
        # solve ik
        pos = torch.from_numpy(positions).float().cuda()
        rot = Rotation.from_quat(quaternions).as_matrix()
        rot = torch.from_numpy(rot).float().cuda()
        rob_tf = pk.Transform3d(pos=pos, rot=rot, device=torch.device("cuda"))
        sol = self.ik.solve(rob_tf)
        joint_angles = sol.solutions.reshape(-1, 7)
        
        # subsample point cloud
        subsample_num = 16384
        idx = np.random.permutation(pc.shape[0])[:subsample_num]
        pc = pc[idx, :]
        if diffuse_pc:
            diffuse_scale = 0.10
            pc = pc + np.random.randn(*pc.shape) * diffuse_scale
        pc = torch.from_numpy(pc).float().cuda().unsqueeze(0).repeat(joint_angles.shape[0], 1, 1)

        # check collision
        tik = time.time()
        if self.use_robot_sdf:
            self.robot_sdf.set_joint_configuration(joint_angles)
            sdf = self.robot_sdf(pc[0:1])[0][:, 0]
            collision_num = (sdf < 0.05).sum(dim=1).cpu().numpy()
        else:
            collision_num = self.collision_checker.check_scene_collision_batch(
                joint_angles, pc, thred=0.01, sphere_repr_only=True
            ).cpu().numpy()
        tok = time.time()
        print("[Target Pose Estimation] Collision check time:", tok - tik)
        print("[Target Pose Estimation] Collision number", collision_num)
        
        if visualize:
            points = pc[0].cpu().numpy()
                
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            if pc_colors is not None:
                colors = pc_colors[idx, :]
                pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
            o3d.visualization.draw_geometries([pcd])
                
        return collision_num, joint_angles
