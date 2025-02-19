from contextlib import contextmanager
import os
import sys
import numpy as np
import torch

from robofin.robots import FrankaRobot
from tracikpy import TracIKSolver
from scipy.spatial.transform import Rotation
import pytorch_kinematics as pk

from manipgen.real_world.industreal_psl_env import FrankaRealPSLEnv

POSE_DIFF = np.array([[ 1.00000953e+00, -3.39917277e-05,  1.58157780e-04, -2.46808251e-05],
                  [ 3.41456136e-05,  1.00000940e+00, -2.54354871e-04,  1.11017342e-04],
                  [-1.58178732e-04,  2.54409425e-04,  9.99999740e-01, -1.99371854e-01],
                  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

PI = np.pi
EPS = np.finfo(float).eps * 4.

@contextmanager
def suppress_stdout():
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(os.devnull, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different

class IndustrealEnvWrapper:
    def __init__(self, env: FrankaRealPSLEnv):
        self.env = env
        full_urdf = FrankaRobot.urdf
        self.chain = pk.build_serial_chain_from_urdf(open(full_urdf).read(), "panda_hand")
        self.chain_gpu = pk.build_serial_chain_from_urdf(open(full_urdf).read(), "panda_hand")
        self.chain_gpu = self.chain_gpu.to(dtype=torch.float32, device=torch.device("cuda"))
        # get robot joint limits
        lim = torch.tensor(self.chain.get_joint_limits())
        
        # clamp limit for safety
        lim[0, :] += np.deg2rad(5.0)
        lim[1, :] -= np.deg2rad(5.0)
        self.joint_limits = torch.tensor(lim).cuda()

        # create the IK object
        # see the constructor for more options and their explanations, such as convergence tolerances
        retry_configs = torch.from_numpy(self.get_joint_angles()).float().unsqueeze(0)
        self.ik = pk.PseudoInverseIK(self.chain, max_iterations=30, num_retries=10,
                                joint_limits=lim.T,
                                early_stopping_any_converged=True,
                                early_stopping_no_improvement="all",
                                debug=False,
                                lr=0.2,
                                retry_configs=retry_configs)
        self.ik_solver = TracIKSolver(
            FrankaRobot.urdf,
            "panda_link0",
            "panda_hand",
        )
        
    # utils for robot proprio from neural mp
    def get_joint_angles(self):
        """
        Get the joint angles of the robot.

        Returns:
            np.ndarray: 7-dof joint angles.
        """
        obs = self.env.get_proprio_obs()
        return obs["q_pos"]

    def get_joint_vels(self):
        """
        Get the joint velocities of the robot.

        Returns:
            np.ndarray: 7-dof joint velocities.
        """
        obs = self.env.get_proprio_obs()
        return obs["q_vel"]

    def get_ee_pose(self, joint_angles=None):
        """
        Get the end effector pose.

        Returns:
            np.ndarray: 7D end effector pose w/ quat as (w, x, y, z).
        """
        if joint_angles is None:
            joint_angles = self.get_joint_angles()
        ret = self.chain.forward_kinematics(torch.from_numpy(joint_angles).float(), end_only=True)
        m = ret.get_matrix()
        pos = m[0, :3, 3]
        quat = Rotation.from_matrix(m[0, :3, :3]).as_quat()
        return np.concatenate([pos, quat])

    def get_gripper_width(self):
        """
        Get the gripper width.

        Returns:
            float: Gripper width.
        """
        obs = self.env.get_proprio_obs()
        return obs["gripper_width"]

    def compute_ik(self, target_ee_pose):
        """
        Compute Inverse Kinematics for the robot.
        Args:
            target_ee_pose (np.ndarray): 7D end effector pose. quat is in wxyz format
        Returns:
            joint_angles (np.ndarray): 7-dof joint angles.
        """
        target_ee_pose_4x4 = pk.Transform3d(pos=target_ee_pose[:3], rot=Rotation.from_quat(target_ee_pose[3:]).as_matrix())
        sol = self.ik.solve(target_ee_pose_4x4)
        converged = sol.converged
        qout = sol.solutions
        return np.array(qout, dtype=np.float32)[0][0]

    def get_point_cloud(self):
        """
        Get the point cloud from the camera sensor.

        Returns:
            np.ndarray: Point cloud.
        """
        return self.env.get_multi_cam_pcd()
    
    def set_robot_joint_state(self, joint_angles):
        """
        Set the robot joint angles. Note that this is not collision avoidance aware.

        Args:
            joint_angles (np.ndarray): 7-dof joint angles.
        """
        self.env.franka_arm.goto_joints(joints=joint_angles, use_impedance=False, ignore_virtual_walls=True)
        
    def execute_joint_action(self, action_target, speed):
        """
        Execute a joint action on the robot.

        Args:
            joint_angles (np.ndarray): 7-dof joint angles.
        """
        self.env.execute_joint_action(action_target, speed)