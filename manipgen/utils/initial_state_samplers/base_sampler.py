from isaacgym import gymtorch
from isaacgymenvs.utils.torch_jit_utils import (
    quat_conjugate, 
    quat_rotate,
    quat_mul, 
)

import torch
import numpy as np

class BaseSampler:
    def __init__(self, env, vision_based=False, cfg=None):
        self.env = env
        self.gym = self.env.gym
        self.sim = self.env.sim
        self.device = self.env.device
        self.num_envs = self.env.num_envs
        self.env.disable_automatic_reset = True
        self.env.disable_hardcode_control = True
        self.cfg = cfg

        # samplers set different level of diveristy for state-based and vision-based policies
        self.vision_based = vision_based

    def _orientation_error(self, desired, current):
        cc = quat_conjugate(current)
        q_r = quat_mul(desired, cc)
        return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

    def _control_ik(self, dpose):
        """Teleport the robot using IK"""
        j_eef_T = torch.transpose(self.env.fingertip_centered_jacobian, 1, 2)
        lmbda = torch.eye(6, device=self.device) * (self.damping**2)
        u = (j_eef_T @ torch.inverse(self.env.fingertip_centered_jacobian @ j_eef_T + lmbda) @ dpose).view(
            -1, 7
        )
        return u
    
    def _get_camera_pose(self):
        """Get the camera pose in world frame"""
        eef_quat = self.env.fingertip_centered_quat
        hand_pos = self.env.hand_pos

        camera_offset = torch.tensor(self.env.cfg_task.env.local_obs.camera_offset, device=self.device)
        camera_offset = camera_offset.unsqueeze(0).expand(self.num_envs, -1)

        camera_pos = hand_pos + quat_rotate(eef_quat, camera_offset)
        camera_quat = eef_quat
        camera_pose = torch.cat([camera_pos, camera_quat], dim=1)

        return camera_pose

    def _check_object_in_camera_view(self, camera_pose, object_pose, object_points, fov=42, clamp_dist=0.28, fov_threshold=5.0):
        """Check if the object is in the camera view
        Args:
            camera_pose: camera pose in world frame
            fov: camera fov
            clamp_dist: maximum visible distance
            object_pose: object pose in world frame
            object_points: object points in object frame
            fov_threshold: the threshold for the object to be considered visible
        Returns:
            visible: whether the object is visible
        """
        
        # convert object pose to camera frame
        object_pos_local = quat_rotate(quat_conjugate(camera_pose[:, 3:7]), object_pose[:, 0:3] - camera_pose[:, 0:3])
        object_quat_local = quat_mul(quat_conjugate(camera_pose[:, 3:7]), object_pose[:, 3:7])

        # convert objects points to camera frame
        num_points = object_points.shape[0]
        object_pos_local = object_pos_local.unsqueeze(1).expand(-1, num_points, -1).reshape(-1, 3)          # (num_envs x num_points, 3)
        object_quat_local = object_quat_local.unsqueeze(1).expand(-1, num_points, -1).reshape(-1, 4)        # (num_envs x num_points, 4)
        object_points = object_points.unsqueeze(0).expand(self.num_envs, -1, -1).reshape(-1, 3)             # (num_envs x num_points, 3)
        object_points_local = quat_rotate(object_quat_local, object_points) + object_pos_local

        # check if the object is in the camera view
        h_angle = torch.atan2(object_points_local[:, 1], object_points_local[:, 2])
        v_angle = torch.atan2(object_points_local[:, 0], object_points_local[:, 2])
        dist = torch.norm(object_points_local, dim=1)

        visible = (torch.abs(h_angle) < np.deg2rad(fov / 2 - fov_threshold)) \
                & (torch.abs(v_angle) < np.deg2rad(fov / 2 - fov_threshold)) \
                & (dist < clamp_dist)
        
        visible = visible.reshape(self.num_envs, -1).any(dim=1)
        
        return visible

    def generate(self, num_samples):
        """Sample initial states for a task
        Args:
            num_samples: number of samples
        Returns:
            init_states: a dict containing the sampled states
        """
        raise NotImplementedError
    
    def generate_easy(self, num_samples):
        """Sample initial states in easy mode for a task
        Args:
            num_samples: number of samples
        Returns:
            init_states: a dict containing the sampled states
        """
        raise NotImplementedError

    def filter(self, init_states):
        """In IsaacGym, environments have different simulation results even with the same initial states.
           This brings some bad initial states that we want to filter out, e.g., the object falls off the gripper at the beginning in a placing policy.
        Args:
            init_states: a dict containing the sampled states
        Returns:
            init_states: filtered initial states
        """
        return init_states
    
    def filter_vision(self, init_states, filter_vision_threshold=0.001):
        """Filter out the initial states based on the vision results.
           In distillation, the object of concern should be visible in the first frame.
        Args:
            init_states: a dict containing the sampled states
            filter_vision_threshold: the object of concern is considered invisible if it takes up less than this fraction of the camera input
        Returns:
            init_states: filtered initial states
        """
        return init_states
