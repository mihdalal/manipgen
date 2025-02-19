import torch
import numpy as np
import cv2

from manipgen.utils.dagger_utils import quat_mul, quat_conjugate, axis_angle_from_quat


def transform_quat(quat):
    """ Transform quaternion in proprioception to match simulation. """
    quat = torch.from_numpy(quat).float().reshape(-1, 4)
    
    # step 1: rotate around x-axis by 180 degrees
    down_q = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(quat.shape[0], -1)
    quat = quat_mul(quat, down_q)
    
    # step 2: follow convention w >= 0
    flip = quat[:, 3] < 0
    quat[flip] = -quat[flip]
    
    return quat[0].numpy()

def get_delta_proprioception(cur_eef_pos, cur_eef_quat, prev_eef_pos, prev_eef_quat, scale=1.0):
    """ 
    Compute delta proprioception to approximate velocity information. 
    
    Args:
        cur_eef_pos (np.ndarray): Current end-effector position.
        cur_eef_quat (np.ndarray): Current end-effector quaternion.
        prev_eef_pos (np.ndarray): Previous end-effector position.
        prev_eef_quat (np.ndarray): Previous end-effector quaternion.
        scale (float): Scale factor for the delta proprioception.
    
    Returns:
        np.ndarray: Delta proprioception.
    """
    cur_eef_pos = torch.from_numpy(cur_eef_pos).float().reshape(-1, 3)
    cur_eef_quat = torch.from_numpy(cur_eef_quat).float().reshape(-1, 4)
    prev_eef_pos = torch.from_numpy(prev_eef_pos).float().reshape(-1, 3)
    prev_eef_quat = torch.from_numpy(prev_eef_quat).float().reshape(-1, 4)
    
    delta_pos = (cur_eef_pos - prev_eef_pos) * scale
    delta_quat = quat_mul(cur_eef_quat, quat_conjugate(prev_eef_quat))
    axis_angle = axis_angle_from_quat(delta_quat)
    delta_rot = axis_angle * scale
    
    delta_proprio = torch.cat([delta_pos, delta_rot], dim=-1)[0].numpy()
    return delta_proprio

def process_depth(depth, cfg):
    """ Process depth image. """
    depth = depth.astype(np.float32) / 1e4      # Convert to meters
    
    # clamp and scale depth
    depth = depth.clip(0, cfg.depth_clamp_val)
    depth = depth / cfg.depth_clamp_val

    # rotate depth to fix orientation
    h, w = depth.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, cfg.process_depth_rotation, 1.0)
    depth = cv2.warpAffine(depth, M, (w, h))    
    
    # crop to policy view: 480 x 640 -> 368 x 368 (before center cropping)
    h_offset = cfg.process_depth_h_offset
    h_boarder = (cfg.raw_depth_h - cfg.obs_depth_w) // 2
    w_offset = cfg.process_depth_w_offset
    w_boarder = (cfg.raw_depth_w - cfg.obs_depth_w) // 2
    depth = depth[h_boarder+h_offset:-h_boarder+h_offset, w_boarder+w_offset:-w_boarder+w_offset]
    
    # resize to 84 x 84
    depth = cv2.resize(depth, (cfg.obs_depth_w_down, cfg.obs_depth_w_down), interpolation=cv2.INTER_NEAREST)
    
    return depth

def compute_policy_observations(
    obs, prev_obs=None, frame0_obs=None, seg_mask=None,
    cfg=None, device="cuda"
):
    """ 
    Compute local policy observations. 
    
    Args:
        obs (dict): Current observation.
        prev_obs (dict): Previous observation.
        frame0_obs (dict): First frame observation.
        seg_mask (np.ndarray): Local segmentation mask (for pick policy).
        cfg (dict): Configuration dictionary.
        device (str): Device to use.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: State observation and visual observation.
    """
    # Compute proprioception
    cur_eef_pos = obs["eef_pos"]
    cur_eef_quat = obs["eef_quat"]
    frame0_eef_pos = frame0_obs["eef_pos"]
    frame0_eef_quat = frame0_obs["eef_quat"]
    prev_eef_pos = prev_obs["eef_pos"]
    prev_eef_quat = prev_obs["eef_quat"]
    delta_proprio = get_delta_proprioception(cur_eef_pos, cur_eef_quat, prev_eef_pos, prev_eef_quat, cfg.delta_proprio_scale)

    state_obs = np.concatenate([cur_eef_pos, cur_eef_quat, frame0_eef_pos, frame0_eef_quat, delta_proprio])
            
    if cfg.offset_eef_pos_by_frame0:
        state_obs[:3] -= state_obs[7:10]
        state_obs[7:10] = 0.0

    # Compute visual observation
    cur_depth = process_depth(obs["cam1_depth"][0], cfg)
    frame0_depth = process_depth(frame0_obs["cam1_depth"][0], cfg)
    obs_list = [cur_depth, frame0_depth]
    if seg_mask is not None:
        obs_list.append(seg_mask)
    visual_obs = np.stack(obs_list)
    
    state_obs = torch.from_numpy(state_obs).float().to(device)
    visual_obs = torch.from_numpy(visual_obs).float().to(device)
    
    # create batch and time dimensions
    state_obs = state_obs.unsqueeze(0).unsqueeze(0)
    visual_obs = visual_obs.unsqueeze(0)
    
    return state_obs, visual_obs

