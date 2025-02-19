import os
import math
import imageio
from collections import OrderedDict
import numpy as np
import torch

# =============================================================================
# We copy the following functions from IsaacGymEnvs 
# so that the real world code does not depend on IsaacGym.
# =============================================================================

def axis_angle_from_quat(quat, eps=1.0e-6):
    """Convert tensor of quaternions to tensor of axis-angles."""
    # Reference: https://github.com/facebookresearch/pytorch3d/blob/bee31c48d3d36a8ea268f9835663c52ff4a476ec/pytorch3d/transforms/rotation_conversions.py#L516-L544

    mag = torch.linalg.norm(quat[:, 0:3], dim=1)
    half_angle = torch.atan2(mag, quat[:, 3])
    angle = 2.0 * half_angle
    sin_half_angle_over_angle = torch.where(torch.abs(angle) > eps,
                                            torch.sin(half_angle) / angle,
                                            1 / 2 - angle ** 2.0 / 48)
    axis_angle = quat[:, 0:3] / sin_half_angle_over_angle.unsqueeze(-1)

    angle = torch.norm(axis_angle, dim=1).unsqueeze(-1)
    axis = axis_angle / angle
    axis_angle = torch.where(
        angle > math.pi, (angle - 2 * math.pi) * axis, axis_angle
    )

    return axis_angle

@torch.jit.script
def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([x, y, z, w], dim=-1).view(shape)

    return quat

@torch.jit.script
def quat_conjugate(a):
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).view(shape)



def make_video(frames, logdir, epoch, name=None):
    filename = os.path.join(logdir, f"viz_{epoch}.mp4" if name is None else name)
    frames = np.asarray(frames)
    with imageio.get_writer(filename, fps=20) as writer:
        for frame in frames:
            writer.append_data(frame)

def get_delta_proprioception(current_proprio, prev_proprio, scale=60.0):
    """ Compute delta proprioception to approximate velocity information. """
    proprio_shape = current_proprio.shape

    current_proprio = current_proprio.reshape(-1, 7)
    prev_proprio = prev_proprio.reshape(-1, 7)

    delta_pos = (current_proprio[:, :3] - prev_proprio[:, :3]) * scale
    delta_quat = quat_mul(
        current_proprio[:, 3:],
        quat_conjugate(prev_proprio[:, 3:])
    )

    axis_angle = axis_angle_from_quat(delta_quat)
    delta_rot = axis_angle * scale

    delta_proprio = torch.cat([delta_pos, delta_rot], dim=-1)
    return delta_proprio.reshape(*proprio_shape[:-1], 6)

def get_student_obs(
        state_obs, 
        visual_obs, 
        state_frame0_obs, 
        visual_frame0_obs, 
        prev_state_obs, 
        visual_obs_type, 
        seg_frame0_obs=None,
        offset_eef_pos_by_frame0=True
    ):
    """ Concatenate current and first frame observations. """
    delta_proprio = get_delta_proprioception(state_obs.clone(), prev_state_obs.clone())
    concat_obs = torch.cat([state_obs, state_frame0_obs, delta_proprio], dim=-1)
    
    if offset_eef_pos_by_frame0:
        concat_obs[..., :3] -= state_frame0_obs[..., :3]
        concat_obs[..., 7:10] = 0.0

    if visual_obs_type == "pcd":
        # add semantic label to visual obs
        cur_frame_label = torch.zeros(*visual_obs.shape[:-1], 1, device=state_obs.device)
        first_frame_label = torch.ones(*visual_frame0_obs.shape[:-1], 1, device=state_obs.device)
        visual_obs = torch.cat([visual_obs, cur_frame_label], dim=-1)
        visual_frame0_obs = torch.cat([visual_frame0_obs, first_frame_label], dim=-1)
        concat_visual_obs = torch.cat([visual_obs, visual_frame0_obs], dim=-2)
    elif visual_obs_type == "depth":
        # two depths as two channels
        obs_list = [visual_obs, visual_frame0_obs]
        if seg_frame0_obs is not None:
            obs_list.append(seg_frame0_obs.clone())
        concat_visual_obs = torch.cat(obs_list, dim=-3)
    else:
        raise ValueError(f"Invalid visual_obs_type: {visual_obs_type}")
    
    return concat_obs, concat_visual_obs

def get_rollout_action(model, state_obs, visual_obs):
    return model.get_action(
        obs_dict={
            "state": state_obs, 
            "visual": visual_obs,
        },
    )

def get_obs_shape_meta(cfg):
    if cfg.dagger.visual_obs_type == "pcd":
        num_points = cfg.pcd_handler.downsample
        obs_shape_meta = {
            'ac_dim': 6, 
            'all_shapes': OrderedDict([('state', [20]), ('visual', [num_points * 2, 4])]),
            'all_obs_keys': ['state', 'visual'],
            'use_images': False,
            'use_depths': False,
        }
    elif cfg.dagger.visual_obs_type == "depth":
        h, w = cfg.task.env.local_obs.width, cfg.task.env.local_obs.height
        c = 3 if cfg.dagger.use_seg_obs else 2
        obs_shape_meta = {
            'ac_dim': 6, 
            'all_shapes': OrderedDict([('state', [20]), ('visual', [c, h, w])]),
            'all_obs_keys': ['state', 'visual'],
            'use_images': False,
            'use_depths': True,
        }
    return obs_shape_meta