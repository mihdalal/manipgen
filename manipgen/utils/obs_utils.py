from abc import ABC, abstractmethod

import torch
from torchvision.transforms.v2 import GaussianBlur, RandomErasing, RandomAffine
import torch.nn.functional as F

import manipgen.utils.pcd_utils as PCDUtils
from isaacgymenvs.utils.torch_jit_utils import quat_mul

import random

def randomize_proprioception(eef_pos, eef_quat, pos_noise, rotation_noise):
    # apply noise to eef position: translation in uniform range [-pos_noise, pos_noise]
    eef_pos_noise = (torch.rand_like(eef_pos) * 2 - 1) * pos_noise
    eef_pos = eef_pos + eef_pos_noise

    # apply noise to eef orientation: rotation along any direction in uniform range [-rotation_noise, rotation_noise]
    p = torch.randn_like(eef_quat[:, :3])
    p = p / p.norm(dim=-1, keepdim=True)
    angle_half = torch.rand(eef_quat.shape[0], 1, device=eef_quat.device) * rotation_noise * 0.5
    eef_quat_noise = torch.cat([p * torch.sin(angle_half), torch.cos(angle_half)], dim=-1)
    eef_quat = quat_mul(eef_quat_noise, eef_quat)

    return eef_pos, eef_quat    


def get_visual_obs_handler(visual_obs_type, cfg):
    return {
        "depth": DepthObsHandler,
        "pcd": PCDObsHandler,
    }[visual_obs_type](cfg)

def get_seg_obs_handler(cfg):
    return SegmentationObsHandler(cfg)


class ObsHandler(ABC):
    @abstractmethod
    def get_visual_obs(envs, prev_vis_obs, **kwargs):
        pass

    @abstractmethod
    def apply_noise(visual_obs, **kwargs):
        pass

    @abstractmethod
    def format_for_robomimic(visual_obs, **kwargs):
        pass


class DepthObsHandler(ObsHandler):
    def __init__(self, cfg):
        self.cfg = cfg.depth_handler

    def get_visual_obs(self, envs, prev_vis_obs, **kwargs):
        depths, seg, rgb = envs.get_depth_observations()
        assert seg is None and rgb is None
        return depths

    def _depth_warping(self, depths, std=0.5, prob=0.8, device=None):
        n, _, h, w = depths.shape

        # Generate Gaussian shifts
        gaussian_shifts = torch.normal(mean=0, std=std, size=(n, h, w, 2), device=device).float()
        apply_shifts = torch.rand(n, device=device) < prob
        gaussian_shifts[~apply_shifts] = 0.0

        # Create grid for the original coordinates
        xx = torch.linspace(0, w - 1, w, device=device)
        yy = torch.linspace(0, h - 1, h, device=device)
        xx = xx.unsqueeze(0).repeat(h, 1)
        yy = yy.unsqueeze(1).repeat(1, w)
        grid = torch.stack((xx, yy), 2).unsqueeze(0)  # Add batch dimension

        # Apply Gaussian shifts to the grid
        grid = grid + gaussian_shifts

        # Normalize grid values to the range [-1, 1] for grid_sample
        grid[..., 0] = (grid[..., 0] / (w - 1)) * 2 - 1
        grid[..., 1] = (grid[..., 1] / (h - 1)) * 2 - 1

        # Perform the remapping using grid_sample
        depth_interp = F.grid_sample(depths, grid, mode='bilinear', padding_mode='border', align_corners=True)

        # Remove the batch and channel dimensions
        depth_interp = depth_interp.squeeze(0).squeeze(0)

        return depth_interp
    
    def _generate_mask(self, n, h, w, device):
        k_lower = self.cfg.augmentation.holes.kernel_size_lower
        k_upper = self.cfg.augmentation.holes.kernel_size_upper
        s_lower = self.cfg.augmentation.holes.sigma_lower
        s_upper = self.cfg.augmentation.holes.sigma_upper
        thresh_lower = self.cfg.augmentation.holes.thresh_lower
        thresh_upper = self.cfg.augmentation.holes.thresh_upper

        # generate random noise
        noise = torch.rand(n, 1, h, w, device=device)

        # apply gaussian blur
        k = random.choice(list(range(k_lower, k_upper+1, 2)))
        noise = GaussianBlur(kernel_size=k, sigma=(s_lower, s_upper))(noise)

        # normalize noise
        noise = (noise - noise.min()) / (noise.max() - noise.min())

        # apply thresholding
        thresh = torch.rand(n, 1, 1, 1, device=device) * (thresh_upper - thresh_lower) + thresh_lower
        mask = (noise > thresh)

        return mask

    def apply_noise(self, visual_obs, device=None, **kwargs):
        if device is None:
            device = visual_obs.device

        visual_obs_shape = visual_obs.shape
        h, w = visual_obs_shape[-2:]
        obs = visual_obs.reshape(-1, 1, h, w).clone().to(device)
        n = obs.shape[0]

        # apply depth warping
        if self.cfg.augmentation.depth_warping.enabled:
            obs = self._depth_warping(
                obs, 
                std=self.cfg.augmentation.depth_warping.std, 
                prob=self.cfg.augmentation.depth_warping.prob,
                device=device
            )

        # apply blurring
        if self.cfg.augmentation.gaussian_blur.enabled:
            transform = GaussianBlur(
                kernel_size=self.cfg.augmentation.gaussian_blur.kernel_size, 
                sigma=(self.cfg.augmentation.gaussian_blur.sigma_lower, self.cfg.augmentation.gaussian_blur.sigma_upper)
            )
            obs = transform(obs)

        # apply non-linear scaling
        if self.cfg.augmentation.scale.enabled:
            intensity = torch.rand(n, 1, 1, 1, device=device) * self.cfg.augmentation.scale.intensity
            corners = (2 * torch.rand(n, 1, 2, 2, device=device) - 1) * intensity + 1
            scale_map = F.interpolate(corners, size=(h, w), mode='bicubic', align_corners=False).reshape(n, 1, h, w)
            apply_scaling = torch.rand(n, device=device) < self.cfg.augmentation.scale.prob
            scale_map[~apply_scaling] = 1.0
            obs = obs * scale_map
            obs = torch.clamp(obs, 0, 1)

        # apply holes
        if self.cfg.augmentation.holes.enabled:
            mask = self._generate_mask(n, h, w, device)
            prob = self.cfg.augmentation.holes.prob
            keep_mask = torch.rand(n, device=device) < prob
            mask[~keep_mask, :] = 0
            obs[mask] = self.cfg.augmentation.holes.fill_value

        ret = obs.reshape(*visual_obs_shape).to(visual_obs.device)

        return ret

    def format_for_robomimic(self, visual_obs, **kwargs):
        # convert image obs from chw to hwc (how robomimic dataset expects it)
        return torch.movedim(visual_obs, -3, -1)


class PCDObsHandler(ObsHandler):
    def __init__(self, cfg):
        self.cfg = cfg.pcd_handler

    def get_visual_obs(self, envs, prev_vis_obs, *, device=None, include_additional_data=False, **kwargs):
        resample_N = self.cfg.downsample
        filter_radius = self.cfg.filter_radius
        eef_trans_noise = self.cfg.augmentation.eef_trans_noise
        eef_rot_noise = self.cfg.augmentation.eef_rot_noise
        
        obs_for_pcd = envs.get_depth_observations(for_point_cloud=True)
        depth_buf, proj_mats, view_mats, eef_pos, eef_quat, additional_data = obs_for_pcd
        depth_buf = PCDUtils.noise_depths(depth_buf)

        global_pcd = PCDUtils.generate_global_point_clouds(
            depths=depth_buf,
            proj_mats=proj_mats,
            view_mats=view_mats,
            additional_data=additional_data if include_additional_data else None,
            device=device,
        )

        eef_pos, eef_quat = randomize_proprioception(eef_pos, eef_quat, eef_trans_noise, eef_rot_noise)
        local_pcd = PCDUtils.localize_point_clouds(
            global_point_clouds=global_pcd,
            eef_pos=eef_pos,
            eef_quat=eef_quat,
            device=None,
            filter=device,
            filter_radius=filter_radius,
        )

        pcd_obs = PCDUtils.resample_point_clouds(
            point_clouds=local_pcd,
            resample_N=resample_N,  # TODO make this sample volume configurable
        )

        if prev_vis_obs is None:
            assert not any(PCDUtils.is_empty(pcd) for pcd in pcd_obs), \
                "Encountered empty PCD observation on first step of simulation"
        else:
            assert len(prev_vis_obs) == len(pcd_obs), "Previous and current visual_obs length mismatch"
            pcd_obs = [
                pcd if not PCDUtils.is_empty(pcd) else prev_vis_obs[idx].to(device)
                for idx, pcd in enumerate(pcd_obs)
            ]

        pcd_obs = torch.stack(pcd_obs, dim=0)
        return pcd_obs

    def apply_noise(self, visual_obs, **kwargs):
        jitter_prob = self.cfg.augmentation.jitter_prob
        jitter_ratio = self.cfg.augmentation.jitter_ratio
        jitter_std = self.cfg.augmentation.jitter_std
        jitter_clip = self.cfg.augmentation.jitter_clip

        visual_obs_shape = visual_obs.shape
        assert len(visual_obs_shape) == 4       # (n_envs, time, n_points, 3)
        n_envs, n_time, n_points, _ = visual_obs_shape
        visual_obs = visual_obs.reshape(-1, 3)

        # select points to jitter
        mask1 = torch.rand(n_envs * n_time, device=visual_obs.device) < jitter_prob                 # select pcd to jitter
        mask2 = torch.rand(n_envs * n_time * n_points, device=visual_obs.device) < jitter_ratio     # select points to jitter
        jitter_mask = mask1.unsqueeze(-1).repeat(1, n_points).reshape(-1) & mask2
        n_jitter_points = jitter_mask.sum().item()

        # sample jitter values
        jitter_std = torch.tensor(
            [jitter_std] * 3,
            dtype=torch.float32,
            device=visual_obs.device,
        ).view(1, 3).repeat(n_jitter_points, 1)

        jitter_mean = torch.zeros_like(jitter_std)
        jitter_dist = torch.distributions.normal.Normal(jitter_mean, jitter_std)
        jitter_value = jitter_dist.sample()
        jitter_value = torch.clamp(jitter_value, -jitter_clip, jitter_clip)

        # apply jitter
        visual_obs[jitter_mask] += jitter_value

        return visual_obs.reshape(*visual_obs_shape)

    def format_for_robomimic(self, visual_obs, **kwargs):
        return visual_obs


class SegmentationObsHandler:
    def __init__(self, cfg):
        self.cfg = cfg.segmentation_handler

    def apply_noise(self, seg_obs, device=None, **kwargs):
        if seg_obs is None:
            return seg_obs
        
        if device is None:
            device = seg_obs.device

        seg_obs_shape = seg_obs.shape
        n = seg_obs_shape[0]
        if len(seg_obs.shape) == 4:
            obs = seg_obs[:, 0].clone().to(device)
        else:
            seq_len = seg_obs.shape[1]
            obs = seg_obs[:, 0, 0].clone().to(device)

        # randomly zero out segmentation masks
        if self.cfg.augmentation.zero_out.enabled:
            zero_out = torch.rand(n, device=device) < self.cfg.augmentation.zero_out.prob
            obs[zero_out] = 0.0

        # random erasing
        if self.cfg.augmentation.random_erasing.enabled:
            transform = RandomErasing(
                p=self.cfg.augmentation.random_erasing.prob, 
                scale=(self.cfg.augmentation.random_erasing.scale_min, self.cfg.augmentation.random_erasing.scale_max)
            )
            obs = transform(obs)

        # randomly apply shifts to the segmentation masks
        if self.cfg.augmentation.random_shift.enabled:
            transform = RandomAffine(
                degrees=0,
                translate=(self.cfg.augmentation.random_shift.max_scale, self.cfg.augmentation.random_shift.max_scale),
            )
            obs_shift = transform(obs)
            shift = torch.rand(n, device=device) < self.cfg.augmentation.random_shift.prob
            obs[shift] = obs_shift[shift]

        seg_obs = obs.unsqueeze(1)
        if len(seg_obs_shape) == 5:
            seg_obs = seg_obs.unsqueeze(1).repeat(1, seq_len, 1, 1, 1)

        return seg_obs
