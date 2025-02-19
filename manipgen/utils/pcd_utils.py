from typing import List, Optional, Union

import torch
import numpy as np


PCDs = Union[torch.Tensor, List[torch.Tensor]]


def is_empty(point_cloud: Optional[torch.Tensor]) -> bool:
    """
    Checks if the point cloud is empty/invalid.
    """
    return point_cloud is None or len(point_cloud) == 0


def generate_global_point_clouds(depths, proj_mats, view_mats, additional_data=None, device=None) -> torch.Tensor:
    """
    Given depth images from various cameras, along with respective projection/view
    (i.e. intrinsic/extrinsic) matrices, create a singular pointcloud representation
    of the scene, using the shared global ref frame. Attaches additional data
    (like rgb/segmentation) if desired.
    """
    if device is None:
        device = depths.device

    depths = depths.to(device)
    proj_mats = proj_mats.to(device)
    view_mats = view_mats.to(device)
    if additional_data is not None:
        additional_data = [data.to(device) for data in additional_data]
    else:
        additional_data = []

    B, C, height, width = depths.shape  # batch, n_cam, h, w
    N = height * width  # number of points in data
    u, v = torch.meshgrid(torch.arange(width, device=device), torch.arange(height, device=device), indexing='xy')
    fu = 2 / proj_mats[:, :, 0, 0].reshape(B, C, 1, 1)
    fv = 2 / proj_mats[:, :, 1, 1].reshape(B, C, 1, 1)

    centerU = width / 2
    centerV = height / 2

    Z = depths
    X = -(u - centerU) / width * Z * fu   # negative
    Y =  (v - centerV) / height * Z * fv  # NOT negative

    Z = Z.reshape(B, C, N)
    X = X.reshape(B, C, N)
    Y = Y.reshape(B, C, N)
    ones = torch.ones(*Z.shape, device=device)

    local_coords = torch.stack((X, Y, Z, ones), dim=-1)
    global_coords = local_coords @ torch.inverse(view_mats)

    unfiltered = global_coords[..., :3]
    assert torch.max(torch.abs(global_coords[..., 3] - 1)) < 1e-5
    if len(additional_data) > 0:
        additional_data = [data.reshape(B, C, N, -1) for data in additional_data]
        unfiltered = torch.cat([unfiltered, *additional_data], dim=-1)
    unfiltered = unfiltered.reshape(B, C * N, -1)

    return unfiltered


def localize_point_clouds(global_point_clouds, eef_pos, eef_quat=None, device=None, filter=True, filter_radius=0.2) -> PCDs:
    """
    Given pointclouds with coordinates in global ref frame, shift the points to
    be relative to the given eef position. Additionally, if desired (i.e. filter argument),
    crop the pointcloud by various metrics (e.g. distance to eef or alignment with eef_quat).

    Parameters:
    - global_point_clouds:  tensor of pointclouds in arbitrary (e.g. global) ref frame
    - eef_pos:              position of eef, in same ref frame as global_point_clouds
    - eef_quat:             orietation of eef as quaternion (currently unused)
    - device (optional):    device to run on (defaults to global_point_clouds.device)
    - filter:               True, if filtering is desired, else False
    - filter_radius:        radius around eef_pos to keep points

    Return: list or tensor of pointclouds, post-localization/filtering.
    """
    assert eef_pos.shape == (len(global_point_clouds), 3), f"got shape {eef_pos.shape}"

    if device is None:
        device = global_point_clouds.device

    global_point_clouds = global_point_clouds.to(device)
    eef_pos = eef_pos.unsqueeze(dim=1).to(device)
    if eef_quat is not None:
        eef_quat = eef_quat.to(device)

    local_xyz = global_point_clouds[..., :3] - eef_pos  # center point cloud around end-effector
    local_points = torch.cat([local_xyz, global_point_clouds[..., 3:]], dim=-1)

    if filter:
        filtered_points = []
        for b_idx, points in enumerate(local_points):
            # points = [N, (XYZ...)]
            xyz = points[:, :3]
            points = points[torch.linalg.vector_norm(xyz, dim=-1) < filter_radius]

            if eef_quat is not None:
                # TODO crop pointcloud based on eef orientation/rotation
                quat = eef_quat[b_idx]

            filtered_points.append(points)
        local_points = filtered_points

    return local_points


def noise_depths(depths: torch.Tensor, added_noise_sigma=None, blur_strength=None) -> torch.Tensor:
    """
    Adds sensor noise (centered gaussian with stdev=added_noise_sigma) and blurs depths, if desired.
    
    The intuition behind blurring the depths stems from analysis of real-world depth sensor outputs.
    In particular, depth outputs seem to blend in value between neighboring pixels, which creates
    rounded edges in the generated PCD and "floating bridges" of points between nearby objects of
    similar depths.
    """
    # shape = [B, C, H, W] -- B envs, C cameras, H x W image
    if added_noise_sigma is not None:
        depths = depths + torch.randn_like(depths) * added_noise_sigma

    if blur_strength is not None:
        # TODO blend neighbor depths to create smoother depth output
        #  - maybe gaussian blur kernel?
        pass

    return depths


def resample_point_clouds(point_clouds: PCDs, resample_N) -> PCDs:
    """
    Resample pointcloud to desired size.
    
    Cases:
    - pcd is empty: Do nothing
    - pcd is smaller than desired: Randomly duplicate points (as evenly as possible) across pcd
    - pcd is larger than desired: Randomly subsample points

    Returns a list of None (i.e. empty pcs) or pcd tensors of correct size
    """
    filtered_pcds = []
    for pcd in point_clouds:
        if is_empty(pcd):
            filtered_pcds.append(None)
            continue

        if len(pcd) < resample_N:
            n_reps = resample_N // len(pcd)
            rand_ind = np.random.choice(len(pcd), size=resample_N % len(pcd), replace=False)
            rand_ind = np.random.permutation(np.concatenate([np.arange(len(pcd))] * n_reps + [rand_ind]))
        else:
            rand_ind = np.random.permutation(len(pcd))[:resample_N]

        filtered_pcds.append(pcd[rand_ind])
 
    return filtered_pcds
