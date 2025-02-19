import cv2
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def display_rgb(rgb):
    """Visualize RGB image."""
    plt.figure()
    plt.imshow(rgb)
    plt.tight_layout()
    plt.show()
    plt.close()

def display_raw_depth(depth, clamp_threshold=0.3, display=True):
    """Convert raw depth image to colorized depth image for visualization."""
    depth = depth.astype(np.float32) / 1e4
    depth = depth.clip(0, clamp_threshold)
    depth = -depth
    min_val = np.min(depth)
    max_val = np.max(depth)
    depth_range = max_val - min_val
    depth_image = (255.0 / depth_range * (depth - min_val)).astype("uint8")
    depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_INFERNO)
    depth_image = depth_image[:, :, ::-1]
    if display:
        display_rgb(depth_image)
    return depth_image

def display_pcd(points, colors=None):
    """Visualize point cloud."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])

def crop_depth_to_policy_view(depth, cfg):
    """Crop depth image to local policy view."""
    assert depth.shape == (cfg.raw_depth_h, cfg.raw_depth_w)
    
    h, w = depth.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, cfg.process_depth_rotation, 1.0)
    depth = cv2.warpAffine(depth, M, (w, h))    

    # crop to policy view (after center cropping)
    h_offset = cfg.process_depth_h_offset
    h_boarder = (cfg.raw_depth_h - cfg.vis_depth_w) // 2
    w_offset = cfg.process_depth_w_offset
    w_boarder = (cfg.raw_depth_w - cfg.vis_depth_w) // 2
    depth = depth[h_boarder+h_offset:-h_boarder+h_offset, w_boarder+w_offset:-w_boarder+w_offset]
    
    # resize to 76 x 76
    depth = cv2.resize(depth, (cfg.vis_depth_w_down, cfg.vis_depth_w_down), interpolation=cv2.INTER_NEAREST)
    
    return depth

def crop_rgb_to_policy_view(rgb, cfg):
    """Crop RGB image to local policy view."""
    assert rgb.shape == (cfg.raw_depth_h, cfg.raw_depth_w, 3)
    
    h, w, _ = rgb.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, cfg.process_depth_rotation, 1.0)
    rgb = cv2.warpAffine(rgb, M, (w, h))
    
    # crop to policy view (after center cropping)
    h_offset = cfg.process_depth_h_offset
    h_boarder = (cfg.raw_depth_h - cfg.vis_depth_w) // 2
    w_offset = cfg.process_depth_w_offset
    w_boarder = (cfg.raw_depth_w - cfg.vis_depth_w) // 2
    rgb = rgb[h_boarder+h_offset:-h_boarder+h_offset, w_boarder+w_offset:-w_boarder+w_offset]
    
    # resize to 76 x 76
    rgb = cv2.resize(rgb, (cfg.vis_depth_w_down, cfg.vis_depth_w_down), interpolation=cv2.INTER_NEAREST)
    
    return rgb

def debug_wrist_view(env, cfg):
    """Display visual observation for local policies."""
    obs = env.get_obs()
    depth = obs['cam1_depth'][0].copy()
    depth = crop_depth_to_policy_view(depth, cfg)
    display_raw_depth(depth)
