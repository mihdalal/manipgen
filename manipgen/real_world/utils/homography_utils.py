import pickle
import os

import numpy as np
import open3d as o3d

def get_cam_constants(cam_cfg):
    """
    Get camera constants from the camera configuration.

    Args:
        cam_cfg (dict): Camera configuration.

    Returns:
        Tuple: Camera constants (fx, fy, cx, cy, ds, pcs).
    """
    K1 = np.array(cam_cfg["intrinsics"])
    mv_shift = np.array(cam_cfg["mv_shift"])

    fx, fy, cx, cy = K1[0, 0], K1[1, 1], K1[0, -1], K1[1, -1]
    ds = 0.0010000000474974513
    pcs = mv_shift
    return fx, fy, cx, cy, ds, pcs

def save_pointcloud(_file, _points, _colors):
    """
    Save a point cloud to a file.

    Args:
        _file (str): Output file name.
        _points (np.ndarray): Point cloud points.
        _colors (np.ndarray): Point cloud colors.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(_points)
    pcd.colors = o3d.utility.Vector3dVector(_colors)
    o3d.io.write_point_cloud(_file, pcd)

class HomographyTransform:
    def __init__(self, key, transform_file, cam_cfg):
        """
        Initialize the HomographyTransform class.

        Args:
            key (str): Key for the transformation file.
            transform_file (str): Transformation file name.
            cam_cfg (dict): Camera configuration.
        """
        transform_file = key + "_" + transform_file + ".pkl"
        self.transform_file = transform_file
        self.transform_matrix = pickle.load(
            open(os.path.join(os.path.dirname(__file__), "../../homography_data/homography_transforms/" + transform_file), "rb")
        )
        self.fx, self.fy, self.cx, self.cy, self.ds, self.pcs = get_cam_constants(cam_cfg)
        self.workspace_min = np.array(cam_cfg["workspace_min"])
        self.workspace_max = np.array(cam_cfg["workspace_max"])
        self.filter_pc = True
        self.cam_cfg = cam_cfg

    def get_robot_coords_vectorized(self, px_arr, depth_im):
        """
        Get robot coordinates from image frame coordinates in a vectorized manner.

        Args:
            px_arr (np.ndarray): Pixel array.
            depth_im (np.ndarray): Depth image.

        Returns:
            np.ndarray: Robot coordinates.
        """
        X, Y = px_arr
        Z = depth_im * self.ds
        X = Z / self.fx * (X - self.cx)
        Y = Z / self.fy * (Y - self.cy)

        img_frame_coords = np.transpose(np.array([X, Y, Z]), (1, 2, 0))
        ones = np.ones(img_frame_coords.shape[:2] + (1,))
        return np.concatenate([img_frame_coords, ones], axis=-1) @ self.transform_matrix + self.pcs

    def get_filtered_pc(self, _points, _colors=None):
        """
        Get filtered point cloud.

        Args:
            _points (np.ndarray): Point cloud points.
            _colors (np.ndarray, optional): Point cloud colors.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Filtered points and colors.
        """
        mask = np.all(_points > self.workspace_min, axis=1) * np.all(
            _points < self.workspace_max, axis=1
        )
        if _colors is None:
            return _points[mask]
        else:
            return _points[mask], _colors[mask]

    def denoise_pc(self, points, colors=None):
        """
        Denoise a point cloud.

        Args:
            points (np.ndarray): Point cloud points.
            colors (np.ndarray, optional): Point cloud colors.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Denoised points and colors.
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        denoised_pc = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1)[0]
        if colors is None:
            return np.asarray(denoised_pc.points)
        else:
            return np.asarray(denoised_pc.points), np.asarray(denoised_pc.colors)

    def get_pointcloud(self, depth, image=None):
        """
        Get a point cloud from a depth image.

        Args:
            depth (np.ndarray): Depth image.
            image (np.ndarray, optional): RGB image.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Point cloud points and colors if image is provided, otherwise just points.
        """
        h, w = depth.shape

        mg = np.meshgrid(np.arange(0, 640, 640 / w), np.arange(0, 480, 480 / h))
        grid = np.concatenate([np.expand_dims(mg[1], 0), np.expand_dims(mg[0], 0)], axis=0)
        points = self.get_robot_coords_vectorized(grid, depth)

        if image is None:
            return points
        else:
            colors = image / 255.0
            return points, colors
