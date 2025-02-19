import cv2
from isaacgym import gymapi
import os
import numpy as np
import imageio.v2 as imageio
from pathlib import Path

color_group = np.array(
    [
        [0, 0, 0],  # Black
        [255, 0, 0],  # Red
        [0, 255, 0],  # Green
        [0, 0, 255],  # Blue
        [255, 0, 255],  # Magenta
        [255, 255, 0],  # Yellow
        [0, 255, 255],  # Cyan
        [128, 0, 0],  # Maroon
        [0, 128, 0],  # Green (dark)
        [0, 0, 128],  # Navy
        [128, 128, 128],  # Gray
    ]
)


def make_gif_from_files(video_dir, files_dir, name=None):
    video_dir = Path(video_dir)
    files_dir = Path(files_dir)
    filenames = os.listdir(files_dir)
    filenames = [f for f in filenames if f.endswith(".png")]
    filenames.sort()
    gif_name = "movie.gif" if name is None else name + ".gif"
    with imageio.get_writer(video_dir / gif_name, mode="I", loop=0) as writer:
        for filename in filenames:
            image = imageio.imread(video_dir / "figs" / filename)
            writer.append_data(image)
    os.system(f"rm {str(files_dir)}/*.png")


def make_gif_from_numpy(video_dir, images, name=None):
    video_dir = Path(video_dir)
    video_dir.mkdir(exist_ok=True)
    gif_name = "movie.gif" if name is None else name + ".gif"
    with imageio.get_writer(video_dir / gif_name, mode="I", loop=0) as writer:
        for image in images:
            writer.append_data(image)

def make_mp4_from_numpy(video_dir, images, movie_name=None, fps=30):
    video_dir = Path(video_dir)
    video_dir.mkdir(exist_ok=True)
    mp4_name = "movie.mp4" if movie_name is None else movie_name + ".mp4"
    with imageio.get_writer(video_dir / mp4_name, fps=fps) as writer:
        for image in images:
            writer.append_data(image)


def camera_shot(env, env_ids=None, camera_ids=None, use_depth=False, use_seg=False):
    """
    Args:
        env: isaacgym environment
        env_ids: list of environment ids to capture images
        camera_ids: list of camera ids to capture images
        use_depth: if true, capture depth image
        use_seg: if true, capture segmentation image
    Returns:
        images: List[List[np.ndarray]], RGB images from all specified environments and cameras
        depth_images: depth images from all specified environments and cameras
        seg_images: segmentation images from all specified environments and cameras
    """

    assert env.capture_video, "Camera is not enabled."
    if env_ids is None:
        env_ids = range(min(env.num_envs, len(env.camera_handles)))
    if camera_ids is None:
        camera_ids = range(len(env.camera_handles[0]))
    if not env.render_viewer:
        if env.device != "cpu":
            env.gym.fetch_results(env.sim, True)
        env.gym.step_graphics(env.sim)
    env.gym.render_all_camera_sensors(env.sim)

    images = []
    depth_images = []
    seg_images = []

    for env_id in env_ids:
        images.append([])
        depth_images.append([])
        seg_images.append([])
        for camera_id in camera_ids:
            camera_handle = env.camera_handles[env_id][camera_id]
            camera_image = env.gym.get_camera_image(
                env.sim, env.env_ptrs[env_id], camera_handle, gymapi.IMAGE_COLOR
            )
            shape = camera_image.shape
            camera_image = camera_image.reshape(shape[0], -1, 4)
            images[-1].append(camera_image)

            if use_depth:
                depth_image = env.gym.get_camera_image(
                    env.sim, env.env_ptrs[env_id], camera_handle, gymapi.IMAGE_DEPTH
                )

                depth_image[depth_image == -np.inf] = -255.0
                depth_image[depth_image == np.inf] = -255.0
                depth_image[depth_image < -255.0] = -255.0
                depth_image[depth_image > 0.0] = 0.0

                # relative depth
                depth_image = (depth_image - np.min(depth_image)) / (
                    np.max(depth_image) - np.min(depth_image + 1e-4)
                )

                # visualize
                depth_image = (255.0 * depth_image).astype(np.uint8)
                depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_INFERNO)
                depth_image = depth_image[:, :, ::-1]

                depth_images[-1].append(depth_image)

            if use_seg:
                seg_image = env.gym.get_camera_image(
                    env.sim, env.env_ptrs[env_id], camera_handle, gymapi.IMAGE_SEGMENTATION
                )
                seg_image = color_group[seg_image].astype(np.uint8)

                seg_images[-1].append(seg_image)

    return images, depth_images, seg_images

def vis_depth(depth):
    min_val = np.min(depth)
    max_val = np.max(depth)
    depth_range = max_val - min_val
    depth_image = (255.0 / depth_range * (depth - min_val)).astype("uint8")
    depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_INFERNO)
    depth_image = depth_image[:, :, ::-1]
    return depth_image

def apply_mask(image, mask):
    mask = mask.astype(np.bool)
    if not mask.any():
        return image
    color = np.array([30, 144, 255]).reshape(1, -1)
    image[mask] = image[mask] * 0.5 + color * 0.5
    return image
