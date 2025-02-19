import argparse
import os
import pickle

import cv2
import hydra
import numpy as np
import pupil_apriltags as apriltag
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from manipgen.real_world.utils.homography_utils import get_cam_constants
from manipgen.real_world.industreal_psl_env import FrankaRealPSLEnv
from manipgen.real_world.utils.neural_mp_env_wrapper import IndustrealEnvWrapper
import manipgen.real_world.utils.calibration_utils as calibration_utils
import industreallib.control.scripts.control_utils as control_utils

def get_args():
    """Gets arguments from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--perception_config_file_name",
        default="perception.yaml",
        help="Perception configuration to load",
    )
    parser.add_argument("-c", "--cam_idx", required=True, help="camera id, single int")
    parser.add_argument(
        "--flip",
        action="store_true",
        help="Flip the eef to get better view for front cameras.",
    )
    parser.add_argument(
        "-d",
        "--debug_mode",
        action="store_true",
        required=False,
        help="Enable output for debugging",
    )
    args = parser.parse_args()

    return args


def get_img_frame_3d_coords(pixel, d_im, fx, fy, cx, cy, ds):
    """
    Get 3D coordinates from a 2D image frame.

    Args:
        pixel (tuple): Pixel coordinates (x, y).
        d_im (np.ndarray): Depth image.
        fx (float): Focal length in the x-axis.
        fy (float): Focal length in the y-axis.
        cx (float): Optical center in the x-axis.
        cy (float): Optical center in the y-axis.
        ds (float): Depth scaling factor.

    Returns:
        np.ndarray: 3D coordinates corresponding to the 2D pixel.
    """
    y, x = np.round(pixel).astype(int)
    Z = d_im[x, y] * ds
    X = Z / fx * (x - cx)
    Y = Z / fy * (y - cy)
    return np.array([X, Y, Z])


def compute_homography(eef_pose, depth, tag_center_pixel, out_file, fx, fy, cx, cy, ds):
    """
    Compute the homography transformation.

    Args:
        eef_pose (np.ndarray): End effector poses.
        depth (np.ndarray): Depth images.
        tag_center_pixel (list): Center pixel coordinates of tags.
        out_file (str): Output file name for saving the homography.
        fx (float): Focal length in the x-axis.
        fy (float): Focal length in the y-axis.
        cx (float): Optical center in the x-axis.
        cy (float): Optical center in the y-axis.
        ds (float): Depth scaling factor.
    """
    A = np.array(
        [
            get_img_frame_3d_coords(tag_center_pixel[i], depth[i], fx, fy, cx, cy, ds)
            for i in range(len(depth))
        ]
    )
    A = np.hstack((A, np.ones((len(A), 1))))

    B = eef_pose

    # split A and B into val:
    train_indices = np.random.choice(A.shape[0], (int(0.8 * A.shape[0]),))
    A_train, B_train = A[train_indices], B[train_indices]
    val_indices = np.delete(
        np.arange(A.shape[0]), np.random.choice(A.shape[0], (int(0.8 * A.shape[0]),))
    )
    A_val, B_val = A[~val_indices], B[~val_indices]

    res, resi = np.linalg.lstsq(A_train, B_train)[:2]
    print("train diff", np.mean((A_train @ res - B_train) ** 2))
    print("val_diff", np.mean((A_val @ res - B_val) ** 2))
    os.makedirs("homography_data/homography_transforms/", exist_ok=True)
    pickle.dump(res, open("homography_data/homography_transforms/" + out_file + ".pkl", "wb"))


if __name__ == "__main__":
    args = get_args()
    hydra.initialize(config_path="../../config/real/", job_name="calibration")
    cfg = hydra.compose(config_name='base')
    
    # disable all cameras except the one we are using
    if cfg.sensors:
        for sensor in cfg.sensors:
            if sensor not in (f'cam{args.cam_idx}',):
                cfg.sensors[sensor].enabled = False
    
    env = FrankaRealPSLEnv(args, cfg)
    neural_mp_env = IndustrealEnvWrapper(env)
    env.reset()

    # initialize apriltag detector
    detector = apriltag.Detector(
        families="tagStandard52h13", quad_decimate=1.0, quad_sigma=0.0, decode_sharpening=0.25
    )

    args = get_args()
    config = calibration_utils.get_perception_config(
        file_name=args.perception_config_file_name, module_name="calibrate_extrinsics"
    )
    if args.flip:
        env.spin_end_effector_180()

    steps = 3
    camera_name = args.cam_idx
    camera = f"cam{camera_name}"
    tag_length = config.tag.length
    tag_active_pixel_ratio = config.tag.active_pixel_ratio

    calibration_dir = f"homography_data/homography_april_tag/"
    os.makedirs(calibration_dir, exist_ok=True)

    ori_target = np.array([1.0, 0.0, 0.0, 0.0])
    if args.flip:
        ori_target = np.array([0.0, 1.0, 0.0, 0.0])
        
    fx, fy, cx, cy, ds, _ = get_cam_constants(cfg.sensors[f"cam{args.cam_idx}"].camera_cfg)
    intrinsics = {
        "cx": cx,
        "cy": cy,
        "fx": fx,
        "fy": fy,
    }

    collected_eef_pos = []
    collected_depth = []
    collected_tag_center_pixel = []
    detected_tag_num = 0
    for idx, delta in tqdm(enumerate(           # TODO: you might need to change the delta values based on position of the camera
        [
            np.array([1.5, 0, 0]), # forward
            np.array([0, -3.0, 0]), # right
            np.array([0, 0, -1.5]), #down
            np.array([0, 1.5, 0]), # left
            np.array([-1.5, 0, 0]), # back
            np.array([0, 0, 0.5]), # up
        ]
    )):
        obs = env.get_obs()
        for i in range(steps):
            if np.all(delta == np.array([0, 0, 0])):
                env.reset()
                continue
            delta_action = delta * 0.05

            pos_target = obs["eef_pos"] + delta_action
            env.execute_frankapy(np.array([*pos_target, *ori_target]), duration=5, use_libfranka_controller=False)
            target_ori_mat = R.from_quat(ori_target).as_matrix()
            
            curr_state = env.franka_arm.get_robot_state()
            curr_pos = curr_state["pose"].translation
            curr_ori_mat = curr_state["pose"].rotation
            control_utils.print_pose_error(curr_pos, curr_ori_mat, pos_target, target_ori_mat)
            obs = env.get_obs()

            image = obs[f"{camera}"][0]
            depth = obs[f"{camera}_depth"][0]

            (
                is_tag_detected,
                tag_pose_t,
                tag_pose_r,
                tag_center_pixel,
                tag_corner_pixels,
                tag_family,
            ) = calibration_utils.get_tag_pose_in_camera_frame(
                detector=detector,
                image=image,
                intrinsics=intrinsics,
                tag_length=tag_length,
                tag_active_pixel_ratio=tag_active_pixel_ratio,
            )
            if config.tag_detection.display_images:
                cv2.namedWindow(winname="RGB Output", flags=cv2.WINDOW_AUTOSIZE)
                cv2.imshow(winname="RGB Output", mat=image[:, :, ::-1])
                cv2.waitKey(delay=1000)
                cv2.destroyAllWindows()

            if is_tag_detected:
                collected_eef_pos.append(neural_mp_env.get_ee_pose()[:3])
                print(collected_eef_pos[-1])
                collected_depth.append(depth)
                collected_tag_center_pixel.append(tag_center_pixel)
                detected_tag_num += 1
                print(f"Has detected {detected_tag_num} tags")

                image_labeled = calibration_utils.label_tag_detection(
                    image=image, tag_corner_pixels=tag_corner_pixels, tag_family=tag_family
                )

                if config.tag_detection.display_images:
                    cv2.imshow("Tag Detection", image_labeled[:, :, ::-1])
                    cv2.waitKey(delay=1000)
                    cv2.destroyAllWindows()

    collected_eef_pos = np.array(collected_eef_pos)
    collected_depth = np.array(collected_depth)
    collected_tag_center_pixel = np.array(collected_tag_center_pixel)
    outfile_name = f"img{camera_name}_hom"
    compute_homography(
        collected_eef_pos,
        collected_depth,
        collected_tag_center_pixel,
        outfile_name,
        fx,
        fy,
        cx,
        cy,
        ds,
    )
    env.reset()
    
    print(f"Collected {detected_tag_num} tags in total.")
    env.close()
    