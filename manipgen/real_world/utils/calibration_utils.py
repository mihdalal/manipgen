# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].
"""IndustRealLib: Perception utilities module.

This module defines utility functions for perceiving a scene with an Intel
RealSense camera.
"""

# Standard Library
import json
import os

# Third Party
import cv2
import numpy as np
from omegaconf import OmegaConf


def get_perception_config(file_name, module_name):
    """Gets an IndustRealLib perception configuration from a YAML file."""
    config = OmegaConf.load(os.path.join(os.path.dirname(__file__), "../..", "config", "real", file_name))[
        module_name
    ]

    return config


def label_tag_detection(image, tag_corner_pixels, tag_family):
    """Labels a tag detection on an image."""
    image_labeled = image.copy()

    corner_a = (int(tag_corner_pixels[0][0]), int(tag_corner_pixels[0][1]))
    corner_b = (int(tag_corner_pixels[1][0]), int(tag_corner_pixels[1][1]))
    corner_c = (int(tag_corner_pixels[2][0]), int(tag_corner_pixels[2][1]))
    corner_d = (int(tag_corner_pixels[3][0]), int(tag_corner_pixels[3][1]))

    # Draw oriented box on image
    cv2.line(img=image_labeled, pt1=corner_a, pt2=corner_b, color=(0, 255, 0), thickness=2)
    cv2.line(img=image_labeled, pt1=corner_b, pt2=corner_c, color=(0, 255, 0), thickness=2)
    cv2.line(img=image_labeled, pt1=corner_c, pt2=corner_d, color=(0, 255, 0), thickness=2)
    cv2.line(img=image_labeled, pt1=corner_d, pt2=corner_a, color=(0, 255, 0), thickness=2)

    # Draw tag family on image
    cv2.putText(
        img=image_labeled,
        text=tag_family.decode("utf-8"),
        org=(corner_a[0], corner_c[1] - 10),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        color=(255, 0, 0),
        thickness=2,
        lineType=cv2.LINE_AA,
    )

    return image_labeled


def get_tag_pose_in_camera_frame(detector, image, intrinsics, tag_length, tag_active_pixel_ratio):
    """Detects an AprilTag in an image. Gets the pose of the tag in the camera frame."""
    gray_image = cv2.cvtColor(src=image.astype(np.uint8), code=cv2.COLOR_BGR2GRAY)
    tag_active_length = tag_length * tag_active_pixel_ratio
    detection = detector.detect(
        img=gray_image,
        estimate_tag_pose=True,
        camera_params=[intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]],
        tag_size=tag_active_length,
    )

    if detection:
        is_detected = True
        pos = detection[0].pose_t.copy().squeeze()  # (3, )
        ori_mat = detection[0].pose_R.copy()
        center_pixel = detection[0].center
        corner_pixels = detection[0].corners
        family = detection[0].tag_family

    else:
        is_detected = False
        pos, ori_mat, center_pixel, corner_pixels, family = None, None, None, None, None

    return is_detected, pos, ori_mat, center_pixel, corner_pixels, family


def get_extrinsics(file_name):
    """Loads the extrinsics from a JSON file."""
    with open(os.path.join(os.path.dirname(__file__), "..", "io", file_name)) as f:
        json_obj = f.read()

    extrinsics_dict = json.loads(json_obj)

    return extrinsics_dict