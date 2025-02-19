"""
Get device serial number and intrinsics matrix for RealSense cameras.
"""

import numpy as np
import pyrealsense2 as rs
from argparse import ArgumentParser

def get_camera_intrinsics(serial_number):
    # Initialize the RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable depth and color streams with the specified serial number
    config.enable_device(serial_number)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start the pipeline
    pipeline.start(config)

    try:
        # Wait for the first frame
        frames = pipeline.wait_for_frames()

        # Get the depth and color intrinsics
        depth_intrinsics = frames.get_depth_frame().profile.as_video_stream_profile().intrinsics

        # Construct the camera intrinsics matrix
        intrinsics_matrix = np.array(
            [
                [depth_intrinsics.fx, 0, depth_intrinsics.ppx],
                [0, depth_intrinsics.fy, depth_intrinsics.ppy],
                [0, 0, 1],
            ]
        )

    finally:
        # Stop the pipeline and release resources
        pipeline.stop()
    
    return intrinsics_matrix


if __name__ == "__main__":
    parser = ArgumentParser()
    args = parser.parse_args()
    device_ls = []
    for cam in rs.context().query_devices():
        device_ls.append(cam.get_info(rs.camera_info(1)))
    device_ls.sort()

    for i, device in enumerate(device_ls):
        try:
            intrinsics = get_camera_intrinsics(device)
            # print intrinsics matrix as proper numpy array with commas
            print()
            print(f"Device {i}: {device}")
            print(f"Intrinsics:\n", repr(intrinsics))
            print()
        except:
            print(f"Failed to get intrinsics for device {i}: {device}")
            continue
