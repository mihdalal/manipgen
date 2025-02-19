"""
manually fine tune the xyz shift of the cameras after Apriltag calibration
"""

import argparse

import hydra
import meshcat
import numpy as np
import open3d as o3d
import urchin
from robofin.robots import FrankaRobot

from manipgen.real_world.industreal_psl_env import FrankaRealPSLEnv
from manipgen.real_world.utils.neural_mp_env_wrapper import IndustrealEnvWrapper

NUM_ROBOT_POINTS = 4096
NUM_OBSTACLE_POINTS = 4096
NUM_TARGET_POINTS = 128
MAX_ROLLOUT_LENGTH = 150

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug-raw-pcd",
        action="store_true",
        help=("If set, will show visualization of raw pcd from each camera"),
    )
    parser.add_argument(
        "--debug-combined-pcd",
        action="store_true",
        help=("If set, will show visualization of the combined pcd"),
    )
    parser.add_argument(
        "-d",
        "--debug_mode",
        action="store_true",
        required=False,
        help="Enable output for debugging",
    )

    args = parser.parse_args()

    hydra.initialize(config_path="../../config/real/", job_name="calibration")
    cfg = hydra.compose(config_name='base')
    env = FrankaRealPSLEnv(args, cfg)
    neural_mp_env = IndustrealEnvWrapper(env)

    env.reset()

    cam_list = []
    cam_shift = {}
    pcd = {}
    rgb = {}

    for cam_key in env.hom.keys():
        cam_list.append(cam_key)
        cam_shift[cam_key] = env.hom[cam_key].pcs
        cam_idx = int(cam_key[-1])
        pcd[cam_key], rgb[cam_key] = env.get_raw_pcd_single_cam(
            cam_idx=cam_idx,
            filter_pcd=False,
            denoise=False,
            debug=args.debug_raw_pcd,
        )

    cpoints, ccolors = env.get_multi_cam_pcd(
        debug_raw_pcd=args.debug_raw_pcd,
        debug_combined_pcd=args.debug_combined_pcd,
        filter_pcd=False,
        denoise=True,
    )

    viz = meshcat.Visualizer()
    # Load the FK module
    urdf = urchin.URDF.load(FrankaRobot.urdf)
    proprio_config = neural_mp_env.get_joint_angles()

    # check pcd of the excluding area
    urdf_config = np.append(proprio_config, 0.04)
    meshes = urdf.visual_trimesh_fk(urdf_config[:8]).items()
    combined_mesh = None
    for mesh, trans in meshes:
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        transformed_vertices = (trans[:3, :3] @ vertices.T + trans[:3, 3:4]).T
        o3d_mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(transformed_vertices),
            triangles=o3d.utility.Vector3iVector(faces),
        )

        if combined_mesh is None:
            combined_mesh = o3d_mesh
        else:
            combined_mesh += o3d_mesh

    excluding_mesh_pcd = combined_mesh.sample_points_uniformly(number_of_points=10000)
    excluding_mesh_pcd = np.asarray(excluding_mesh_pcd.points)
    excluding_mesh_colors = np.zeros((3, excluding_mesh_pcd.shape[0]))
    excluding_mesh_colors[2, :] = 1

    # load the robot meshes
    for idx, (k, v) in enumerate(urdf.visual_trimesh_fk(np.append(proprio_config, 0.04)).items()):
        viz[f"robot/{idx}"].set_object(
            meshcat.geometry.TriangularMeshGeometry(k.vertices, k.faces),
            meshcat.geometry.MeshLambertMaterial(color=0xEEDD22, wireframe=False),
        )
        viz[f"robot/{idx}"].set_transform(v)
    for cam_key in env.hom.keys():
        viz[f"point_cloud_{cam_key}"].set_object(
            meshcat.geometry.PointCloud(
                position=pcd[cam_key].T,
                color=rgb[cam_key].T[::-1, :],
                size=0.005,
            )
        )

    while True:
        input_str = input(
            f"\ncalibrating xyz shift of the camera extrinsics, enter your input in the following format:\n1. to add 0.01 meter shift along the x axis (robot frame) of the point cloud capture by <camera_name>, enter 'camera_name x 0.01'\n2. the available cameras right now are {cam_list}\n3. enter 'e' to exit\ninput: "
        )
        if input_str == "e":
            break
        else:
            try:
                input_list = input_str.split()
                cam_key = input_list[0]
                axis = input_list[1]
                shift = float(input_list[2])
                if axis in ["x", "y", "z"]:
                    index = ["x", "y", "z"].index(axis)
                else:
                    continue

                cam_shift[cam_key][index] += shift
                pcd[cam_key][:, index] += shift
                viz[f"point_cloud_{cam_key}"].set_object(
                    meshcat.geometry.PointCloud(
                        position=pcd[cam_key].T,
                        color=rgb[cam_key].T[::-1, :],
                        size=0.005,
                    )
                )
            except:
                continue

    print("finalized mv_shift for all cameras:")
    for cam_key in env.hom.keys():
        print(f"{cam_key}: {cam_shift[cam_key].tolist()}")
    print("please update these camera mv_shifts to your camera configs")
    
    env.close()