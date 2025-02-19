import numpy as np
import trimesh as tm
import torch

from itertools import product

# We might need to include some values derived from the sampled points in the observation
# To ensure that the observation is consistent, we isolate the randomness in this file
NUMPY_SEED = 3407

def sample_mesh_points(mesh, num_points):
    """Sample random points on the object mesh. 
    Args:
        mesh: trimesh.Trimesh
        num_points: number of points to sample
    Returns:
        points: sampled points
    """

    random_state = np.random.get_state()
    np.random.seed(NUMPY_SEED)
    
    prob = mesh.area_faces / mesh.area
    face_id = np.random.choice(mesh.faces.shape[0], size=(num_points,), p=prob)
    face_vertices = mesh.vertices[mesh.faces[face_id]]

    random_point = np.random.uniform(0, 1, size=(num_points, 2))
    flip = random_point.sum(axis=1) > 1.0
    random_point[flip] = 1 - random_point[flip]
    points = (
        face_vertices[:, 0]
        + (face_vertices[:, 1] - face_vertices[:, 0]) * random_point[:, 0:1]
        + (face_vertices[:, 2] - face_vertices[:, 0]) * random_point[:, 1:2]
    )

    np.random.set_state(random_state)

    return points

def sample_mesh_points_on_convex_hull(mesh, num_points, allow_vertices=True):
    """ Sample points on the convex hull of the object mesh.
    If the number of points is larger than the number of vertices in the convex hull, simply return the vertices.
    The sampled points are useful when computing the lowest point of the object in simulation.
    Args:
        mesh: trimesh.Trimesh
        num_points: maximum number of points to sample
        allow_vertices: if True, allow using the vertices of the convex hull as the sampled points when num_points is larger
    Returns:
        points: sampled points
    """
    convex_hull = mesh.convex_hull
    if not allow_vertices or num_points < convex_hull.vertices.shape[0]:
        points = sample_mesh_points(convex_hull, num_points)
    else:
        points = convex_hull.vertices
    return points

def sample_grasp_points(franka_grasp_data, mesh, num, uniform=True, finger_pos=False, rotation=False, threshold=0.01):
    """ Sample a subset of grasp points
    Args:
        franka_grasp_data: raw grasp data
        mesh: mesh of the object
        num: number of points to sample
        uniform: whether to make the sampling uniform in the space
        rotation: whether to return rotation of the gripper
        finger_pos: if True, return the finger positions instead of the grasp points
    Returns:
        sample_points: sampled grasp points (or finger positions)
    """

    random_state = np.random.get_state()
    np.random.seed(NUMPY_SEED)

    grasp_pos = franka_grasp_data["grasp_pos"]
    grasp_rot = franka_grasp_data["grasp_rot"]
    lf_pos = franka_grasp_data["lf_pos"]
    rf_pos = franka_grasp_data["rf_pos"]
    success_rate = (
        franka_grasp_data["success"] / franka_grasp_data["trials"]
    )

    # filter bad poses
    grasp_points = []
    grasp_rotations = []
    lf_points = []
    rf_points = []
    for i in range(len(success_rate)):
        if success_rate[i] > threshold:
            grasp_points.append(grasp_pos[i])
            grasp_rotations.append(grasp_rot[i])
            lf_points.append(lf_pos[i])
            rf_points.append(rf_pos[i])

    assert len(grasp_points) > 0

    grasp_points = np.vstack(grasp_points)
    grasp_rotations = np.vstack(grasp_rotations)
    lf_points = np.vstack(lf_points)
    rf_points = np.vstack(rf_points)

    num_points = len(grasp_points)
    if uniform:
        point_low, point_high = grasp_points.min(axis=0), grasp_points.max(axis=0)
        grid_size = 64
        dx = (point_high - point_low) / grid_size
        voxel_idx = ((grasp_points - point_low) / dx - 0.5).astype(int)
        assert (voxel_idx >= 0).all() and (voxel_idx < grid_size).all()
        voxel_idx = (
            voxel_idx[:, 0]
            + voxel_idx[:, 1] * grid_size
            + voxel_idx[:, 2] * grid_size * grid_size
        )

        voxel_cnt = np.zeros(grid_size**3)
        for i in range(num_points):
            voxel_cnt[voxel_idx[i]] += 1

        prob = 1 / voxel_cnt[voxel_idx]
        prob /= prob.sum()

        # sample points
        idxs = np.random.choice(num_points, min(num, num_points), p=prob, replace=False)
    else:
        idxs = np.random.choice(num_points, min(num, num_points), replace=False)

    if finger_pos:
        sample_points = np.hstack([lf_points[idxs], rf_points[idxs]])
        print(f"Sampled {len(idxs)} pairs of finger positions.")
    else:
        sample_points = grasp_points[idxs]
        if rotation:
            sample_points = np.hstack([sample_points, grasp_rotations[idxs]])
        print(f"Sampled {len(idxs)} grasp points.")

    np.random.set_state(random_state)

    return sample_points

def sample_mesh_keypoints(mesh, num_keypoints):
    """ Sample keypoints on the object mesh using iterative farthest point sampling
    Args:
        mesh: trimesh.Trimesh
        num_keypoints: number of keypoints to sample
    Returns:
        keypoints: sampled keypoints
    """

    points = mesh.vertices
    # ensure that the number of points is at least num_points
    while points.shape[0] < num_keypoints:
        points = np.vstack([points, sample_mesh_points(mesh, num_keypoints - points.shape[0])])

    # convert to torch tensor
    points = torch.from_numpy(points).float()
    if torch.cuda.is_available():
        points = points.cuda()
    
    # choose the farthest point from origin as the first keypoint
    dist = torch.norm(points, dim=-1)
    idx = torch.argmax(dist).item()
    selected = torch.zeros(len(points), dtype=torch.bool).to(points.device)
    selected[idx] = True

    while selected.sum() < num_keypoints:
        idx = find_farthest_point(points, selected)
        selected[idx] = True

    print(f"Sampled {num_keypoints} keypoints for observation.")
    return points[selected].cpu().numpy()


@torch.jit.script
def find_farthest_point(points, selected):
    # type: (Tensor, Tensor) -> Tensor
    """ Find the farthest point from the selected points
    Args:
        points: all points, including the selected points
        selected: index of selected points
    Returns:
        idx: index of the farthest point
    """

    num_points = points.size(0)
    num_selected = selected.sum()
    selected_points = points[selected]

    dist = torch.norm(
        selected_points.expand(num_points, num_selected, 3) - points.unsqueeze(1),
        dim=-1,
    )
    point_min_dist = torch.min(dist, dim=-1)[0]
    point_min_dist[selected] = -1e9
    idx = torch.argmax(point_min_dist)
    return idx

def get_keypoint_offsets_6d(device: torch.device):
    """Get keypoints for pose alignment comparison. Pose is aligned if all axis are aligned."""

    # order: origin, x, y, z, -x, -y, -z
    keypoint_corners = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    keypoint_corners = torch.tensor(keypoint_corners, device=device, dtype=torch.float32)
    keypoint_corners = torch.cat((keypoint_corners, -keypoint_corners[1:]), dim=0)  # use both negative and positive
    
    return keypoint_corners
