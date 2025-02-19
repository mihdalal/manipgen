from isaacgym import gymtorch
from isaacgymenvs.utils.torch_jit_utils import (
    quat_mul,
    to_torch,
    quat_rotate,
    quat_from_euler_xyz,
    quat_conjugate,
    tf_combine,
)
from isaacgymenvs.utils.utils import set_seed
import isaacgymenvs.tasks.factory.factory_control as fc

import numpy as np
import torch
import trimesh as tm
import hydra

from manipgen.envs import environments

import os
import sys
import math
import time
from argparse import ArgumentParser


class GraspPoseSampler:
    def __init__(self, env):
        self.env = env
        self.gym = self.env.gym
        self.sim = self.env.sim
        self.device = self.env.device
        self.num_envs = self.env.num_envs

        franka_default_arm_dof_pos = torch.tensor(
            self.env.cfg_task.randomize.franka_arm_initial_dof_pos,
            device=self.device,
        ).repeat(self.num_envs, 1)
        franka_default_gripper_pos = torch.ones(self.num_envs, 2, device=self.device) \
            * self.env.asset_info_franka_table.franka_gripper_width_max / 2.0
        
        self.franka_default_dof_pos = torch.cat([franka_default_arm_dof_pos, franka_default_gripper_pos], dim=-1)

    def _load_rest_pose(self):
        # set franka states
        object_euler_xy = self.env.grasp_data[self.env.object_code][
            self.env.object_scale
        ]["object_euler_xy"]
        object_z = self.env.grasp_data[self.env.object_code][self.env.object_scale][
            "object_init_z"
        ]

        num_rest_samples = len(object_euler_xy)
        object_rest_pos = torch.zeros((num_rest_samples, 3), device=self.device)
        object_rest_pos[:, 0] = self.env.table_pose.p.x
        object_rest_pos[:, 1] = self.env.table_pose.p.y
        object_rest_pos[:, 2] = (
            object_z + self.env.cfg_base.env.table_height
        )
        object_rest_rot = quat_from_euler_xyz(
            object_euler_xy[:, 0],
            object_euler_xy[:, 1],
            torch.zeros(num_rest_samples, device=self.device),
        )

        print(f"Loaded {num_rest_samples} rest poses.")
        return object_rest_pos, object_rest_rot

    def _antipodal_sampling(self, num_samples):
        mesh = self.env.object_mesh
        
        # sample the first contact point on mesh
        prob = mesh.area_faces / mesh.area
        face_id = np.random.choice(mesh.faces.shape[0], size=(num_samples,), p=prob)
        face_vertices = mesh.vertices[mesh.faces[face_id]]

        normal = mesh.face_normals[face_id]

        random_point = np.random.uniform(0, 1, size=(num_samples, 2))
        flip = random_point.sum(axis=1) > 1.0
        random_point[flip] = 1 - random_point[flip]
        contact1 = (
            face_vertices[:, 0]
            + (face_vertices[:, 1] - face_vertices[:, 0]) * random_point[:, 0:1]
            + (face_vertices[:, 2] - face_vertices[:, 0]) * random_point[:, 1:2]
        )

        # sample a ray from the point, find intersection with mesh
        intersector = tm.ray.ray_triangle.RayMeshIntersector(mesh)
        contact2_list = []
        for id in range(num_samples):
            contact2 = contact1[id]
            for attempt in range(5):
                ray_origins = contact1[id : id + 1]
                ray_directions = -normal[id : id + 1]
                ray_directions += np.random.randn(1, 3) * 0.1
                locations, _, _ = intersector.intersects_location(
                    ray_origins, ray_directions, multiple_hits=True
                )

                locations = np.array(locations)
                if len(locations) > 1:
                    # find the farthest point
                    dist = np.linalg.norm(locations - ray_origins, axis=-1)
                    contact2 = locations[dist.argmax()]
                    break

            contact2_list.append(contact2)

        contact2 = np.array(contact2_list)

        # compute center of the two contact points
        center = (contact1 + contact2) * 0.5
        return torch.from_numpy(center).float().to(self.device)

    def _sample_grip_site_poses(
        self, grip_site_pos_local, object_rest_pos, object_rest_rot, num_samples
    ):
        # sample grip site position
        grip_site_pos = quat_rotate(
            object_rest_rot.repeat(len(grip_site_pos_local), 1), grip_site_pos_local
        ) + object_rest_pos.unsqueeze(0)

        # sample grip site rotation
        down_q = to_torch(
            num_samples * [1.0, 0.0, 0.0, 0.0], device=self.device
        ).reshape(num_samples, 4)

        rot1 = torch.zeros((num_samples, 4), device=self.device)
        theta_half = (
            (2 * torch.rand(num_samples, device=self.device) - 1) * math.pi * 0.25
        )
        rot1[:, 2] = torch.sin(theta_half)
        rot1[:, 3] = torch.cos(theta_half)

        rot2 = torch.zeros((num_samples, 4), device=self.device)
        p = torch.randn((num_samples, 3), device=self.device)
        p = p / p.norm(dim=-1, keepdim=True)
        theta_half = (
            (2 * torch.rand(num_samples, device=self.device) - 1) * math.pi * 0.125
        )
        rot2[:, :3] = p * torch.sin(theta_half).unsqueeze(-1)
        rot2[:, 3] = torch.cos(theta_half)

        grip_site_rot = quat_mul(rot1, down_q)
        grip_site_rot = quat_mul(rot2, grip_site_rot)

        return grip_site_pos, grip_site_rot

    def reset_states(self, object_rest_pos, object_rest_rot):
        # set object state
        self.env.object_pos[:, :] = object_rest_pos
        self.env.object_quat[:, :] = object_rest_rot
        self.env.object_linvel[:, :] = 0.0
        self.env.object_angvel[:, :] = 0.0

        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.env.root_state)
        )

        # set franka states
        self.env.dof_pos[:, :self.env.franka_num_dofs] = self.franka_default_dof_pos
        self.env.dof_vel[:, :] = 0.0
        self.env.dof_torque[:] = 0.0

        self.gym.set_dof_actuation_force_tensor(
            self.sim, gymtorch.unwrap_tensor(self.env.dof_torque)
        )
        self.gym.set_dof_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.env.dof_state)
        )

    def step(self, target_pos, target_rot, gripper):
        pos_error, axis_angle_error = fc.get_pose_error(
            fingertip_midpoint_pos=self.env.fingertip_centered_pos,
            fingertip_midpoint_quat=self.env.fingertip_centered_quat,
            ctrl_target_fingertip_midpoint_pos=target_pos,
            ctrl_target_fingertip_midpoint_quat=target_rot,
            jacobian_type=self.env.cfg_ctrl['jacobian_type'],
            rot_error_type='axis_angle')

        delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)
        gripper = torch.ones((self.num_envs, 2), device=self.device) * gripper
        gripper_target = torch.where(gripper == 1.0, 0.08, -0.04)

        self.env._apply_actions_as_ctrl_targets(
            actions=delta_hand_pose,
            ctrl_target_gripper_dof_pos=gripper_target,
            do_scale=True,
        )
        
        self.env.simulate_and_refresh()

    def take_steps(self, num_steps, target_pos, target_rot, gripper):
        for _ in range(num_steps):
            self.step(target_pos, target_rot, gripper)

    def record_state(self):
        object_pos = self.env.object_pos
        object_rot = self.env.object_quat
        grip_site_pos = self.env.fingertip_centered_pos
        grip_site_rot = self.env.fingertip_centered_quat
        lf_pos = self.env.left_fingertip_pos
        rf_pos = self.env.right_fingertip_pos
        hand_pos = self.env.hand_pos

        # record grip site pos and rot in object frame
        grasp_pos_local = quat_rotate(
            quat_conjugate(object_rot), grip_site_pos - object_pos
        )
        grasp_rot_local = quat_mul(quat_conjugate(object_rot), grip_site_rot)
        lf_pos_local = quat_rotate(quat_conjugate(object_rot), lf_pos - object_pos)
        rf_pos_local = quat_rotate(quat_conjugate(object_rot), rf_pos - object_pos)
        hand_pos_local = quat_rotate(quat_conjugate(object_rot), hand_pos - object_pos)

        return grasp_pos_local, grasp_rot_local, lf_pos_local, rf_pos_local, hand_pos_local

    def check_state(
        self, 
        grasp_pos_local, 
        object_height_min=0.20,
        grasp_pos_dist_max=0.20,
        gripper_width_min=0.002, 
        gripper_width_max=0.065,
    ):
        # object should be lifted up
        success = (
            self.env.object_pos[:, 2]
            > self.env.cfg_base.env.table_height + object_height_min
        )
        # grasp position should be close to object
        success &= grasp_pos_local.norm(dim=-1) < grasp_pos_dist_max

        # check contact
        contact = self.env.check_contact(self.env.franka_left_finger_ids_sim)
        contact &= self.env.check_contact(self.env.franka_right_finger_ids_sim)
        contact &= self.env.check_contact(self.env.object_rigid_body_ids_sim)
        success &= contact

        # gripper width should be within valid range
        success &= (self.env.gripper_dof_pos.sum(dim=-1) > gripper_width_min) & (self.env.gripper_dof_pos.sum(dim=-1) < gripper_width_max)

        return success

    def generate(
        self,
        attempt_iters=1,
        stop_grasp_num=1000,
        move_to_object_steps=60,
        close_gripper_steps=60,
        lift_up_steps=60,
        shake_steps=20,
        filter_threshold=0.01,
        object_height_min=0.20,
        grasp_pos_dist_max=0.20,
        gripper_width_min=0.002, 
        gripper_width_max=0.065,
    ):
        print(
            f"Processing object {self.env.object_code} at scale {self.env.object_scale:.2f}"
        )

        grip_site_pos_local = self._antipodal_sampling(self.num_envs)

        object_rest_pos_all, object_rest_rot_all = self._load_rest_pose()

        num_rest_poses = len(object_rest_pos_all)
        rec_grasp_pos, rec_grasp_rot, rec_success, rec_trials = [], [], [], []
        rec_lf_pos, rec_rf_pos, rec_hand_pos = [], [], []
        for pose_id in range(num_rest_poses):
            print(f"Sampling grasps for pose {pose_id}...")
            tot_success = 0
            grasp_pos_list, grasp_rot_list = [], []
            lf_pos_list, rf_pos_list, hand_pos_list = [], [], []
            for iter in range(attempt_iters):
                tik = time.time()

                object_rest_pos, object_rest_rot = (
                    object_rest_pos_all[pose_id],
                    object_rest_rot_all[pose_id],
                )

                grip_site_pos, grip_site_rot = self._sample_grip_site_poses(
                    grip_site_pos_local, object_rest_pos, object_rest_rot, self.num_envs
                )
                self.reset_states(
                    object_rest_pos.unsqueeze(0), object_rest_rot.unsqueeze(0)
                )

                # move to objet
                self.take_steps(move_to_object_steps, grip_site_pos, grip_site_rot, 1.0)

                # close gripper
                self.take_steps(close_gripper_steps, grip_site_pos, grip_site_rot, -1.0)

                # lift up
                grip_site_pos[:, 2] += 0.3
                self.take_steps(lift_up_steps, grip_site_pos, grip_site_rot, -1.0)

                # record state before shaking
                grasp_pos_local, grasp_rot_local, lf_pos_local, rf_pos_local, hand_pos_local = self.record_state()

                # add noise
                for _ in range(3):
                    rot = torch.zeros((self.num_envs, 4), device=self.device)
                    p = torch.randn((self.num_envs, 3), device=self.device)
                    p = p / p.norm(dim=-1, keepdim=True)
                    theta_half = (2 * torch.rand(self.num_envs, device=self.device) - 1) * math.pi / 4.0
                    rot[:, :3] = p * torch.sin(theta_half).unsqueeze(-1)
                    rot[:, 3] = torch.cos(theta_half)
                    # rot = torch.randn((self.num_envs, 4), device=self.device)
                    grip_site_rot = quat_mul(rot, grip_site_rot)
                    self.take_steps(shake_steps, grip_site_pos, grip_site_rot, -1.0)

                # check status and save valid grasps
                success = self.check_state(
                    grasp_pos_local, object_height_min, grasp_pos_dist_max, gripper_width_min, gripper_width_max
                )

                num_success = success.sum().item()
                tot_success += num_success

                if num_success > 0:
                    grasp_pos_list.append(grasp_pos_local[success])
                    grasp_rot_list.append(grasp_rot_local[success])
                    lf_pos_list.append(lf_pos_local[success])
                    rf_pos_list.append(rf_pos_local[success])
                    hand_pos_list.append(hand_pos_local[success])

                tok = time.time()
                print(
                    f"Iter {iter} ({tok - tik:.2f}s): {num_success} grasps found, total {tot_success} grasps."
                )

                if tot_success >= stop_grasp_num:
                    break

            if tot_success > 0:
                rec_grasp_pos.append(torch.cat(grasp_pos_list, dim=0).cpu().numpy())
                rec_grasp_rot.append(torch.cat(grasp_rot_list, dim=0).cpu().numpy())
                rec_lf_pos.append(torch.cat(lf_pos_list, dim=0).cpu().numpy())
                rec_rf_pos.append(torch.cat(rf_pos_list, dim=0).cpu().numpy())
                rec_hand_pos.append(torch.cat(hand_pos_list, dim=0).cpu().numpy())
            else:
                rec_grasp_pos.append(np.zeros((0, 3)))
                rec_grasp_rot.append(np.zeros((0, 4)))
                rec_lf_pos.append(np.zeros((0, 3)))
                rec_rf_pos.append(np.zeros((0, 3)))
                rec_hand_pos.append(np.zeros((0, 3)))
            rec_success.append(tot_success)
            rec_trials.append((iter + 1) * self.num_envs)
            sys.stdout.flush()

        rec_success = np.array(rec_success)
        rec_trials = np.array(rec_trials)
        object_euler_xy = self.env.grasp_data[self.env.object_code][self.env.object_scale]["object_euler_xy"].cpu().numpy()
        object_z = self.env.grasp_data[self.env.object_code][self.env.object_scale]["object_init_z"].cpu().numpy()
        grasp_data = {
            "grasp_pos": rec_grasp_pos,
            "grasp_rot": rec_grasp_rot,
            "lf_pos": rec_lf_pos,
            "rf_pos": rec_rf_pos,
            "hand_pos": rec_hand_pos,
            "success": rec_success,
            "trials": rec_trials,
            "object_euler_xy": object_euler_xy,
            "object_z": object_z,
        }
        

        save_path = os.path.join(
            self.env.asset_root,
            "unidexgrasp/graspdata",
            self.env.object_code.replace("/", "-")
            + f"-{int(100 * self.env.object_scale):03d}.npy",
        )
        np.save(save_path, grasp_data)

        success_rate = rec_success / rec_trials
        num_valid = (success_rate > filter_threshold).sum()

        print("Grasp data saved to", save_path)
        print(f"{num_valid} out of {num_rest_poses} poses have success rate > {filter_threshold:.3f}.")


def load_objects_and_scales(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    objects = []
    scales = []
    for line in lines:
        object, scale = line.split()
        objects.append(object)
        scales.append(float(scale))

    return objects, scales


def main(args):
    if not os.path.exists("assets/unidexgrasp/graspdata"):
        os.makedirs("assets/unidexgrasp/graspdata")

    if args.object_file != "":
        codes, scales = load_objects_and_scales(args.object_file)
    else:
        codes = [args.object_code]
        scales = [args.object_scale]

    if args.num_batch > 1:
        codes = [codes[i] for i in range(args.batch_id, len(codes), args.num_batch)]
        scales = [scales[i] for i in range(args.batch_id, len(scales), args.num_batch)]
    
    hydra.initialize(config_path="../config")

    for object_code, object_scale in zip(codes, scales):
        save_path = os.path.join(
            "assets/unidexgrasp/graspdata",
            object_code.replace("/", "-") + f"-{int(100 * object_scale):03d}.npy",
        )
        if os.path.exists(save_path) and not args.replace_existing:
            print(
                f"Grasp data for object {object_code} at scale {object_scale:.2f} already exists. Skipping..."
            )
            continue

        overrides = [
            "task=pick", 
            "sample_grasp_pose=True",
            f"device={args.device}",
            f"num_envs={args.num_envs}",
            f"render={args.render}",
            f"object_code={object_code}",
            f"object_scale={object_scale}",
            "task.rl.pos_action_scale=[0.2,0.2,0.2]",
            "task.rl.rot_action_scale=[0.2,0.2,0.2]",
            "task.ctrl.task_space_impedance.task_prop_gains=[1000,1000,1000,50,50,50]",
            "task.ctrl.task_space_impedance.task_deriv_gains=[63.2,63.2,63.2,1.41,1.41,1.41]",
            "task.env.enable_clutter_and_obstacle=False",
        ]
        cfg = hydra.compose(config_name="config_train", overrides=overrides)
        
        env = environments["pick"](
            cfg=cfg,
        )
        env.disable_automatic_reset = True

        sampler = GraspPoseSampler(env)
        sampler.generate(filter_threshold=args.filter_threshold)

        env.destroy()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--num_envs", type=int, default=4096, help="Number of environments."
    )
    parser.add_argument(
        "--render", action="store_true", help="Whether to set up viewer."
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to use."
    )
    parser.add_argument(
        "--object_code",
        type=str,
        default="core/bottle-b13f6dc78d904e5c30612f5c0ef21eb8",
        help="Object code in unidexgrasp dataset.",
    )
    parser.add_argument(
        "--object_scale",
        type=float,
        default=0.1,
        help="Object scale in unidexgrasp dataset.",
    )
    parser.add_argument(
        "--object_file",
        type=str,
        default="",
        help="Path to subset of unidexgrasp dataset.",
    )
    parser.add_argument(
        "--replace_existing",
        action="store_true",
        help="Whether to replace existing grasp data.",
    )
    parser.add_argument(
        "--filter_threshold",
        type=float,
        default=0.01,
        help="Threshold for filtering invalid poses.",
    )
    parser.add_argument(
        "--num_batch",
        type=int,
        default=1,
        help="Number of batch for generating grasp data.",
    )
    parser.add_argument(
        "--batch_id",
        type=int,
        default=0,
        help="Batch id for generating grasp data.",
    )
    args = parser.parse_args()

    set_seed(0)
    main(args)
