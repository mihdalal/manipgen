import os
import time
import traceback
import cv2
import numpy as np
import torch
from PIL import Image
from manipgen.real_world.vlm_planning.segmentation_utils import (
    load_grounding_dino_model,
    load_sam_model,
    get_segmentation,
)
from manipgen.real_world.vlm_planning.gpt_utils import (
    prompt_gpt4v,
)
from manipgen.real_world.vlm_planning.target_pose_utils import (
    backtrack_eef_position,
    offset_eef_position,
    estimate_quat_from_object_points,
    get_candidate_eef_quat_from_handle_points,
    sample_collision_free_eef_pose,
    get_candidate_eef_quat_for_tightspace,
    TargetPoseChecker,
)
import open3d as o3d

class VLMPlanner:
    def __init__(self, env, cam_ids):
        """
        Initializes the VLMPlanner class with the environment and planners

        Args:
            env (FrankaRealPSLEnv): The environment object
            cam_ids (list): The camera ids to use for VLM planning.
        """
        # use all cameras for point cloud
        # use selected cameras for VLM
        self.cam_ids = cam_ids
        
        self.root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
        self.config_file = os.path.join(self.root_dir, "Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
        self.grounded_checkpoint = os.path.join(self.root_dir, "Grounded-Segment-Anything/groundingdino_swint_ogc.pth")
        self.sam_checkpoint = os.path.join(self.root_dir, "Grounded-Segment-Anything/sam_vit_h_4b8939.pth")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        self.grounding_dino_model = load_grounding_dino_model(self.config_file, self.grounded_checkpoint, self.device)
        self.sam_model = load_sam_model(self.sam_checkpoint, self.device)
        self.env = env
        self.target_pose_checker = TargetPoseChecker(self.env, use_robot_sdf=True)

        # set up log dir
        self.log_dir = os.path.join(self.root_dir, "manipgen/real_world/manipgen_logs")
        if os.path.exists(self.log_dir):
            os.system(f"rm -rf {self.log_dir}")
        os.makedirs(self.log_dir)

    def get_vlm_images(self, obs):
        """Get images from the cameras"""
        ims = []
        ids = []
        for cam_key in self.env.hom.keys():
            cam_id = int(cam_key[3:])
            ims.append(obs[f"{cam_key}"][0])
            ids.append(cam_id)
        return ims, ids

    @torch.no_grad()
    def get_segmentation(self, raw_image, tags, out_path, box_threshold=0.20, text_threshold=0.10, iou_threshold=0.20):
        """Get segmentation masks for the objects in the image"""
        out_image, out_dict, pred_phrases, masks, mask_images = get_segmentation(
            self.grounding_dino_model, self.sam_model,
            raw_image, tags, out_path, box_threshold, text_threshold, iou_threshold, self.device
        )
        return out_image, out_dict, pred_phrases, masks, mask_images
    
    def get_segmentation_for_target_pose_estimation(self, raw_image, tags, skill, obj, cam_key, debug=True):
        """Get segmentation masks of the target object for target pose estimation"""
        _, _, pred_phrases, masks, mask_images = self.get_segmentation(
            raw_image, tags,
            os.path.join(self.log_dir, f"{obj}_{cam_key}.png" if debug else None),
        )

        best_id = 0
        best_prob = -1.0
        print("[Target Pose Tags]:", tags)
        print("[Target Pose Pred Phrases]:", pred_phrases)
        base_pred_phrases = []
        base_ids = []
        for idx, pred_phrase in enumerate(pred_phrases):
            # extract probablity in ()
            prob = float(pred_phrase.split("(")[1].split(")")[0])
            # remove prob from phrase
            pred_phrase = pred_phrase.split("(")[0].strip().split(".")[0]
            if "robot" in pred_phrase:
                continue
            
            if any([word in pred_phrase for word in obj.split()]):
                base_pred_phrases.append(pred_phrase)
                base_ids.append(idx)
                
                if prob > best_prob and "robot" not in pred_phrase:
                    best_prob = prob
                    best_id = idx
                
        if len(base_ids) == 0:
            print(f"Warning: no base pred phrases match '{obj}'!")
            return None
        
        print(f"Getting segmentation for {obj}")
        if len(base_pred_phrases) > 1:
            # Grounded-SAM is not good at language grounding, so we ask GPT to pick the best phrase
            print("Asking GPT to pick the best phrase...")
            input_text = "(" + skill + ", " + obj + ", " + ", ".join(pred_phrases) + ")"
            tagging_image = cv2.imread(os.path.join(self.log_dir, "sam_out_0.png"))[:, :, ::-1]
            pred_phrase = prompt_gpt4v(text=input_text, ims=[np.array(raw_image), *mask_images, tagging_image], agent_type='segmentation_doublechecker')
            best_id = pred_phrases.index(pred_phrase)
        else:
            best_id = base_ids[0]
        print(f"Segmentation result for '{obj}':", pred_phrases[best_id])
        
        masks = masks.cpu().numpy()[best_id][0]
        
        return masks
    
    def plan(self, task_prompt, previous_plan=None, previous_tags=None):
        """
        Plan the task using VLM. Return a plan of the form [(object, skill, is_tabletop), ...].
        
        Args:
            task_prompt (str): The task prompt.
            previous_plan (str, optional): The previous plan. Defaults to None.
            previous_tags (list, optional): The previous tags. Defaults to None.
        
        Returns:
            Tuple[list, list]: The plan and the tags.
        """
        # get images from the cameras
        obs = self.env.get_obs()
        ims, cam_ids = self.get_vlm_images(obs)
        if previous_tags is None:
            tags = prompt_gpt4v(text=None, ims=ims, agent_type='tagging')
        else:
            tags = previous_tags

        # get segmentation masks for the objects in the images
        segs = []
        for idx, (im, cam_id) in enumerate(zip(ims, cam_ids)):
            if self.cam_ids is not None and cam_id not in self.cam_ids:
                continue
            out = self.get_segmentation(Image.fromarray(im), tags, out_path=os.path.join(self.log_dir, "sam_out_0.png"))
            
            segs.append(np.array(out[0]))
        
        # get plan from VLM
        if previous_plan is not None:
            text_prompt = "Task: " + task_prompt + ". Previously Predicted Plan: " + str(previous_plan)
        else:
            text_prompt = "Task: " + task_prompt
        text_prompt = "Detected Tags: " + str(tags) + ". " + text_prompt
        plan = prompt_gpt4v(text=text_prompt, ims=ims + segs, agent_type='planning')
        return plan, tags
    
    def add_umi_finger_length_in_target_pose(self, target_pose):
        """
        frankapy takes center of the gripper as eef position.
        Neural MP takes base of the palm as eef position.
        This function converts the Neural MP target pose to frankapy target pose.
        """
        target_pos, target_quat = target_pose[:3], target_pose[3:]
        target_pos = backtrack_eef_position(target_pos, target_quat, -0.1993)
        target_pose = np.concatenate([target_pos, target_quat])
        return target_pose
    
    def estimate_target_pose(self, obj, skill, is_table_top=True, visualize=False, debug=False):
        """Estimate initial pose for local policies"""
        # TODO: need to find better prompt for handle
        if (skill == "place" or skill == "open" or skill == "close"):
            if ("drawer" in obj or "door" in obj) and "handle" not in obj:
                obj = obj + " handle"
        
        caption = obj + ". robot arm."

        # construct point cloud from all cameras
        points_combined = []                        # object points: estimate target pose
        points_combined_scene = []                  # scene points: detect collision
        points_combined_scene_colors = []
        masks_all = []
        
        obs = self.env.get_obs()
        t = time.time()
        for cam_key in self.env.hom.keys():
            cam_id = int(cam_key[3:])
            depth_numpy = obs[f"{cam_key}_depth"][0]
            img_numpy = obs[f"{cam_key}"][0]
            img_pil = Image.fromarray(img_numpy)
            colors = None
            try:
                if self.cam_ids is None or cam_id in self.cam_ids:
                    masks = self.get_segmentation_for_target_pose_estimation(img_pil, caption, skill, obj, cam_key, debug)
                    if masks is None:
                        continue
                    masks_all.append(masks)
                    
                    pc = self.env.hom[cam_key].get_pointcloud(depth_numpy)
                    masked_pc = pc[masks]
                    masked_img = img_numpy[masks] / 255.0

                    points, colors = self.env.hom[cam_key].get_filtered_pc(masked_pc.reshape(-1, 3), masked_img.reshape(-1, 3))
                    points, colors = self.env.hom[cam_key].denoise_pc(points, colors)
                    points_combined.append(points)
                points, colors = self.env.hom[cam_key].get_pointcloud(depth_numpy, img_numpy)
                points_combined_scene.append(points.reshape(-1, 3))
                points_combined_scene_colors.append(colors.reshape(-1, 3))
            except:
                print(traceback.format_exc())
        print("Time taken for segmentation: ", time.time() - t)
        if len(points_combined) == 0:
            print("Warning! No segmentation mask found for target pose estimation!")
            points_combined = points_combined_scene
        points = np.concatenate(points_combined, axis=0)
        scene_points = np.concatenate(points_combined_scene, axis=0)
        scene_colors = np.concatenate(points_combined_scene_colors, axis=0)
        scene_points, scene_colors = self.env.hom[cam_key].get_filtered_pc(scene_points, scene_colors)
        downsample_indices = np.random.choice(scene_points.shape[0], min(200000, scene_points.shape[0]), replace=False)
        scene_points, scene_colors = scene_points[downsample_indices], scene_colors[downsample_indices] / 255.
        scene_points, scene_colors = self.env.hom[cam_key].denoise_pc(scene_points, scene_colors)

        # estimate target pose
        object_pos = np.mean(np.asarray(points), axis=0).copy()
        if skill in ("pick", "place"):
            print("Is Table Top:", is_table_top)        
            if is_table_top:
                # no obstacle above the object: point downwards
                target_pos = object_pos + np.array([0.0, 0.0, 0.26])
                if skill == "pick":
                    target_quat = estimate_quat_from_object_points(points)
                    if ("cup" in obj.split()) or ("bowl" in obj.split()) or ("mug" in obj.split()):
                        target_pos = offset_eef_position(target_pos, target_quat, np.array([0.0, -0.03, 0.0]))
                    else:
                        target_pos = offset_eef_position(target_pos, target_quat, np.array([-0.03, 0.0, 0.0]))      # get a better view of the object
                else:
                    target_quat = np.array([1.0, 0.0, 0.0, 0.0])                    
            else:
                # tight-space pick and place: reach the region from the side
                offset = np.array([0.0, 0.0, 0.08]) if skill == "place" else np.array([0.0, 0.0, 0.0])
                candidate_eef_quat = get_candidate_eef_quat_for_tightspace()
                target_pos, target_quat = sample_collision_free_eef_pose(object_pos + offset, candidate_eef_quat, scene_points, scene_colors, self.target_pose_checker, diffuse_pc=True)
        
            # special case for placing in drawer
            if skill == "place" and ("drawer" in obj or "handle" in obj):
                # step 1: assume we want to grasping handle, estimate target pose
                candidate_eef_quat = get_candidate_eef_quat_from_handle_points(points, masks_all, num_samples=5)
                target_pos, target_quat = sample_collision_free_eef_pose(object_pos, candidate_eef_quat, scene_points, scene_colors, self.target_pose_checker, diffuse_pc=True)
                # step 2: get position of the handle
                target_pos = self.add_umi_finger_length_in_target_pose(np.concatenate([target_pos, target_quat]))[:3]
                # step 3: move inside the drawer
                target_pos = backtrack_eef_position(target_pos, target_quat, offset=-0.15)
                # step 4: point downwards to place, also add offset to height to avoid collision
                target_quat = np.array([1.0, 0.0, 0.0, 0.0])
                target_pos = target_pos + np.array([0.0, 0.0, 0.38])
        
        elif skill in ("open", "close"):
            candidate_eef_quat = get_candidate_eef_quat_from_handle_points(points, masks_all)
            target_pos, target_quat = sample_collision_free_eef_pose(object_pos, candidate_eef_quat, scene_points, scene_colors, self.target_pose_checker, diffuse_pc=True)
        else:
            raise ValueError(f"Skill {skill} not supported for target pose estimation.")
        
        if visualize:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.visualization.draw_geometries([pcd])
        return np.concatenate([target_pos, target_quat]), (scene_points, scene_colors)
    
    def get_local_seg_mask(self, text_prompt, skill="pick"):
        """Get local segmentation mask for local observation"""
        text_prompt = text_prompt + "."
        if skill == "pick":
            obs = self.env.get_obs(wrist_cam_only=True)
            img_numpy = obs["cam1"][0]
            # mask out grippers
            img_numpy[-150:, :250, :] = 0
            img_numpy[-150:, -210:, :] = 0
            img_pil = Image.fromarray(img_numpy)
            out_image, out_dict, pred_phrases, masks, _ = self.get_segmentation(
                img_pil, 
                text_prompt,
                os.path.join(self.log_dir, "local_obs.png"),
                box_threshold = 0.3,
                text_threshold = 0.25
            )
            cfg = self.env.cfg.perception_config
            if len(pred_phrases) > 0:
                best_id = 0
                best_prob = -1.0
                for idx, pred_phrase in enumerate(pred_phrases):
                    print(idx, pred_phrase)
                    # extract probablity in ()
                    prob = float(pred_phrase.split("(")[1].split(")")[0])
                    if prob > best_prob:
                        best_prob = prob
                        best_id = idx

                mask = masks.cpu().numpy()[best_id][0]
                # crop to policy view: 480 x 640 -> 368 x 368 (before center cropping)
                h_offset = cfg.process_depth_h_offset
                h_boarder = (cfg.raw_depth_h - cfg.obs_depth_w) // 2
                w_offset = cfg.process_depth_w_offset
                w_boarder = (cfg.raw_depth_w - cfg.obs_depth_w) // 2
                mask = mask[h_boarder+h_offset:-h_boarder+h_offset, w_boarder+w_offset:-w_boarder+w_offset]
                # resize to 84 x 84
                mask = (mask > 0.5).astype(np.float32)
                mask = cv2.resize(mask, (cfg.obs_depth_w_down, cfg.obs_depth_w_down), interpolation=cv2.INTER_NEAREST)
            else:
                print("Warning: No segmentation mask found for local observation!")
                mask = np.zeros((cfg.obs_depth_w_down, cfg.obs_depth_w_down), dtype=np.float32)
        else:
            mask = None
        return mask
