import os
import sys
import random
import time
import torch
import torchvision
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
import groundingdino.datasets.transforms as T
sys.path.append("Grounded-Segment-Anything/segment_anything")
from segment_anything import SamPredictor, build_sam

def load_grounding_dino_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    model.eval()
    return model

def load_sam_model(model_checkpoint_path, device):
    model = SamPredictor(build_sam(checkpoint=model_checkpoint_path).to(device))
    return model

def convert_tag_to_caption(tag):
    if type(tag) is list:
        caption = ('. '.join(tag)).lower()
    else:
        caption = tag.lower().strip().replace(",", ".")
    # caption is a comma separated string of tags, remove "robot arm" from the tags
    caption = caption.replace("robot arm, ", "")
    caption = caption.replace("orange robotic arm, ", "")
    # add coco classes to the caption
    # caption = caption + " " + ". ".join(COCO_CLASSES)
    print("GroundingDINO caption:", caption)
    return caption

def draw_mask(mask, draw, random_color=False):
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 153) if random_color else (30, 144, 255, 153)
    nonzero_coords = np.transpose(np.nonzero(mask))
    for coord in nonzero_coords:
        draw.point(coord[::-1], fill=color)

def draw_box(box, draw, label, w, h):
    color = tuple(np.random.randint(0, 255, size=3).tolist())
    line_width = int(max(4, min(20, 0.006 * max(draw.im.size))))
    draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline=color, width=line_width)
    if label:
        font_path = os.path.join(cv2.__path__[0], 'qt', 'fonts', 'DejaVuSans.ttf')
        font_size = int(max(12, min(60, 0.02 * max(draw.im.size))))
        font = ImageFont.truetype(font_path, size=font_size)
        bbox = draw.textbbox((box[0], box[1]), str(label), font) if hasattr(font, "getbbox") else (box[0], box[1], w + box[0], box[1] + h)
        draw.rectangle(bbox, fill=color)
        draw.text((box[0], box[1]), str(label), fill="white", font=font)

def get_grounding_output(model, image, caption, box_threshold, text_threshold, device="cpu"):
    """Get the grounding output from GroundingDINO"""
    caption = convert_tag_to_caption(caption)
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    # filter out low confidence boxes
    logits = outputs["pred_logits"].cpu().sigmoid()[0]
    boxes = outputs["pred_boxes"].cpu()[0]
    filt_mask = logits.max(dim=1)[0] > box_threshold
    logits_filt = logits[filt_mask]
    boxes_filt = boxes[filt_mask]
    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:6]})")
        scores.append(logit.max().item())
    model.to("cpu")
    return boxes_filt, torch.Tensor(scores), pred_phrases

@torch.no_grad()
def get_segmentation(grounding_dino_model, sam_model, raw_image, tags, out_path=None, box_threshold=0.20, text_threshold=0.10, iou_threshold=0.20, device="cpu"):
    """Get segmentation masks for the objects in the image"""
    # transform image
    t = time.time()
    raw_image = raw_image.convert("RGB")
    transform = T.Compose([T.RandomResize([800], max_size=1333), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    image, _ = transform(raw_image, None)
    
    # GroundingDINO
    boxes_filt, scores, pred_phrases = get_grounding_output(
        grounding_dino_model, 
        image, tags, 
        box_threshold, text_threshold, 
        device=device
    )
    print("Grounding dino time:", time.time() - t)
    print("GroundingDINO finished")
    
    # SAM
    t = time.time()
    image = np.asarray(raw_image)
    sam_model.set_image(image)
    size = raw_image.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]
    boxes_filt = boxes_filt.cpu()
    print(f"Before NMS: {boxes_filt.shape[0]} boxes")
    nms_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
    boxes_filt = boxes_filt[nms_idx]
    pred_phrases = [pred_phrases[idx] for idx in nms_idx]
    print(f"After NMS: {boxes_filt.shape[0]} boxes")
    transformed_boxes = sam_model.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)
    masks, _, _ = sam_model.predict_torch(point_coords=None, point_labels=None, boxes=transformed_boxes.to(device), multimask_output=False)
    print("SAM time:", time.time() - t)
    print("SAM finished")
    
    t = time.time()
    if out_path is not None:
        mask_image = Image.new('RGBA', size, color=(0, 0, 0, 0))
        mask_draw = ImageDraw.Draw(mask_image)
        w, h = raw_image.size
        image_draw = ImageDraw.Draw(raw_image)
        out_dict = {}
        for box, label, mask in zip(boxes_filt, pred_phrases, masks):
            draw_box(box, image_draw, label, w, h)
            out_dict[label] = box.tolist()
            draw_mask(mask[0].cpu().numpy(), mask_draw, random_color=True)
        out_image = raw_image.convert('RGBA')
        out_image.alpha_composite(mask_image)

        # save to disk
        if out_path is not None:
            out_image.save(out_path)
        
        # create separate images for each mask
        mask_images = []
        for box, label, mask in zip(boxes_filt, pred_phrases, masks):
            mask_image = Image.new('RGBA', size, color=(0, 0, 0, 0))
            mask_draw = ImageDraw.Draw(mask_image)
            draw_box(box, mask_draw, label, w, h)
            draw_mask(mask[0].cpu().numpy(), mask_draw, random_color=True)
            mask_images.append(np.array(mask_image.convert('RGBA')))
    else:
        out_image = raw_image
        out_dict = {}
        mask_images = []
    print("Drawing time:", time.time() - t)
    return out_image, out_dict, pred_phrases, masks, mask_images
