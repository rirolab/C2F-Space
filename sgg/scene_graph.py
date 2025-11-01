import os
from torchvision.ops import nms
import torch
import cv2
# Grounding DINO
from huggingface_hub import hf_hub_download
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import annotate, load_image, predict
import numpy as np
# segment anything
from segment_anything import SamPredictor, sam_model_registry


class SceneGraphWithGDino():
    def __init__(
            self,
            groundingDino_rootpath='./sgg/Grounded-Segment-Anything',
            TEXT_PROMPT="objects",
            device='cuda:0'
            ):

        self.groundingdino_model = self.load_model_hf("ShilongLiu/GroundingDINO", "groundingdino_swinb_cogcoor.pth", "GroundingDINO_SwinB.cfg.py",
                                                      device=device)
        self.sam = sam_model_registry["vit_b"](checkpoint=os.path.join(groundingDino_rootpath, 'segment_anything/sam_vit_b_01ec64.pth'))
        self.sam.to(device=device)
        self.sam_predictor = SamPredictor(self.sam)
        self.TEXT_PROMPT=TEXT_PROMPT
        
        self.boxes = None
        self.scene_graph = None
    
    def load_model_hf(self, repo_id, filename, ckpt_config_filename, device='cuda:0'):
        """Function to load the necessary models.

        Args:
            repo_id (string): Name of the githubb repo id.
            filename (string): Cache file name
            ckpt_config_filename (_type_): Name of the checkpoint config file.
            device (str, optional): Which cuda to assign. Defaults to 'cuda:0'.

        Returns:
            _type_: Returns the model.
        """
        
        cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)
        args = SLConfig.fromfile(cache_config_file) 
        model = build_model(args)
        args.device = device

        cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
        checkpoint = torch.load(cache_file, map_location='cpu')
        log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
        _ = model.eval()
        
        return model 

    def run_dino(self, image, box_threshold=0.2, text_threshold=0.2, text_prompts=['object', 'objects']):
        image = np.array(image)[..., ::-1]  # BGR to RGB
        cv2.imwrite("temp.png", image)
        image_source, image = load_image("temp.png")
        text_prompt = ".".join(text_prompts)
        # print(text_prompt)
        boxes, logits, phrases = predict(
            model=self.groundingdino_model,
            image=image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        H, W, _ = image_source.shape
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
        generic_prompts = ["object", "objects", "object objects", "objects object"]
        unique_phrases = set(phrases)
        # print(unique_phrases)
        if unique_phrases.issubset(set(generic_prompts)):
            nms_idx = nms(
                boxes=boxes_xyxy,
                scores=logits,
                iou_threshold=0.6
            )
        else:
            nms_idx = []
            for phrase in unique_phrases:
                idxs = [i for i, p in enumerate(phrases) if p == phrase]
                if len(idxs) > 1:
                    b = boxes_xyxy[idxs]
                    s = logits[idxs]
                    nms_idx_sub = nms(
                        boxes=b,
                        score=s,
                        iou_threshold=0.4
                    )
                    nms_idx.extend([idxs[i] for i in nms_idx_sub])
                else:
                    nms_idx.extend(idxs[0])
        boxes = boxes[nms_idx]
        boxes_xyxy = boxes_xyxy[nms_idx]
        logits = logits[nms_idx]
        phrases = [phrases[i] for i in nms_idx]
        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits,
                                      phrases=phrases)
        cv2.imwrite("temp_annotated.png", annotated_frame)  # RGB to BGR
        return boxes_xyxy.tolist(), logits.tolist(), phrases, annotated_frame

    def run_sam(self, image, boxes_xyxy):
        image = np.array(image)[..., ::-1]  # BGR to RGB
        cv2.imwrite("temp.png", image)
        image_source, image = load_image("temp.png")
        boxes_xyxy_tensor = torch.Tensor(boxes_xyxy)
        self.sam_predictor.set_image(image_source)
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_xyxy_tensor, image_source.shape[:2]).to(self.sam_predictor.device)
        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        return masks.tolist()

    def run(self, image):
        # detect boxes
        boxes, logits, phrases, annotated_frame = self.run_dino(image, text_prompts=["object", "objects"])
        # then run sam for all
        masks = self.run_sam(image, boxes)
        # need to generate dict
        scene_graph = {}
        for i in range(len(boxes)):
            scene_graph[i] = {
                "bbox": boxes[i],
                "mask": masks[i]
            }
        self.sg = scene_graph
        return scene_graph

