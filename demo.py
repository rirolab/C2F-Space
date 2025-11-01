import argparse, os
from PIL import Image
import random
from networkx import center
import json

import numpy as np
import torch
from tqdm import tqdm
import cv2
import shutil

from scene_generation.environments.environment import Environment
from scene_generation import tasks
import scene_generation.utils.general_utils as utils
import random
from coarse_vlm.src.core import spatial_reasoning
from scipy.signal import correlate2d
from fine_refinement.demo import DemoGPS
from torch_geometric.graphgym.config import cfg

random.seed(0)
np.random.seed(0)

class LINGOSpaceInference:
    def __init__(self, ckpt_folder="coarse_vlm/checkpoints"):
        self.spatial_reasoner = spatial_reasoning.SpatialReasoner()
        self.dataset_folder = os.path.join("coarse_vlm", self.spatial_reasoner.dataset_folder)
        self.output_folder = os.path.join("coarse_vlm", self.spatial_reasoner.output_folder)
        shutil.rmtree(self.dataset_folder, ignore_errors=True)
        shutil.rmtree(self.output_folder, ignore_errors=True)
        os.makedirs(os.path.join(self.dataset_folder, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.dataset_folder, "scene_graph"), exist_ok=True)
        os.makedirs(self.output_folder, exist_ok=True)
        with open(os.path.join(self.dataset_folder, "info.json"), "w") as f:
            json.dump({}, f)
        # Set superpixel model
        self.superpixel_model = DemoGPS(
            cfg=cfg,
            pretrained_dir=ckpt_folder,
            ckpt_idx=99
        )

    def find_best_placement(self, allowed_mask, object_mask, angles):
        def rotate_with_padding(mask, angle):
            h, w = mask.shape
            diag = int(np.ceil(np.sqrt(h**2 + w**2)))
            pad_y = (diag - h) // 2
            pad_x = (diag - w) // 2

            padded = np.pad(mask, ((pad_y, diag - h - pad_y), (pad_x, diag - w - pad_x)), mode='constant')
            center = (padded.shape[1] // 2, padded.shape[0] // 2)
            rot = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                padded, rot, (padded.shape[1], padded.shape[0]),
                flags=cv2.INTER_NEAREST, borderValue=0
            )
            return rotated, pad_y, pad_x

        best_score = -np.inf
        best_center = None
        best_angle = None

        for angle in angles:
            rot_obj, pad_y, pad_x = rotate_with_padding(object_mask, angle)
            orig_mask = np.copy(rot_obj)
            orig_mask = orig_mask.astype(np.uint8)
            rot_obj_h, rot_obj_w = rot_obj.shape

            allowed_mask = allowed_mask.astype(np.float32)
            rot_obj = rot_obj.astype(np.float32)
            inside_score = correlate2d(allowed_mask, rot_obj, mode='valid')
            score_map = inside_score
            idx = np.unravel_index(np.argmax(score_map), score_map.shape)
            peak_y, peak_x = idx

            score = score_map[idx]
            if score > best_score:
                best_score = score
                best_angle = angle

                center_y = peak_y + rot_obj_h // 2
                center_x = peak_x + rot_obj_w // 2
                best_center = (center_x, center_y)
                new_mask = np.zeros_like(allowed_mask)
                new_mask[peak_y:peak_y+rot_obj_h, peak_x:peak_x+rot_obj_w] = orig_mask
                new_mask = np.where(new_mask != 0, 255, 0)
                new_mask = new_mask.astype(np.uint8)
        return best_center, best_angle, new_mask



    def run(self, image, lang_goal, scene_graph=None, pick_mask=None):
        img_idx = len(os.listdir(os.path.join(self.dataset_folder, "images")))
        cv2.imwrite(os.path.join(self.dataset_folder, "images", f"{img_idx}.png"), np.array(image)[:, :, ::-1])
        with open(os.path.join(self.dataset_folder, "info.json"), "r") as f:
            ori_info = json.load(f)
        ori_info[f"{img_idx}"] = {"image_path": f"{img_idx}.png", "instruction": lang_goal}
        with open(os.path.join(self.dataset_folder, "info.json"), "w") as f:
            json.dump(ori_info, f, indent=4)
        # generate scene graph
        with open(os.path.join(self.dataset_folder, "scene_graph", f"{img_idx}.json"), "w") as f:
            json.dump({"objects_info": scene_graph}, f, indent=4)

        self.spatial_reasoner.run()

        with open(os.path.join(self.output_folder, "result_json.json"), "r") as f:
            result_info = json.load(f)
        # find result
        result = result_info.get(f"{img_idx}", None)
        if result is not None:
            result_img = np.zeros(np.array(image).shape[:2], dtype=np.uint8)
            for i in range(len(result["Center Coordinates"])):
                center_x, center_y = result["Center Coordinates"][i]
                r0, r1 = result["Axes Length"][i]
                angle = result["Angle"][i]
                cv2.ellipse(result_img, (center_x, center_y), (round(r0), round(r1)), angle, 0, 360, 255, thickness=-1)

            mask, pred = self.superpixel_model.run(image, result_img)
            # use the pick mask to get x, y, theta
            pose = [center_x, center_y]
            angle = 0
            if pick_mask is not None:
                pose, angle, overlay_mask = self.find_best_placement((pred / np.max(pred)).astype(np.float32), pick_mask, angles=list(range(-30, 30, 5))) # set rotation candidates
            vis_image = np.array(image).copy()
            contours, _ = cv2.findContours(overlay_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_image, contours, -1, (255, 255, 0), thickness=3)
            cv2.imwrite("result.png", vis_image[:, :, ::-1])
            return pose, angle
        else:
            return None
    

class Tester:
    def __init__(self, model, task_info, args):
        self.model = model
        self.args = args
        self.task_info = task_info
        self.env = _set_environment(args)

    def run(self):
        for i, task_ in enumerate(self.task_info):
            task_name = self.task_info[task_]['task']
            self.evaluate_task(mode='test', eval_task=task_name)

    def evaluate_task(self, mode='test', eval_task=None):
        eval_list = range(self.task_info[eval_task]['n_demos'])
        pbar = tqdm(eval_list)

        for i in pbar:
            task = tasks.names[eval_task]()
            seed = i
            task.mode = mode
            task.name = eval_task
            np.random.seed(seed)
            random.seed(seed)


            self.env.set_task(task)
            obs, _ = self.env.reset() # get rgb image and depth in 3 views
            info = self.env.info
            if self.args.record:
                self.env.start_rec(f'{i+1:06d}')
                self.env.place_pix = None
            for j in range(1):
                
                image_size = (320, 640)
                config = self.env.agent_cams[0]
                color_, depth_, segm_ = self.env.render_camera(
                    config,
                    image_size,
                    shadow=0
                )
                
                processed_img = Image.fromarray(color_).convert('RGB')
                # generate scene graph
                scene_graph = self.env.scene_graph_generator.run(processed_img)

                obj_id = max(self.env.obj_ids['rigid'])
                pick_pose = self.env.task.get_pick_pose(self.env, obj_id)

                pick_mask = np.where(segm_ == obj_id, 1, 0).astype(np.uint8)
                bbox = np.min(pick_mask.nonzero()[1]), np.min(pick_mask.nonzero()[0]), np.max(pick_mask.nonzero()[1]), np.max(pick_mask.nonzero()[0])
                cropped_pick_mask = pick_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                result = self.model.run(processed_img, info['lang_goal'], scene_graph=scene_graph, pick_mask=cropped_pick_mask)
                
                if self.args.record:
                    color = utils.get_image(obs)[..., :3]
                    color = color.transpose(1, 0, 2)
                    caption = info['lang_goal']
                    self.env.add_video_frame_text(caption=caption)

                place_res, place_angle = result
                place_pix = [int(place_res[0]), int(place_res[1])]
                
                place_pix = [min(place_pix[0], 639), min(place_pix[1], 319)]
                p1_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, np.deg2rad(place_angle)))
                self.env.place_pix = place_pix
                place_pos = utils.raw_pix_to_xyz(
                    place_pix, depth_[place_pix[1], place_pix[0]], self.env.agent_cams[0], image_size
                )

                place_pose = (np.asarray(place_pos), np.asarray(p1_xyzw))
                act =  {'pose0': pick_pose, 'pose1': place_pose}
                
                obs, _, _, done, _, _ = self.env.step(act)     
                _, _, obj_mask = task.get_true_image(self.env)
                info = self.env.info
                done = self.env.task.done()   
                if done:
                    break
            
            if self.args.record:
                self.env.end_rec()


def _set_environment(args):
    record_cfg = {
        'save_video': args.record,
        'save_video_path': "./videos",
        'add_text': False,
        'fps': 25,
        'video_height': 320,
        'video_width': 640,
        }
    env = Environment(
        'scene_generation/environments/assets/',
        disp=False,
        shared_memory=False,
        hz=480,
        record_cfg=record_cfg
    )
    return env


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--task", default='demo-task', type=str)
    argparser.add_argument("--ndemos_test", default=1, type=int)
    argparser.add_argument("--record", default=False, action='store_true')
    argparser.add_argument("--device", default='cuda', type=str)
    argparser.add_argument("--ckpt_folder", default="coarse_vlm/checkpoints", type=str)
    
    args = argparser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    task_name = args.task
    task_info = {
        task_name: {
            'task': task_name,
            'n_demos': args.ndemos_test,
        }
    }
    
    model = LINGOSpaceInference(ckpt_folder=args.ckpt_folder)
    tester = Tester(model, task_info, args)
    tester.run()


if __name__ == "__main__":
    main()
