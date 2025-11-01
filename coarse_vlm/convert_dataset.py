import os, cv2, shutil, pickle, json
import numpy as np
import argparse
import random
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import pandas as pd
from openai import OpenAI
from pydantic import BaseModel
from typing import List

def llm_parse(text, idxs):
    class MathReasoning(BaseModel):
        reference_objects: List[int]
        relations: List[str]
    prompt = """Please identify 1) reference objects and 2) relations in the given instruction.
Instruction: {text}
object idxs: {idxs}
Please return the idx of the reference objects as a list, and the relations as a list of strings. Make sure the length of the two lists are the same.
""".format(text=text, idxs=idxs)
    api_key = os.environ.get("OPENAI_API_KEY")  
    client = OpenAI(api_key = api_key)
    kwargs = {
        "model": "o4-mini-2025-04-16",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            },
        ],
        "response_format": MathReasoning,
        "max_completion_tokens": 10000,
    }
    completion = client.beta.chat.completions.parse(**kwargs)
    reference_objects = completion.choices[0].message.parsed.reference_objects
    relations = completion.choices[0].message.parsed.relations
    reference_objects = [idx for idx in reference_objects]
    relations = [rel.lower() for rel in relations]
    return reference_objects, relations
                  
def make_sg(boxes=None):
        """
        return scene graph based on self.boxes
        :: scene_graph = list( (id_1, 'in', id_2), ...)
        """
        def is_on(box1, box2):
            return abs(box1[4] - box2[4] - 1) < 0.000001 or (box1[4] == 0 and box2[4] == 3)

        def is_in(box1, box2):
            # box1 is in box2 >> return True
            return box1[0] > box2[0] and box1[1] > box2[1] and box1[2] < box2[2] and box1[3] < box2[3] and box1[4] == box2[4]
        
        def is_near(box1, box2):
            cx1,cy1 = (box1[0]+box1[2])/2, (box1[1]+box1[3])/2
            cx2,cy2 = (box2[0]+box2[2])/2, (box2[1]+box2[3])/2       
            return ((cx1-cx2)**2+(cy1-cy2)**2) < (150**2) and box1[4] == box2[4]
        
        sg = []
        for i1, id1 in enumerate(boxes.keys()):
            for i2, id2 in enumerate(boxes.keys()):
                if i1 >= i2:
                    continue
                box1, box2 = boxes[id1], boxes[id2]
                if is_on(box1, box2):
                    sg.append((id1, 'on', id2))
                elif is_on(box2, box1):
                    sg.append((id2, 'on', id1))
                elif is_in(box1, box2):
                    sg.append((id1, 'in', id2))
                elif is_in(box2, box1):
                    sg.append((id2, 'in', id1))
                elif is_near(box1,box2):
                    sg.append((id1, 'near', id2))
                    sg.append((id2, 'near', id1))
        return sg
def get_iou(pred_mask, gt_mask):
    pred_mask = np.where(pred_mask != 0, 1, 0)
    gt_mask = np.where(gt_mask != 0, 1, 0)
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return intersection / union if union > 0 else 0

# python convert_dataset.py --iit_path data/dataset_jan --iit_output_path output/dataset_jan_fin --kaist_path /harddisk/nlu/iit-data/jan-dataset-test/composite
# python convert_dataset.py --iit_path data/dataset_jan --iit_output_path output/dataset_jan_fin --kaist_path /harddisk/nlu/iit-data/jan-dataset-fin/composite
# python convert_dataset.py --iit_path data/dataset_jan --iit_output_path output/dataset_jan_el --kaist_path /harddisk/nlu/iit-data/jan-dataset-el/composite
# python convert_dataset.py --iit_path data/dataset_jan --iit_output_path output/test_all_o4_new_2 --kaist_path /harddisk/nlu/iit-data/real-final-el2/composite
# python convert_dataset.py --iit_path data/dataset_jan --iit_output_path output/test_all_o4_nn_1 --kaist_path /harddisk/nlu/iit-data/real-final-el/composite
# python convert_dataset.py --iit_path data/dataset_jan --iit_output_path output/test_all_o4_nn_2 --kaist_path /harddisk/nlu/iit-data/real-final-el2/composite

# python convert_dataset.py --iit_path data/dataset_jan --iit_output_path output/dataset_jan_3 --kaist_path /harddisk/nlu/iit-data/isr-final-el2/composite

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--iit_path', required=True)
    parser.add_argument('--iit_output_path')
    parser.add_argument('--kaist_path', required=True)

    args = parser.parse_args()

    base_path = args.iit_path
    base_output_path = args.iit_output_path
    new_base_path = args.kaist_path 
    temp_path = "/harddisk/nlu/iit-data/dataset-updated-LINGO/composite"
    need_to_check_idx = []
    # need_to_check_idx = [11, 37, 53, 91, 92, 94, 106, 131, 140, 158, 200, 213, 233, 234, 240, 241, 242, 243, 244, 245,
    #                      286, 287, 288, 289, 290, 300, 316, 339, 341, 352, 355, 357, 363, 375, 377, 390, 393, 396]
    # need_to_check_idx = [str(id) for id in need_to_check_idx]
    random.seed(0)
    with open(os.path.join(base_path, "info.json"), "r") as f:
        base_info = json.load(f)
        
    whole_idxs = list(base_info.keys())
    with open(os.path.join(base_path, "split_info.json"), "r") as f:
        idxs_info = json.load(f)
        train_idxs = idxs_info["train_idxs"]
        val_idxs = idxs_info["val_idxs"]
        test_idxs = idxs_info["test_idxs"]
        category_info = idxs_info["category_info"]    
            
    if os.path.exists(os.path.join(base_output_path, "cut_2.json")):
        with open(os.path.join(base_output_path, "cut_2.json"), "r") as f:
            prior_info = json.load(f)
    
    for mode, idxs in zip(['train', 'val', 'test'], [train_idxs, val_idxs, test_idxs]):
        # if mode == 'train':
        #     continue
        ious = []
        for inner_i, idx in enumerate(tqdm(sorted(idxs))):
            # if idx not in prior_info:
            #     continue
            name = str(idx).zfill(6) + '-' + str(idx)
            os.makedirs(os.path.join(new_base_path+f'-{mode}', "images", name), exist_ok=True)
            os.makedirs(os.path.join(new_base_path+f'-{mode}', "priors", name), exist_ok=True)
            os.makedirs(os.path.join(new_base_path+f'-{mode}', "results", name), exist_ok=True)
            if not os.path.exists(os.path.join(new_base_path+f'-{mode}', "info")):
                os.makedirs(os.path.join(new_base_path+f'-{mode}', "info"))
            img_path = os.path.join(base_path, "images", base_info[idx]['image_path'])
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            cv2.imwrite(os.path.join(new_base_path+f'-{mode}', "images", name, '0_0.png'), img)
            
            gt_path = os.path.join(base_path, "ground_truth", str(idx)+".png")
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            new_gt = (gt == 255)*255
            new_gt = new_gt.astype(np.uint8)
            cv2.imwrite(os.path.join(new_base_path+f'-{mode}', "results", name, '0_0.png'), new_gt)
            # plot the original image and gt together
            gt_vis = np.stack([new_gt, np.zeros_like(new_gt), np.zeros_like(new_gt)], axis=2)
            vis_img = cv2.addWeighted(img, 0.5, gt_vis, 0.5, 0, dtype=cv2.CV_8U)
                        
            scene_graph_path = os.path.join(base_path, "scene_graph", base_info[idx]['image_path'].replace(".jpg", ".json").replace(".png", ".json"))
            # the scene graph info contains bounding box informations
            with open(scene_graph_path, "r") as f:
                scene_graph = json.load(f)
            pass_always = False
            if os.path.exists(os.path.join(new_base_path+f'-{mode}', "info", f"{name}.pkl")):
                pass_always = True
                with open(os.path.join(new_base_path+f'-{mode}', "info", f"{name}.pkl"), "rb") as f:
                    info = pickle.load(f)
                while type(info) is not dict:
                    info = info[0]
            else:
                info = {}
            if not pass_always:
                names = {}
                boxes = {}
                info['deformable'] = {}
                info['move'] = {}
                info['move_goal'] = {}
                rigid = {}
                fixed = {}
                for obj_id, obj_name in enumerate(scene_graph['objects_info'].keys()):
                    # it will return the object names
                    if 'projected_box' in scene_graph['objects_info'][obj_name]:
                        x0, y0, x1, y1 = scene_graph['objects_info'][obj_name]['projected_box']
                    else:
                        x0, y0, x1, y1 = scene_graph['objects_info'][obj_name]['bbox']
                    # bbox format is x0, w, y0, h
                    # let's put all object as rigid
                    level = scene_graph['objects_info'][obj_name].get('level', 1)
                    names[obj_id+2] = obj_name
                    if level == 0 or level == 3: # fixed one, such as shelf
                        fixed[obj_id+2] = (((round(x0), round(y0)), (round((x1)), round((y1)))),
                                    (0.0, 0.0, 0.0, 1.0),
                                        obj_name, level)
                    else:
                        rigid[obj_id+2] = (((round(x0), round(y0)), (round((x1)), round((y1)))),
                                        (0.0, 0.0, 0.0, 1.0),
                                            obj_name, level)
                    boxes[obj_id+2] = (round(x0), round(y0), round((x1)), round((y1)), level)
                # latter add scene graph, same as previous rule
                # add shelf latter..?
                if "shelf" not in names.values() and "shelf_1" not in names.values():
                    fixed = {0: (((0, 0), (w-1, h-1)), (0.0, 0.0, 0.0, 1.0), 'plane', 0)}
                    names[0] = 'plane'
                    boxes[0] = (0, 0, w-1, h-1, 0)
                info['fixed'] = fixed
                info['rigid'] = rigid
                info['names'] = names
                info['graph'] = make_sg(boxes)
                # import pdb;pdb.set_trace()
                print("draw")
                for obj_idx, obj_bbox in boxes.items():
                    cv2.rectangle(vis_img, obj_bbox[:2], obj_bbox[2:4], color=(200, 200, 200), thickness=2)
                    cv2.putText(vis_img, str(obj_idx), obj_bbox[:2], cv2.FONT_HERSHEY_SIMPLEX, 2, (200, 200, 200), 2)
                cv2.imwrite("tmp.png", vis_img)
                print(base_info[idx]["instruction"])
                print(names)
                print(boxes)
                check = False
                if mode == 'train' or mode == 'val':
                    # if 'rel_ids' in info:
                    #     print(idx)
                    #     print(info["ref_ids"])
                    #     print(info['parsing'])
                    #     check = str(input("Ok? T/F")) == "T"
                    if (not check):
                        rel_id = 2 # int(input("rel id: "))
                        # ref_ids, relations = llm_parse(base_info[idx]["instruction"], names)
                        # print(ref_ids, relations)
                        # check = str(input("Ok? T/F")) == "T"
                        if not check:
                            count = int(input("count: "))
                            info['rel_ids'] = [rel_id for c in range(count)] # should be generated latter with gpt
                            info['ref_ids'] = [] # should be generated latter with gpt
                            info['relations'] = [] # should be generated latter with gpt
                            parsing = {'action': 'move', 'source': 'object'}
                            parsing_target = []
                            for i in range(count):
                                # ref_name = input("ref name: ")
                                ref_ids = input("ref ids: ").split(", ")
                                relation = input("relations: ")
                                info['ref_ids'].append(ref_ids)
                                info['relations'].append(relation)
                                # parsing_target.append((ref_name, relation))
                            # parsing["target"] = parsing_target
                            # info["parsing"] = parsing
                        else:
                            count = len(ref_ids)
                            info['rel_ids'] = [rel_id for c in range(count)] # should be generated latter with gpt
                            info['ref_ids'] = ref_ids
                            info['relations'] = relations
            if 'relations' in info:
                info['prior_pred'] = info['relations']
                info['prior_ids'] = info['ref_ids']
            else:
                info['prior_pred'] = []
                info['prior_ids'] = []
            info['prior_instruction'] = base_info[idx]["instruction"]
            info['lang_goal'] = base_info[idx]["instruction"]

            if os.path.exists(os.path.join(base_output_path, "cut_2.json")):
                # make prior image
                prior = np.zeros((h, w), dtype=np.uint8)
                # polygon_cordinates = np.array(prior_info[idx]["polygon_coordinates"], dtype=np.int32)
                # # print(polygon_cordinates)
                # cv2.fillPoly(prior, [polygon_cordinates], color=255)
                if idx in prior_info:
                    if "Axes Length" in prior_info[idx]:
                        for i in range(len(prior_info[idx]["Center Coordinates"])):
                            x, y = prior_info[idx]["Center Coordinates"][i]
                            r0, r1 = prior_info[idx]["Axes Length"][i]
                            angle = prior_info[idx]["Angle"][i]
                            cv2.ellipse(prior, (round(x), round(y)), (round(r0), round(r1)), angle, 0, 360, color=255, thickness=-1)
                        # x, y = prior_info[idx]["Center Coordinates"][0]
                        # r0, r1 = prior_info[idx]["Axes Length"][0]
                        # # r0 /= 2
                        # # r1 /= 2
                        # angle = prior_info[idx]["Angle"]
                        # cv2.ellipse(prior, (round(x), round(y)), (round(r0), round(r1)), angle, 0, 360, color=255, thickness=-1)
                    else:
                        x, y = prior_info[idx]["Center Coordinates"][0]
                        cv2.circle(prior, [round(x), round(y)], prior_info[idx]["Radius"], color=255, thickness=-1)
                
                cv2.imwrite(os.path.join(new_base_path+f'-{mode}', "priors", name, '0_0.png'), prior)
                tmp_vis = np.stack([np.zeros_like(prior), prior, np.zeros_like(prior)], axis=2)
                vis_img = cv2.addWeighted(vis_img, 0.7, tmp_vis, 0.3, 0, dtype=cv2.CV_8U)
                
                # cv2.imwrite("tmp.png", vis_img)
                # then update prior_ids, too
                if idx in prior_info:
                    if 'Relevant Objects' in prior_info[idx]:
                        prior_ids = []
                        for id in prior_info[idx]["Relevant Objects"]:
                            if id+2 in info["names"]:
                                prior_ids.append(id+2)
                        # prior_ids = [id+2 for id in prior_info[idx]["Relevant Objects"]]
                        info['prior_ids'] = prior_ids
                with open(os.path.join(new_base_path+f'-{mode}', "priors", name, 'augment_info.json'), "w") as f:
                    aug_info = {}
                    prior = cv2.imread(os.path.join(new_base_path+f'-{mode}', "priors", name, '0_0.png'), cv2.IMREAD_GRAYSCALE)
                    prior_dst = np.where(prior != 0, 1, 0).astype(np.uint8)
                    # dst = cv2.distanceTransform(prior_dst, cv2.DIST_L2, 3)
                    # max_idx = np.argmax(dst)
                    # y, x = np.unravel_index(max_idx, prior.shape)
                    M = cv2.moments(prior_dst)
                    if M["m00"] != 0:
                        x = int(M["m10"] / M["m00"])
                        y = int(M["m01"] / M["m00"])
                    else:
                        x, y = 0, 0  # for empty mask
                    # visualize the dot in the center of the prior
                    gt_path = os.path.join(base_path, "ground_truth", str(idx)+".png")
                    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                    iou = get_iou(prior_dst, gt)
                    ious.append(iou)
                    # print(f"IOU: {iou}")
                    vis_img = np.copy(gt)
                    vis_img = np.stack([vis_img, vis_img, vis_img], axis=2)
                    # cv2.circle(vis_img, (x, y), 5, (0, 0, 255), -1)
                    # cv2.imwrite("tmp.png", vis_img)
                    # _ = input("temp")
                    # import pdb;pdb.set_trace()
                    res = gt[min(y, h-1), min(x, w-1)] == 255
                    for augment in range(4):
                        if augment < 2:
                            horizontal_flip = False
                        else:
                            horizontal_flip = True
                        if augment % 2 == 0:
                            vertical_flip = False
                        else:
                            vertical_flip = True
                        aug_info[f"0_{augment}"] = {
                            'horizontal_flip': horizontal_flip,
                            'vertical_flip': vertical_flip,
                            'result': 1 if res else 0,
                            # 'result': 1 if iou > 0.1 else 0,
                        }
                    json.dump(aug_info, f)
            # elif os.path.exists(os.path.join(base_output_path, f"{idx}.json")):
            #     # then generate prior image based on the polygon
            #     with open(os.path.join(base_output_path, f"{idx}.json"), "r") as f:
            #         prior_info = json.load(f)
            #     x0, y0 = prior_info['polygon'][0]
            #     x1, y1 = prior_info['polygon'][2]
            #     prior = np.zeros((h, w))
            #     prior[round(y0):round(y1), round(x0):round(x1)] = 255
            #     cv2.imwrite(os.path.join(new_base_path+f'-{mode}', "priors", name, '0_0.png'), prior)
            category = category_info[idx] if mode != "test" else prior_info[idx]["Instruction Type"] if idx in prior_info else "unknown"
            category = category.lower()
            if category == "simple":
                category = 0
            elif category == "multiple instances" or category == "predicate":
                category = 1
            else:
                category = 2    
            info['category'] = category
            # has two ways, output.json and my custom function
            
            assert 'prior_ids' in info
            with open(os.path.join(new_base_path+f'-{mode}', "info", f"{name}.pkl"), "wb") as f:
                pickle.dump([info], f)
        print(f"Average IOU for {mode} set: {np.mean(ious)}")

