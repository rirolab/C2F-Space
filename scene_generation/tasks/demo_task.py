"""Packing Google Objects tasks."""

import os

import numpy as np
from scene_generation.tasks.task import Task
import scene_generation.utils.general_utils as utils
import pybullet as p


class DemoTask(Task):

    def __init__(self):
        super().__init__()
        self.save = True
        self.max_steps = 1
        self.overlap = False
        self.cnt = 0
        self.tilt_object = ['banana']
        self.other_tilt = ['dish', 'butterfinger', 'white_green_cloth', 'scissors'] 
        self.third_tilt = ['basket']        

    def reset(self, env):
        super().reset(env)

        object_template = 'object-template.urdf'
        chosen_objs = ['Tray', 'banana', 'strawberry', 'dish', 'white_green_cloth', 'scissors', 'screwdriver', 'butterfinger']
        labels = {}
        object_ids = []
        object_points = {}
        # Dummy info for cliport env
        zone_size = (1, 0.6, 0.02) # total zone size
        pos = (0.5, 0.5, 0.05)
        rot = utils.eulerXYZ_to_quatXYZW((0, 0, 0))
        zone_pose = (pos, rot)
        position_lists = [
                        ((0.5, -0.2, 0.005), (0.0, 0.0, 0.0, 1.0)),
                          ((0.42, -0.17, 0.01), (0.0, 0.0, 0.0, 1.0)),
                          ((0.53, -0.27, 0.01), (0.0, 0.0, 0.0, 1.0)),
                          ((0.4, 0.23, 0.0), (0.0, 0.0, 0.0, 1.0)),
                          ((0.6, -0.13, 0.0), (0.0, 0.0, 0.0, 1.0)),
                          ((0.61, 0.05, 0.01), (0.0, 0.0, 0.0, 1.0)),
                          ((0.59, 0.13, 0.01), (0.0, 0.0, 0.0, 1.0)),
                          ((0.39, 0.24, 0.01), (0.0, 0.0, 0.0, 1.0))]
        angle_lists = [-1.2, 
                       -0.5,
                       0.5,
                       0.0,
                       0.4,
                       1.5,
                       -0.1, 0.1]

        for i in range(len(chosen_objs)):
            object_name = chosen_objs[i]
            object_template = os.path.join('objects', object_name, 'urdf', f'{object_name}.urdf')
            if os.path.exists(os.path.join(self.assets_root, object_template)):
                if object_name == "banana":
                    scale = 70
                else:
                    scale = 1
                replace = {'package:/': os.path.join(self.assets_root, os.path.join('objects')),
                           'SCALE': [scale, scale, scale]}
            else:
                object_template = 'object-template.urdf'
                object_name_with_underscore = object_name.replace(" ", "_").replace(".obj", "")
                mesh_file = os.path.join(self.assets_root,
										 'meshes',
										 f'{object_name_with_underscore}.obj')
                
                if "butterfinger" in object_name:
                    scale = 0.75
                else:
                    scale = 0.8
                
                replace = {'FNAME': (mesh_file,),
                        'SCALE': [scale, scale, scale],
                        'COLOR': (0.2, 0.2, 0.2)}
            urdf = self.fill_template(object_template, replace)

            if object_name in self.tilt_object:
                angle = [0, 0, 0, 1]
            elif object_name in self.other_tilt:
                angle = [0, 1, 0, 0]
            elif object_name in self.third_tilt:
                angle = [1, 0, 0, 0]
            else:
                angle = [0.707, 0, 0, 0.707]

            random_angle = angle_lists[i]
            angle_tmp = utils.eulerXYZ_to_quatXYZW([0, 0, random_angle])
            angle = utils.q_mult(angle, angle_tmp)
            size, displacement = env.get_urdf_size(urdf, angle)
            
            pose = position_lists[i]
            
            if pose[0] is not None:
                # Initialize with a slightly tilted pose so that the objects aren't always erect.
                ps = ((pose[0][0] + displacement[0], pose[0][1] + displacement[1], pose[0][2]+displacement[2]), angle)
                try:
                    # print(shape_size, size)
                    box_id = env.add_object(urdf, ps)
                    labels[box_id] = object_name
                    if os.path.exists(urdf):
                        os.remove(urdf)
                    if object_template == "object-template.urdf":
                        texture_file = os.path.join(self.assets_root,
											'textures',
											f'{object_name_with_underscore}.png')
                        texture_id = p.loadTexture(texture_file)
                        p.changeVisualShape(box_id, -1, textureUniqueId=texture_id)
                        p.changeVisualShape(box_id, -1, rgbaColor=[1, 1, 1, 1])
                    object_ids.append((box_id, (0, None)))

                    object_points[box_id] = self.get_mesh_object_points(box_id)

                except Exception as e:
                    print("Failed to load Google Scanned Object in PyBullet")
                    print(object_name)
                    print(f"Exception: {e}")

        # the zone pose was the reason why it always put in the middle
        self.set_goals(object_ids, object_points, zone_pose, zone_size, env)

        self.cnt += 1

        if self.save:
            return labels
    def choose_objects(self, object_names, k):
        repeat_category = None
        return np.random.choice(object_names, k, replace=self.overlap), repeat_category

    
    def get_pick_pose(self, env, obj_id):
        # Oracle uses perfect RGB-D orthographic images and segmentation.
        _, hmap, obj_mask = self.get_true_image(env)
        pick_mask = np.uint8(obj_mask == obj_id)


        # Trigger task reset if no object is visible.
        if pick_mask is None or np.sum(pick_mask) == 0:
            self.goals = []
            self.lang_goals = []
            print('Object for pick is not visible. Skipping demo.')
            return

        # want to get height mask, sampling high points
        height_mask = hmap * pick_mask
        max_height = np.max(height_mask)
        min_height = np.min(height_mask[height_mask > 0])

        threshold = min_height
        pick_mask = np.uint8(height_mask >= threshold)    
        # Get picking pose.
        xs, ys = np.where(pick_mask)
        pick_pix = (int(np.mean(xs)), int(np.mean(ys)))
        pick_pos = utils.pix_to_xyz(pick_pix, hmap,
                                    self.bounds, self.pix_size)
        pick_pose = (np.asarray(pick_pos), np.asarray((0, 0, 0, 1)))
        return pick_pose

    def set_goals(self, object_ids, object_points, zone_pose, zone_size, env=None):
        # Random picking sequence.
        num_pack_objs = 1
        object_ids = object_ids[:num_pack_objs]
        env.obj_ids['deformable']=[c[0] for c in object_ids]
        env.obj_ids['move']=[c[0] for c in object_ids]
        true_poses = []
        for obj_idx, (object_id, _) in enumerate(object_ids):
            true_poses.append(zone_pose)

            chosen_obj_pts = dict()
            chosen_obj_pts[object_id] = object_points[object_id]

            self.goals.append(([(object_id, (0, None))], np.int32([[1]]), [zone_pose],
                               False, True, 'neglect',
                               (chosen_obj_pts, [(zone_pose, zone_size)]),
                               1 / len(object_ids))) # put dummy goal
            self.lang_goals.append("Put the Butterfinger to the left of the banana at about twice the distance between the scissors and the screwdriver.")


        # Only mistake allowed.
        self.max_steps = len(object_ids)+1
