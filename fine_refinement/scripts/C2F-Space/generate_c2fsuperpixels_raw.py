import os

import numpy as np
import torch
import torch_scatter
from PIL import Image
from torch.utils import data

import random
import scipy
import pickle
from skimage.segmentation import slic
from skimage.future import graph
from skimage import filters, color

import scipy.ndimage
from collections import defaultdict

import time
import dgl
import networkx as nx
from tqdm import tqdm
from joblib import delayed

from tqdm.auto import tqdm
from joblib import Parallel
from itertools import cycle, islice
import json, cv2
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Positional encoding propsed by the NeRF paper (https://arxiv.org/abs/2003.08934)
    Implementation based on https://github.com/jiupinjia/rocket-recycling/blob/main/policy.py#L22

    Args:
        in_dim: input dimension
        K: number of frequencies
        scale: scale of the positional encoding
        include_input: whether to include the input in the positional encoding
    
    Returns:
        h: positional encoding of the input
    """
    def __init__(self, dim_in, K=5, scale=1.0, include_input=True):
        super().__init__()
        self.K = K
        self.scale = scale
        self.include_input = include_input
        self.dim_out = dim_in * (K*2 + include_input)

    def forward(self, x):
        x = x * self.scale
        if self.K == 0:
            return x

        h = [x] if self.include_input else []
        for i in range(self.K):
            h.append(torch.sin(2**i * torch.pi * x))
            h.append(torch.cos(2**i * torch.pi * x))
        h = torch.cat(h, dim=-1) / self.scale
        return h

class ProgressParallel(Parallel):
    """A helper class for adding tqdm progressbar to the joblib library."""
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            result = Parallel.__call__(self, *args, **kwargs)
            self.print_progress()
            return result

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()



def get_sp_node_label(superpixels, gt_mask,):
    # generate labels!
    gt_mask = np.where(gt_mask !=0, 255, 0)
    tmp_torch = torch.Tensor(gt_mask)
    labels_torch = torch.Tensor(superpixels).type(torch.int64)
    out = torch_scatter.scatter(tmp_torch.flatten(), labels_torch.flatten(), reduce='mean')
    idxs = np.where(out > 255.0*0.9)[0]

    sp_node_label = np.zeros(superpixels.max()+1, dtype=np.int64)
    sp_node_label[idxs] = 1
    return sp_node_label
            


def process_single_image_slic(params):

    img, prob_img, gt_mask, args, shuffle, pos_encoding, category = params
    args_split, args_seed, args_n_sp, args_compactness = args
    img_original = np.copy(img)

    random.seed(args_seed)
    np.random.seed(args_seed)
    
    assert img.dtype == np.uint8, img.dtype
    img = (img / 255.).astype(np.float32)
    assert np.max(img) > 0.5
    
    assert prob_img.dtype == np.uint8
    prob_img = (prob_img / 255.).astype(np.float32)
    logit_img = prob_img.copy()
    logit_img = np.clip(logit_img, 1e-2, 1.0 - 1e-2)
    logit_img = np.log(logit_img) - np.log(1.0 - logit_img)

    n_sp_extracted = args_n_sp + 1  # number of actually extracted superpixels (can be different from requested in SLIC)
    
    # number of superpixels we ask to extract (larger to extract more superpixels - closer to the desired n_sp)
    n_sp_query = args_n_sp + 10
    
    while n_sp_extracted > args_n_sp:
        superpixels = slic(img, n_segments=n_sp_query, compactness=args_compactness, multichannel=len(img.shape) > 2, start_label=0)
        sp_indices = np.unique(superpixels)
        n_sp_extracted = len(sp_indices)
        n_sp_query -= 1  # reducing the number of superpixels until we get <= n superpixels

    assert n_sp_extracted <= args_n_sp and n_sp_extracted > 0, (args_split, n_sp_extracted, args_n_sp)
    
    # make sure superpixel indices are numbers from 0 to n-1
    assert n_sp_extracted == np.max(superpixels) + 1, ('superpixel indices', np.unique(superpixels))  
    # Creating region adjacency graph based on boundary
    gimg = color.rgb2gray(img_original)
    edges = filters.sobel(gimg)
    img = gimg
    
    try:
        g = graph.rag_boundary(superpixels, edges)
    except ValueError: # Error thrown when graph size is perhaps 1
        print("ignored graph")
        g = nx.complete_graph(sp_indices) # so ignoring these for now and placing dummy info
        nx.set_edge_attributes(g, 0., "weight")
        nx.set_edge_attributes(g, 0, "count")
    
    if shuffle:
        ind = np.random.permutation(n_sp_extracted)
    else:
        ind = np.arange(n_sp_extracted)

    sp_order = sp_indices[ind].astype(np.int32)
    if len(img.shape) == 2:
        img = img[:, :, None]
    h, w = img.shape[:2]
    n_ch = 1 if img.shape[2] == 1 else 3

    sp_intensity, sp_coord, logits = [], [], []
    for seg in sp_order:
        mask = (superpixels == seg).squeeze()
        avg_value = np.zeros(n_ch)
        std_value = np.zeros(n_ch)
        max_value = np.zeros(n_ch)
        min_value = np.zeros(n_ch)
        
        prob_avg_value = np.zeros(1)
        prob_std_value = np.zeros(1)
        prob_max_value = np.zeros(1)
        prob_min_value = np.zeros(1)
        
        for c in range(n_ch):
            avg_value[c] = np.mean(img[:, :, c][mask])
            std_value[c] = np.std(img[:, :, c][mask])
            max_value[c] = np.max(img[:, :, c][mask])
            min_value[c] = np.min(img[:, :, c][mask])
        # add calculation for mean, std, max, min for prob, too!
        prob_avg_value[0] = np.mean(prob_img[mask])
        prob_std_value[0] = np.std(prob_img[mask])
        prob_max_value[0] = np.max(prob_img[mask])
        prob_min_value[0] = np.min(prob_img[mask])
        
        logit_mean = np.mean(logit_img[mask])
        logits.append(logit_mean)
        
        cntr = np.array(scipy.ndimage.center_of_mass(mask))  # row, col
        
        cntr = cntr[0] / h, cntr[1] / w
        assert cntr[0] < 1.0 and cntr[1] < 1.0
        one_hot_cat = np.zeros(3)
        one_hot_cat[category] = 1
        sp_intensity.append(np.concatenate((avg_value,
                                    max_value,
                                    min_value,
                                    prob_avg_value,
                                    prob_max_value,
                                    prob_min_value, one_hot_cat), -1))
        sp_coord.append(cntr)
    sp_intensity = np.array(sp_intensity, np.float32)
    sp_coord = np.array(sp_coord, np.float32)
    logits = np.array(logits, np.float32)

    rag_boundary_graphs = dgl.from_networkx(g.to_directed(),edge_attrs=['weight', 'count'])
    sp_data = sp_intensity, sp_coord, sp_order, logits
        
    sp_node_labels = get_sp_node_label(superpixels, gt_mask)
    if sp_node_labels is None:
        return
    return superpixels, rag_boundary_graphs, sp_data, sp_node_labels



class C2F_Images_Masks(data.Dataset):
    def __init__(self, mode, compactness, root='../../datasets'):
        self.root = root
        self.mode = mode
        self.all_superpixels = []
        self.all_rag_boundary_graphs = []
        self.all_sp_data = []
        self.all_sp_node_labels = []
        self.all_meta_info = []
        
        self.n_sp = 500
        self.compactness = compactness
        self.seed = 41
        self.dataset = 'composite'
        self.out_dir = f"../../datasets/C2FSuperpixels" # '.'
        
        self.args = self.mode, self.seed, self.n_sp, self.compactness
        self.pos_encoding = PositionalEncoding(dim_in=2, K=4)

        self.num_images = self._pack_images_masks(mode)

        
    def _pack_images_masks(self, mode):
        assert mode in ['train', 'val', 'test']
        
        dataType = mode

        base_imgs = [
            os.path.join(l, "0_0.png")
            for l in sorted(os.listdir(os.path.join(self.root, f"{self.dataset}-{dataType}", 'images')))
            ] # TODO: need to change this
        print("original length: ", len(base_imgs))
        max_length = 4 if mode == "train" else 1
        all_imgs = [img for img in base_imgs for _ in range(max_length)]
        print("total length: ", len(all_imgs))
        augments = list(islice(cycle(list(range(max_length))), len(all_imgs)))
        each_count = defaultdict(int)
        if mode == 'train' and len(all_imgs) > 10:
            filtered_imgs = []
            filtered_augments = []
            for img_fname, augment in zip(all_imgs, augments):
                json_path = os.path.join(self.root, f"{self.dataset}-{dataType}", 'priors', img_fname)
                json_path = json_path.replace("0_0.png", "augment_info.json")
                info_path = os.path.join(self.root, f"{self.dataset}-{dataType}", 'info', img_fname).replace("/0_0.png", ".pkl")
                if os.path.exists(info_path):
                    with open(info_path, "rb") as f:
                        info = pickle.load(f)[0]
                    category = info['category']
                else:
                    category = 2
                with open(json_path, "r") as f:
                    aug_info = json.load(f)
                    res = aug_info[f"0_{augment}"]["result"]
                    if res:
                        filtered_imgs.append(img_fname)
                        filtered_augments.append(augment)
                        each_count[category] += 1
            all_imgs = filtered_imgs
            augments = filtered_augments

        parallel = ProgressParallel(n_jobs=4, batch_size=32, prefer='threads', use_tqdm=True, total=len(all_imgs))
        all_imgs_outs = parallel(delayed(self.get_superpixel_graph)(img_fname, dataType=dataType, augment=augment) for img_fname, augment in zip(all_imgs, augments))
        
        os.makedirs(self.out_dir, exist_ok=True)

        count = 0
        for img_fname, prob_fname, out in all_imgs_outs:
            if out is None: continue
            self.all_superpixels.append(out[0])
            self.all_rag_boundary_graphs.append(out[1])
            self.all_sp_data.append(out[2])
            self.all_sp_node_labels.append(out[3])

            self.all_meta_info.append({
                'img_fname': img_fname,
                'prob_fname': prob_fname
                }
                )
            count += 1

        with open('%s/%dsp_%dcmpt_%s_superpixels.pkl' % (self.out_dir, self.n_sp, self.compactness, self.mode), 'wb') as f:
            pickle.dump(self.all_superpixels, f)
        with open('%s/%dsp_%dcmpt_%s.pkl' % (self.out_dir, self.n_sp, self.compactness, self.mode), 'wb') as f:
            pickle.dump((self.all_sp_node_labels, self.all_sp_data), f)
        with open('%s/%dsp_%dcmpt_%s_rag_boundary_graphs.pkl' % (self.out_dir, self.n_sp, self.compactness, self.mode), 'wb') as f:
            pickle.dump(self.all_rag_boundary_graphs, f)
        with open('%s/%dsp_%dcmpt_%s_meta_info.pkl' % (self.out_dir, self.n_sp, self.compactness, self.mode), 'wb') as f:
            pickle.dump(self.all_meta_info, f)

        print(f"{count} / {len(all_imgs)} images processed")
        return count
    
    def _img_fname_to_info_fname(self, img_fname):
        return img_fname.replace('/0_0.png', '.pkl')
    
    def _img_fname_to_seg_fname(self, img_fname):
        return img_fname

    def get_superpixel_graph(self, img_fname, dataType, augment):
        image_path = os.path.join(self.root, f"{self.dataset}-{dataType}", 'images', img_fname)
        gt_path = image_path.replace("images", "results")
        info_path = image_path.replace("images", "info").replace("/0_0.png", ".pkl")
        if os.path.exists(info_path):
            with open(info_path, "rb") as f:
                info = pickle.load(f)[0]
            category = info['category']
        else:
            category = 2

        prob_path = image_path.replace("images", "priors")
        prob_fname = "0_0.png"
        if augment < 2:
            horizontal_flip = False
        else:
            horizontal_flip = True
        if augment % 2 == 0:
            vertical_flip = False
        else:
            vertical_flip = True
        
        img = Image.open(image_path).convert('RGB')
        if horizontal_flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if vertical_flip:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

        # prob image
        prob_img = Image.open(prob_path).convert('L')
        if horizontal_flip:
            prob_img = prob_img.transpose(Image.FLIP_LEFT_RIGHT)
        if vertical_flip:
            prob_img = prob_img.transpose(Image.FLIP_TOP_BOTTOM)

        prob_img = np.array(prob_img, np.uint8)
        prob_img = np.where(prob_img != 0, 1, 0).astype(np.uint8)
        if np.sum(prob_img) != 0:
            dist_inside = cv2.distanceTransform(prob_img, cv2.DIST_L2, 5).astype(np.float32)
            prob_img = dist_inside / np.max(dist_inside) * 255
            prob_img = prob_img / np.max(prob_img) * 255
            prob_img = prob_img.astype(np.uint8)
        
        # gt image
        gt_mask = Image.open(gt_path).convert('L')
        if horizontal_flip:
            gt_mask = gt_mask.transpose(Image.FLIP_LEFT_RIGHT)
        if vertical_flip:
            gt_mask = gt_mask.transpose(Image.FLIP_TOP_BOTTOM)


        slic_result = process_single_image_slic(
            (np.array(img, np.uint8),
             np.array(prob_img, np.uint8),
             np.array(gt_mask, np.uint8),
             self.args,
             False,
             self.pos_encoding, category)
             )

        return img_fname, prob_fname, slic_result
    
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return self.num_images
    

root = "../../data/"

t0 = time.time()
print("[I] Reading and loading Images and Masks for TRAIN set for sp=100, cmpt=30..")
C2F_Images_Masks('train', compactness=10, root=root)
print("[I] Time taken: {:.4f}s".format(time.time()-t0))

t0 = time.time()
print("[I] Reading and loading Images and Masks for VAL set for sp=100, cmpt=10..")
C2F_Images_Masks('val', compactness=10, root=root)
print("[I] Time taken: {:.4f}s".format(time.time()-t0))


t0 = time.time()
print("[I] Reading and loading Images and Masks for TEST set for sp=100, cmpt=10..")
C2F_Images_Masks('test', compactness=10, root=root)
print("[I] Time taken: {:.4f}s".format(time.time()-t0))
