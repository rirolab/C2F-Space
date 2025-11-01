import graphgps  # noqa
from graphgps.finetuning import load_pretrained_model_cfg, init_model_from_pretrained

from torch_geometric import seed_everything
from torch_geometric.loader import DataLoader

from torch_geometric.graphgym.config import set_cfg, load_cfg
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.model_builder import create_model

import argparse
import torch
import os
import numpy as np
import pickle
import shutil

import numpy as np
import cv2
import subprocess

class DemoGPS:
    def __init__(self, cfg, pretrained_dir, ckpt_idx):
        args = argparse.Namespace(
            cfg_file=None,
            repeat=1,
            mark_done=False,
            opts=[]
        )
        args.cfg_file = f'{pretrained_dir}/config.yaml'
        set_cfg(cfg)
        cfg.set_new_allowed(True)
        cfg.run_dir = ''
        load_cfg(cfg, args)

        seed_everything(cfg.seed)

        cfg.train.finetune = pretrained_dir
        cfg = load_pretrained_model_cfg(cfg)
        cfg.accelerator = 'cuda:0'
        self.cfg = cfg

        self.model = create_model()
        self.model = init_model_from_pretrained(self.model, cfg.train.finetune,
                                                    cfg.train.freeze_pretrained, ckpt_idx=ckpt_idx)
        self.model.eval()

    def set_datapoint(self, image, pred):
        # remote directory, save the image and pred at the destination temp directory
        os.chdir("fine_refinement")
        for mode in ["train", "val", "test"]:
            if os.path.exists(f"data/composite-{mode}"):
                shutil.rmtree(f"data/composite-{mode}")
        if os.path.exists("datasets/C2FSuperpixels"):
            shutil.rmtree("datasets/C2FSuperpixels")
        os.makedirs("data/composite-test/images/0")
        os.makedirs("data/composite-test/results/0")
        os.makedirs("data/composite-test/priors/0")
        cv2.imwrite("data/composite-test/images/0/0_0.png", cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
        pred = np.where(pred == 0, 0, 255).astype(np.uint8)
        cv2.imwrite("data/composite-test/results/0/0_0.png", pred)
        cv2.imwrite("data/composite-test/priors/0/0_0.png", pred)
        shutil.copytree("data/composite-test", "data/composite-val")
        shutil.copytree("data/composite-test", "data/composite-train")
        # chdir
        os.chdir("scripts/C2F-Space")
        subprocess.run(["python", "generate_c2fsuperpixels_raw.py"])
        subprocess.run(["python", "prepare_c2f_pygsource.py"])
        os.chdir("../..")
    
    def generate_mask(self, prediction):
        prediction = (prediction / np.max(prediction) * 255).astype(np.uint8)
        t, t_otsu = cv2.threshold(prediction, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return t_otsu

    def run(self, image, base_pred):
        w, h = image.size
        self.set_datapoint(image, base_pred)
        # import pdb; pdb.set_trace()
        with open(f"datasets/C2FSuperpixels/500sp_10cmpt_test_superpixels.pkl", "rb") as f:
            data = pickle.load(f)    
        loaders = create_loader()
        dataset = loaders[2].dataset
        dataloader = DataLoader(dataset, batch_size=1)
        for idx, batch in enumerate(dataloader):
            batch = batch.to(self.cfg.accelerator)
            with torch.no_grad():
                batch_clone = batch.clone()
                pred, true = self.model(batch_clone)
                if self.cfg.model.residual:
                    pred = 0.9 * pred + 0.1 * batch.logits.unsqueeze(1)
            pred_img = np.zeros((h, w))
            sp = data[idx]
            # import pdb; pdb.set_trace()
            for i, val in enumerate(torch.sigmoid(pred[:, 0]).cpu().numpy()):
                mask = sp == i
                pred_img[mask] = val
            pred_img = pred_img / np.max(pred_img) * 255
            result_mask = self.generate_mask(prediction=pred_img)
            break
        os.chdir("../")
        prob_img = np.stack([pred_img, base_pred, np.zeros_like(result_mask)], axis=-1).astype(np.uint8)
        prob_img = cv2.addWeighted(prob_img, 0.5, np.array(image), 0.5, 0)
        # prob_img = mark_boundaries(prob_img, sp, color=(0, 1, 0))
        cv2.imwrite("result_img.png", cv2.cvtColor(np.array(prob_img), cv2.COLOR_RGB2BGR))
        # cv2.imwrite("result_prob.png", cv2.cvtColor((prob_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        # import pdb; pdb.set_trace()
        return result_mask, pred_img
        
    