"""
MIT License

Copyright (c) 2024 Mohamed El Banani

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from datetime import datetime
import argparse
import sys
import os
from PIL import Image, ImageOps
import hydra
import numpy as np
import torch
import torch.nn.functional as nn_F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from evals.datasets.scannet_pairs import ScanNetPairsDataset
from evals.utils.correspondence import (
    compute_binned_performance,
    estimate_correspondence_depth,
    project_3dto2d,
)
from evals.utils.transformations import so3_rotation_angle, transform_points_Rt
from torchvision import transforms as transforms
from metric_learning import viz_feat
from sklearn.decomposition import PCA


def read_image(image_path: str, exif_transpose: bool = True) -> Image.Image:
    """Reads a NAVI image (and rotates it according to the metadata)."""
    with open(image_path, "rb") as f:
        with Image.open(f) as image:
            if exif_transpose:
                image = ImageOps.exif_transpose(image)
            image.convert("RGB")
            return image


# @hydra.main("./configs", "scannet_correspondence", None)
@hydra.main("./configs", "scannet_correspondence", None)
def main(cfg: DictConfig):
    print(f"Config: \n {OmegaConf.to_yaml(cfg)}")
    features = []

    # ===== Get model and dataset ====
    model = instantiate(cfg.backbone, output="dense", return_multilayer=cfg.multilayer)
    model = model.to("cuda")
    dataset = ScanNetPairsDataset()

    rgb_transform = transforms.Compose(
        [
            transforms.Resize((360, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    dataset_root = "/data/fanry/Desktop/fmft/realworld/selfData"
    seqs = ["hand_out"] #, "hand_in", "car_in", "car_out"]
    # seqs = ["hand_out", "hand_in", "car_in", "car_out"]
    # seq_len = 400

    save_dir = cfg.vis_dir
    save_dir = os.path.join("./seq_vis_real", save_dir)
    
    for seq in seqs:
        seq_root = os.path.join(dataset_root, seq, "color")
        # seq_len = len(os.listdir(seq_root))
        imgs = sorted(os.listdir(seq_root))
        seq_dir = os.path.join(save_dir, seq)
        os.makedirs(seq_dir, exist_ok=True)
        print(seq_dir)
        for idx, img_name in enumerate(imgs):
            rgb_0 = read_image(os.path.join(seq_root, img_name))
            rgbs = rgb_transform(rgb_0).unsqueeze(0)
            feats = model(rgbs.cuda())

            feats_0 = nn_F.normalize(nn_F.interpolate(feats[0:1].detach(), scale_factor=4), dim=1)
            _, c, h, w = feats_0.shape
            feats_0 = feats_0.squeeze(0).permute((1,2,0)).detach().cpu().numpy()
            feats_0 = feats_0.reshape(-1, c)
            if idx == 0:
                pca = PCA(n_components=3)
                pca.fit(feats_0)

            pca_features = pca.transform(feats_0)
            pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
            pca_features = pca_features * 255
            res_pred = Image.fromarray(pca_features.reshape(h, w, 3).astype(np.uint8))
            res_pred.save(os.path.join(seq_dir, img_name))
            # np.save(os.path.join(save_dir, f"{i}.npy"), feats[0].detach().cpu().numpy())
            # feats_0 = nn_F.normalize(nn_F.interpolate(feats[0:1].detach(), scale_factor=4), dim=1)
            # viz_feat(feats_0, os.path.join(seq_dir, img_name))
        
if __name__ == "__main__":
    main()
