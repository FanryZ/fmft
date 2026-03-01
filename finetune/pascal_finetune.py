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

import hydra
import numpy as np
import torch
import torch.nn.functional as nn_F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as T
from PIL import Image

from metric_learning import metric_learning_finetune_outlier, interpolate_features, outliers, conv_training
from evaluate_pascal_pf import load_pascal_data, resize
from evals.utils.vggt_utils import vis_track_points


@hydra.main("../configs", "pascal_finetune", None)
def main(cfg: DictConfig):
    print(f"Config: \n {OmegaConf.to_yaml(cfg)}")
    imagenet_norm = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    # ===== Get model and dataset ====
    model = instantiate(cfg.backbone, output="dense", return_multilayer=cfg.multilayer)
    model = model.to("cuda")
    categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                'bus', 'car', 'cat', 'chair', 'cow',
                'diningtable', 'dog', 'horse', 'motorbike', 'person',
                'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    img_size = 700
    # img_size = 840
    patch_size = model.patch_size
    stride = patch_size
    img_size = (img_size // patch_size) * patch_size
    for i, cat in enumerate(categories):
        if cfg.select_idx >= 0 and cfg.select_idx != i:
            continue

        # files, kps, _ = load_pascal_data('./data/PF-dataset-PASCAL', size=img_size, category=cat, same_view=False)
        files, kps, _ = load_pascal_data('./data/PF-dataset-PASCAL', size=img_size, category=cat, same_view=True)
        img1 = Image.open(files[0]).convert('RGB')
        img1 = resize(img1, img_size, resize=True, to_pil=True, edge=False)
        img1_kps = kps[0]
        # img1_y, img1_x = img1_kps[:, 1].numpy(), img1_kps[:, 0].numpy()

        # Load image 2
        img2 = Image.open(files[1]).convert('RGB')
        img2 = resize(img2, img_size, resize=True, to_pil=True, edge=False)
        img2_kps = kps[1]

        # Get patch index for the keypoints
        # img2_y, img2_x = img2_kps[:, 1].numpy(), img2_kps[:, 0].numpy()
        
        img1 = torch.from_numpy(np.array(img1) / 255.).cuda().float().permute(2, 0, 1)
        img2 = torch.from_numpy(np.array(img2) / 255.).cuda().float().permute(2, 0, 1)
        img1_desc = model(imagenet_norm(img1[None]))
        img2_desc = model(imagenet_norm(img2[None]))

        vis = img1_kps[:, 2] * img2_kps[:, 2] > 0
        img1_kps = img1_kps[vis][:, :2].cuda()
        img2_kps = img2_kps[vis][:, :2].cuda()
        # vis_track_points(torch.stack([img1, img2])[None], img1_kps, img2_kps)
        
        outlier_points_0 = outliers(img1_kps, img1.shape[-2:], model.patch_size, margin=5)
        outlier_points_1 = outliers(img2_kps, img2.shape[-2:], model.patch_size, margin=5)
        conv_layer = conv_training(img1_desc, img2_desc, img1_kps[None, ...], img2_kps[None, ...], 
            outlier_points_0[None, ...], outlier_points_1[None, ...], img1.shape[-2:], 
            cfg.neg_pair_num, cfg.iter_num, cfg.temperature, cfg.weight_decay)

        feats_0 = conv_layer(img1_desc)
        feats_1 = conv_layer(img2_desc)
        
        interpolated_feats_0 = interpolate_features(feats_0, img1_kps[None, ...], img1.shape[-2:])
        interpolated_feats_1 = interpolate_features(feats_1, img2_kps[None, ...], img2.shape[-2:])
        outlier_feats0 = interpolate_features(feats_0, outlier_points_0[None, ...], img1.shape[-2:])
        outlier_feats1 = interpolate_features(feats_1, outlier_points_1[None, ...], img2.shape[-2:])

        np_feats_0 = interpolated_feats_0[0].detach().cpu().numpy()
        np_feats_1 = interpolated_feats_1[0].detach().cpu().numpy()
        np_outlier_feats0 = outlier_feats0[0].detach().cpu().numpy()
        np_outlier_feats1 = outlier_feats1[0].detach().cpu().numpy()

        metric_model, metric_model_score = metric_learning_finetune_outlier(
            np_feats_0, np_feats_1, np_outlier_feats0, np_outlier_feats1, cfg.neg_pair_num)
        transform_mat = torch.from_numpy(metric_model.components_.T).to(feats_0)
        
        output_path = f"./exp/pascal_finetune"
        os.makedirs(output_path, exist_ok=True)
        if cfg.select_idx >= 0:
            torch.save(conv_layer.state_dict(), os.path.join(output_path, f"conv_layer_select.pth"))
            torch.save(transform_mat, os.path.join(output_path, f"proj_layer_select.pth"))
        else:
            torch.save(conv_layer.state_dict(), os.path.join(output_path, f"conv_layer_{i}.pth"))
            torch.save(transform_mat, os.path.join(output_path, f"proj_layer_{i}.pth"))

    print(f"Features saved to {output_path}")


if __name__ == "__main__":
    main()
