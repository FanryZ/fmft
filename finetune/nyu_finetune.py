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

from evals.datasets.nyu_pair import NYUPair
from evals.utils.vggt_utils import patch_track, track_point_filter, vis_track_points
from metric_learning import metric_learning_finetune_outlier, interpolate_features, outliers, conv_training


# @hydra.main("./configs", "scannet_correspondence", None)
@hydra.main("../configs", "nyu_finetune", None)
def main(cfg: DictConfig):
    print(f"Config: \n {OmegaConf.to_yaml(cfg)}")

    # ===== Get model and dataset ====
    model = instantiate(cfg.backbone, output="dense", return_multilayer=cfg.multilayer)
    model = model.to("cuda")
    dataset = NYUPair()

    range_num = len(dataset) if cfg.pair_num <= 0 else min(cfg.pair_num, len(dataset))
    for i in tqdm(range(range_num)):
        instance = dataset.__getitem__(i)
        rgbs = torch.stack((instance["rgb_0"], instance["rgb_1"]), dim=0)
        rgb_path_0 = instance["rgb_path_0"]
        rgb_path_1 = instance["rgb_path_1"]

        feats = model(rgbs.cuda())
        feats_0, feats_1 = torch.split(feats, [1, 1], dim=0)
        patch_size = model.patch_size
        
        images, track_list, vis_score, conf_score = patch_track([rgb_path_0, rgb_path_1], patch_size)
        points_masked0_1, points_masked1_1 = track_point_filter(track_list, vis_score, conf_score, 0.2)
        images, track_list, vis_score, conf_score = patch_track([rgb_path_1, rgb_path_0], patch_size)
        points_masked1_2, points_masked0_2 = track_point_filter(track_list, vis_score, conf_score, 0.2)
        points_masked0 = torch.cat((points_masked0_1, points_masked0_2), dim=0)
        points_masked1 = torch.cat((points_masked1_1, points_masked1_2), dim=0)
        vis_track_points(images, points_masked1, points_masked0)
        # _, n, c, vh, vw = images.shape
        # if cfg.normalize:
        #     feats_0 = nn_F.normalize(feats_0, dim=2)
        #     feats_1 = nn_F.normalize(feats_1, dim=2)

        outlier_points_0 = outliers(points_masked0, images.shape[-2:], patch_size)
        outlier_points_1 = outliers(points_masked1, images.shape[-2:], patch_size)
        conv_layer = conv_training(feats_0, feats_1, points_masked0[None, ...], points_masked1[None, ...], 
            outlier_points_0[None, ...], outlier_points_1[None, ...], images.shape[-2:], cfg.neg_pair_num, cfg.iter_num)

        feats_0 = conv_layer(feats_0)
        feats_1 = conv_layer(feats_1)
        
        interpolated_feats_0 = interpolate_features(feats_0, points_masked0[None, ...], images.shape[-2:])
        interpolated_feats_1 = interpolate_features(feats_1, points_masked1[None, ...], images.shape[-2:])
        outlier_feats0 = interpolate_features(feats_0, outlier_points_0[None, ...], images.shape[-2:])
        outlier_feats1 = interpolate_features(feats_1, outlier_points_1[None, ...], images.shape[-2:])

        np_feats_0 = interpolated_feats_0[0].detach().cpu().numpy()
        np_feats_1 = interpolated_feats_1[0].detach().cpu().numpy()
        np_outlier_feats0 = outlier_feats0[0].detach().cpu().numpy()
        np_outlier_feats1 = outlier_feats1[0].detach().cpu().numpy()

        metric_model, metric_model_score = metric_learning_finetune_outlier(
            np_feats_0, np_feats_1, np_outlier_feats0, np_outlier_feats1, cfg.neg_pair_num)
        transform_mat = torch.from_numpy(metric_model.components_.T).to(feats_0)
        # feats_0 = (feats_0.permute(0, 2, 3, 1) @ transform_mat).permute(0, 3, 1, 2)
        # feats_1 = (feats_1.permute(0, 2, 3, 1) @ transform_mat).permute(0, 3, 1, 2)

        # conv_feats_0 = conv_layer(feats_0)
        # conv_feats_1 = conv_layer(feats_1)   
        # if cfg.multilayer:
        #     feats = torch.cat(feats, dim=1)
        
        output_path = f"./exp/nyu_finetune"
        os.makedirs(output_path, exist_ok=True)
        torch.save(conv_layer.state_dict(), os.path.join(output_path, f"conv_layer_{i}.pth"))
        torch.save(transform_mat, os.path.join(output_path, f"proj_layer_{i}.pth"))

    print(f"Features saved to {output_path}")


if __name__ == "__main__":
    main()
