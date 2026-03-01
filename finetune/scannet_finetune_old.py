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

import hydra
import numpy as np
import torch
import torch.nn.functional as nn_F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

# from evals.datasets.scannet_pairs import ScanNetPairsDataset
from evals.datasets.scannet_pair_base import ScanNetPairBase
from evals.utils.vggt_utils import patch_track, track_point_filter
from metric_learn import ITML

def interpolate_features(feature, pts, image_size):
    """
        Args:
        feature (torch.Tensor): [N, H, W, C]
        pts (torch.Tensor): [N, N_p, 2]
        image_size (tuple): (2, )
        Return:
        interpolated_feature: [N, N_p, C]
    """
    h, w = image_size
    pts = pts[:, None, ...]
    pts_grid = torch.zeros_like(pts).float()
    pts_grid[:, :, :, 0] = pts[:, :, :, 0] / w * 2 - 1.
    pts_grid[:, :, :, 1] = pts[:, :, :, 1] / h * 2 - 1.
    descs = nn_F.grid_sample(feature, pts_grid, "bilinear", align_corners=True)
    descs = descs[:, :, 0, :].permute(0, 2, 1)
    return descs


def metric_learning_finetune(feats_0, feats_1, neg_pair_num = 2):
    """
    Args:
        feats_0 (np.ndarray): [N, C]
        feats_1 (np.ndarray): [N, C]
    """
    feats = np.concatenate([feats_0, feats_1], axis=0)
    pos_pair = []
    for i in range(feats_0.shape[0]):
        pos_pair.append([i, i + feats_0.shape[0]])
    neg_pair = []
    for i in range(feats_0.shape[0] * neg_pair_num):
        neg_1 = np.random.randint(0, feats_0.shape[0] - 2)
        neg_2 = np.random.randint(neg_1 + 1, feats_0.shape[0] - 1) + feats_0.shape[0]
        neg_pair.append([neg_1, neg_2])
    y = [1] * len(pos_pair) + [-1] * len(neg_pair)
    pairs = np.concatenate([np.array(pos_pair), np.array(neg_pair)], axis=0)

    model = ITML(preprocessor=feats, max_iter=20, verbose=True)
    model.fit(pairs, y)
    return model, model.pair_score(pairs)


def metric_learning_distance(feats_0, feats_1, points_0, points_1, neg_pair_num=2, min_distance_ratio=0.3):
    """
    Args:
        feats_0 (np.ndarray): [N, C] features from first image
        feats_1 (np.ndarray): [N, C] features from second image
        points_0 (np.ndarray): [N, 2] 2D points from first image (normalized to [0,1]x[0,1])
        points_1 (np.ndarray): [N, 2] 2D points from second image (normalized to [0,1]x[0,1])
        neg_pair_num: number of negative pairs per positive pair
        min_distance_ratio: minimum distance ratio (relative to image diagonal) for negative pairs
    """
    assert len(feats_0) == len(feats_1) == len(points_0) == len(points_1)
    num_points = len(feats_0)
    
    # Combine features for the metric learning model
    feats = np.concatenate([feats_0, feats_1], axis=0)
    
    # Calculate image diagonal for distance normalization
    img_diagonal = np.linalg.norm([1.0, 1.0])  # Since points are in [0,1]x[0,1]
    min_distance = min_distance_ratio * img_diagonal
    
    # Positive pairs (matched points)
    pos_pairs = np.column_stack((np.arange(num_points), 
                               np.arange(num_points) + num_points))
    
    # Generate negative pairs
    neg_pairs = []
    for i in range(num_points):
        # Get the corresponding point in the second image
        corresponding_pt = points_1[i]
        
        # Calculate distances from the corresponding point to all other points in the second image
        distances = np.linalg.norm(points_1 - corresponding_pt, axis=1)
        
        # Find points that are sufficiently far from the corresponding point
        far_indices = np.where(distances > min_distance)[0]
        
        # Remove the current point's index if it's in the far_indices
        far_indices = far_indices[far_indices != i]
        
        # If we have enough far points, sample from them
        if len(far_indices) > 0:
            # Sample negative pairs
            selected = np.random.choice(far_indices, 
                                      size=min(neg_pair_num, len(far_indices)),
                                      replace=False)
            for idx in selected:
                # Create negative pair: point i from first image with far point from second image
                neg_pairs.append([i, num_points + idx])  # i is in first image, idx is in second image
    
    # Convert to numpy arrays
    pos_pairs = np.array(pos_pairs)
    neg_pairs = np.array(neg_pairs) if len(neg_pairs) > 0 else np.empty((0, 2), dtype=int)
    
    # Combine positive and negative pairs
    pairs = np.vstack([pos_pairs, neg_pairs]) if len(neg_pairs) > 0 else pos_pairs
    y = np.array([1] * len(pos_pairs) + [-1] * len(neg_pairs))
    
    # Train metric learning model
    model = ITML(preprocessor=feats, max_iter=20, verbose=True)
    model.fit(pairs, y)
    # Calculate score (e.g., accuracy on training pairs)
    return model, model.pair_score(pairs)

# @hydra.main("./configs", "scannet_correspondence", None)
@hydra.main("./configs", "scannet_finetune", None)
def main(cfg: DictConfig):
    print(f"Config: \n {OmegaConf.to_yaml(cfg)}")

    # ===== Get model and dataset ====
    model = instantiate(cfg.backbone, output="dense", return_multilayer=cfg.multilayer)
    model = model.to("cuda")
    dataset = ScanNetPairBase()
    loader = DataLoader(
        dataset, 8, num_workers=4, drop_last=False, pin_memory=True, shuffle=False
    )

    # extract features
    output_feats_0 = []
    output_feats_1 = []
    feats_org = []
    transform_mats = []
    for i in tqdm(range(len(dataset))):
        instance = dataset.__getitem__(i)
        rgbs = torch.stack((instance["rgb_0"], instance["rgb_1"]), dim=0)
        # instance["point_0"], instance["point_1"]
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
        # _, n, c, vh, vw = images.shape

        interpolated_feats_0 = interpolate_features(feats_0, points_masked0[None, ...], images.shape[-2:])
        interpolated_feats_1 = interpolate_features(feats_1, points_masked1[None, ...], images.shape[-2:])
        if cfg.normalize:
            interpolated_feats_0 = nn_F.normalize(interpolated_feats_0, dim=2)
            interpolated_feats_1 = nn_F.normalize(interpolated_feats_1, dim=2)
        
        print(interpolated_feats_0.shape)
        print(interpolated_feats_1.shape)
        feats_0 = interpolated_feats_0[0].detach().cpu().numpy()
        feats_1 = interpolated_feats_1[0].detach().cpu().numpy()
        output_feats_0.append(feats_0)
        output_feats_1.append(feats_1)
        feats_org.append(feats.detach().cpu().numpy())
        # metric_model, metric_model_score = metric_learning_finetune(feats_0, feats_1, cfg.neg_pair_num)
        
        points_masked0 = points_masked0.detach().cpu().numpy()
        points_masked1 = points_masked1.detach().cpu().numpy()
        metric_model, metric_model_score = metric_learning_distance(feats_0, feats_1, points_masked0, points_masked1, cfg.neg_pair_num)
        transform_mats.append(metric_model.components_)

        if cfg.multilayer:
            feats = torch.cat(feats, dim=1)

    output_feats = [np.stack([feat0, feat1]) for feat0, feat1 in zip(output_feats_0, output_feats_1)]
    transform_mats = np.stack(transform_mats)
    feats_org = np.stack(feats_org)
    
    # Save with numpy's savez_compressed for efficient storage
    model_name = cfg.backbone._target_.replace("evals.models.", "").replace(".", "_")
    if not cfg.normalize:
        model_name += "_raw"
    output_path = f"./exp/scannet_{model_name}_feats.npz"
    np.savez_compressed(output_path, *output_feats, transform_mats=transform_mats, feats_org=feats_org)
    print(f"Features saved to {output_path}")


if __name__ == "__main__":
    main()
