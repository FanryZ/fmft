from datetime import datetime
import argparse
import sys
import os
import cv2

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


def get_correspondences(feat_map1, feat_map2):
    """
    Finds correspondences using cosine similarity.
    
    Args:
        feat_map1 (torch.Tensor): Feature map from the first image (C x H x W).
        feat_map2 (torch.Tensor): Feature map from the second image (C x H x W).
    Returns:
        tuple: A tuple of two tensors representing matched keypoint coordinates.
    """
    C, H, W = feat_map1.shape
    feat_map1_flat = feat_map1.view(C, -1).T  # (H*W) x C
    feat_map2_flat = feat_map2.view(C, -1)  # C x (H*W)
    
    # Cosine similarity
    similarity_matrix = torch.matmul(feat_map1_flat, feat_map2_flat)
    
    # Find best matches
    matches1 = torch.argmax(similarity_matrix, dim=1) # indices of matches in feat_map2
    
    # Convert flat indices to 2D coordinates
    coords1_flat_idx = torch.arange(H*W)
    coords1_y = coords1_flat_idx // W
    coords1_x = coords1_flat_idx % W
    coords1 = torch.stack([coords1_x, coords1_y], dim=1)
    
    coords2_y = matches1 // W
    coords2_x = matches1 % W
    coords2 = torch.stack([coords2_x, coords2_y], dim=1)
    
    return coords1, coords2


def estimate_pose(coords1, coords2, K):
    """
    Estimates relative pose (R, t) from matched keypoints.
    
    Args:
        coords1 (torch.Tensor): Matched keypoints from image 1 (N x 2).
        coords2 (torch.Tensor): Matched keypoints from image 2 (N x 2).
        K (np.ndarray): The camera intrinsic matrix (3 x 3).
        
    Returns:
        tuple: A tuple of the rotation matrix (3x3) and translation vector (3x1).
    """
    pts1 = coords1.numpy().astype(np.float32)
    pts2 = coords2.numpy().astype(np.float32)
    
    E, mask = cv2.findEssentialMat(pts1, pts2, K, cv2.RANSAC, 0.999, 1.0, maxIters=1000)
    
    if E is None:
        return None, None
        
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
    
    return R, t


def rotation_error(R_est, R_gt):
    """
    Calculates the angular error between two rotation matrices.
    
    Args:
        R_est (np.ndarray): Estimated rotation matrix.
        R_gt (np.ndarray): Ground truth rotation matrix.
        
    Returns:
        float: Angular error in degrees.
    """
    # The relative rotation between the two matrices
    R_rel = np.dot(R_est.T, R_gt)
    
    # The trace of the relative rotation matrix is related to the rotation angle
    # trace(R) = 1 + 2 * cos(theta)
    trace = np.trace(R_rel)
    cos_theta = (trace - 1) / 2
    
    # Handle potential floating-point errors
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    error_rad = np.arccos(cos_theta)
    error_deg = np.rad2deg(error_rad)
    
    return error_deg


def translation_error(t_est, t_gt):
    """
    Calculates the angular error between two translation vectors.

    Args:
        t_est (np.ndarray): Estimated translation vector.
        t_gt (np.ndarray): Ground truth translation vector.

    Returns:
        float: Angular error in degrees.
    """
    # Normalize vectors
    t_est_norm = t_est / np.linalg.norm(t_est)
    t_gt_norm = t_gt / np.linalg.norm(t_gt)
    
    # Dot product
    dot_product = np.dot(t_est_norm.flatten(), t_gt_norm.flatten())
    
    # Handle potential floating-point errors
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    error_rad = np.arccos(dot_product)
    error_deg = np.rad2deg(error_rad)
    
    return error_deg


def get_total_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params
get_total_params = lambda model: sum(p.numel() for p in model.parameters())

@hydra.main("./configs", "scannet_pose", None)
def main(cfg: DictConfig):
    print(f"Config: \n {OmegaConf.to_yaml(cfg)}")
    features = []

    # ===== Get model and dataset ====
    model = instantiate(cfg.backbone, output="dense", return_multilayer=cfg.multilayer)
    model = model.to("cuda")
    dataset = ScanNetPairsDataset()

    # extract features
    err_2d = []

    R_gts = []
    R_ests = []
    rot_errs = []
    trans_errs = []
    
    for i in tqdm(range(len(dataset))):
        instance = dataset.__getitem__(i)
        rgbs = torch.stack((instance["rgb_0"], instance["rgb_1"]), dim=0)
        deps = torch.stack((instance["depth_0"], instance["depth_1"]), dim=0)

        h, w = rgbs.shape[2], rgbs.shape[3]
        K_mat = instance["K"].clone()
        # Rt_gt = instance["Rt_1"].float()[:3, :4]
        # R_gt.append(Rt_gt[:3, :3])
        Rt_gt = instance["Rt_1"].numpy()

        feats = model(rgbs.cuda())
        if cfg.multilayer:
            feats = torch.cat(feats, dim=1)

        # scale depth and intrinsics
        feats = feats.detach().cpu()
        feat_0, feat_1 = feats[0], feats[1]
        
        deps = nn_F.interpolate(deps, scale_factor=cfg.scale_factor)
        K_mat[:2, :] *= cfg.scale_factor

        # compute corr
        corr_xyz0, corr_xyz1, corr_dist = estimate_correspondence_depth(
            feats[0], feats[1], deps[0], deps[1], K_mat.clone(), cfg.num_corr
        )
        uv_0 = project_3dto2d(corr_xyz0, K_mat.clone())
        uv_1 = project_3dto2d(corr_xyz1, K_mat.clone())

        R, t = estimate_pose(uv_0, uv_1, K_mat.numpy())
        Rt_est = np.hstack((R, t))
        # print(Rt_est)
        R_gts.append(Rt_gt)
        R_ests.append(Rt_est)
        rot_errs.append(rotation_error(R, Rt_gt[:3, :3]))
        trans_errs.append(translation_error(t, Rt_gt[:3, 3]))

    print("\n--- Summary Metrics ---")
    print(f"Mean Angular Rotation Error: {np.mean(rot_errs):.2f} degrees")
    print(f"Median Angular Rotation Error: {np.median(rot_errs):.2f} degrees")
    print(f"Percentage of poses with < 5 deg error: {np.sum(np.array(rot_errs) < 5) / len(rot_errs) * 100:.2f}%")
    print(f"Percentage of poses with < 10 deg error: {np.sum(np.array(rot_errs) < 10) / len(rot_errs) * 100:.2f}%")
    print(f"Percentage of poses with < 15 deg error: {np.sum(np.array(rot_errs) < 15) / len(rot_errs) * 100:.2f}%")

    print(f"Mean Angular Translation Error: {np.mean(trans_errs):.2f} degrees")
    print(f"Median Angular Translation Error: {np.median(trans_errs):.2f} degrees")
    results = [
        f"{np.mean(rot_errs):.2f}",
        f"{np.sum(np.array(rot_errs) < 5) / len(rot_errs) * 100:.2f}",
        f"{np.sum(np.array(rot_errs) < 10) / len(rot_errs) * 100:.2f}",
        f"{np.sum(np.array(rot_errs) < 15) / len(rot_errs) * 100:.2f}",
        f"{np.mean(trans_errs):.2f}",
    ]

    np.save("rot_errs.npy", np.array(rot_errs))
    np.save("trans_errs.npy", np.array(trans_errs))

    time = datetime.now().strftime("%d%m%Y-%H%M")
    exp_info = cfg.backbone
    results = ", ".join(results)
    log = f"{time}, {exp_info}, scannet_pose, {results} \n"
    with open(cfg.log_file, "a") as f:
        f.write(log)
    

if __name__ == "__main__":
    main()
