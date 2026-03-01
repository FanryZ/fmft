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
from hydra import initialize, compose
from typing import List, Optional, Tuple
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.decomposition import PCA
from PIL import Image
from easydict import EasyDict
import json
from pathlib import Path

from evals.datasets.scannet_pairs import ScanNetPairsDataset

def get_model(cfg_path: str, cfg_name: str, overrides: Optional[List[str]] = None):
    with initialize(version_base=None, config_path=cfg_path):
        cfg = compose(config_name=cfg_name, overrides=overrides or [])
    node = cfg.get("backbone")
    print(node)
    model = instantiate(node, output="dense", return_multilayer=cfg.multilayer)
    return model, cfg

def _pca_to_rgb(t: torch.Tensor) -> torch.Tensor:
    assert t.ndim == 4, f"expect [B,C,H,W], got {t.shape}"
    B, C, H, W = t.shape
    X = t.permute(0, 2, 3, 1).reshape(-1, C).detach().float().cpu().numpy()  # [B*H*W, C]

    pca = PCA(n_components=3)
    X3 = pca.fit_transform(X)  # [B*H*W, 3]

    mn, mx = X3.min(), X3.max()
    X3 = (X3 - mn) / (mx - mn)
    X3 = (X3 * 255).astype(np.uint8)

    # 还原成 [B,3,H,W]
    img = torch.from_numpy(X3).view(B, H, W, 3).permute(0, 3, 1, 2).contiguous()
    return img  # uint8

def viz_feat(feat: torch.Tensor, our_feat: torch.Tensor, file_path_0):
    """
    将 feat 和 our_feat 各自做 PCA(->3通道伪彩), 取各自的第0/第1张，
    拼成 2x2 大图:  [feat0 | feat1]
                    [our0  | our1 ]
    并保存到 file_path_0.
    """
    file_path_0 = Path(file_path_0)

    # --- PCA -> RGB ---
    feat_rgb = _pca_to_rgb(feat)         # [B,3,H,W], uint8
    our_feat_rgb = _pca_to_rgb(our_feat) # [B,3,H,W], uint8

    # --- 取需要的帧（尽量拿到两张，不足就复用） ---
    def take_two(img3chw: torch.Tensor):
        B, _, H, W = img3chw.shape
        if B == 0:
            raise ValueError("Empty batch in features.")
        if B == 1:
            return [img3chw[0], img3chw[0]]
        return [img3chw[0], img3chw[1]]

    f0, f1 = take_two(feat_rgb)
    o0, o1 = take_two(our_feat_rgb)

    # --- 转 PIL.Image ---
    def to_pil(x3hw: torch.Tensor) -> Image.Image:
        x = x3hw.permute(1, 2, 0).cpu().numpy()  # [H,W,3]
        return Image.fromarray(x.astype(np.uint8), mode="RGB")

    im_f0, im_f1 = to_pil(f0), to_pil(f1)
    im_o0, im_o1 = to_pil(o0), to_pil(o1)

    # --- 拼接 2x2 画布 ---
    W_img, H_img = im_f0.size
    canvas = Image.new("RGB", (W_img * 2, H_img * 2), color=(0, 0, 0))
    canvas.paste(im_f0, (0, 0))              # 左上: feat[0]
    canvas.paste(im_f1, (W_img, 0))          # 右上: feat[1]
    canvas.paste(im_o0, (0, H_img))          # 左下: our_feat[0]
    canvas.paste(im_o1, (W_img, H_img))      # 右下: our_feat[1]

    file_path_0.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(file_path_0)
    print(f"[Saved] {file_path_0.resolve()}")

# def viz_feat(feat, our_feat, file_path_0):
#     B, C, H, W = feat.shape
#     # [B, C, H, W] -> [B*H*W, C]
#     print(B, C, H, W)
#     X = feat.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()
#     projected_featmap = X.reshape(-1, X.shape[-1])

#     pca = PCA(n_components=3)
#     pca.fit(projected_featmap)
#     pca_features = pca.transform(projected_featmap)
#     print('shape of projected: ', projected_featmap.shape)
#     print('shape of pca features: ', pca_features.shape)
#     pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
#     pca_features = pca_features * 255
#     pca_features = torch.from_numpy(pca_features).reshape(B, H, W, 3).permute(0, 3, 1, 2)
#     # print(np.array(pca_features[0]).shape)
#     res_pred_1 = Image.fromarray(np.array(pca_features[0].permute(1, 2, 0)).astype(np.uint8))
#     res_pred_2 = Image.fromarray(np.array(pca_features[1].permute(1, 2, 0)).astype(np.uint8))

def main():
    features = []
    pair_list = []
    model_name = "corr_b14"

    # ===== Get model and dataset ====
    model, cfg = get_model(cfg_path="./configs", cfg_name="feats_correspondence_vis",
                      overrides=[f"backbone={model_name}"])
    model = model.to("cuda")
    # print(cfg)
    
    our_model, cfg1 = get_model(cfg_path="./configs", cfg_name="feats_correspondence_vis",
                      overrides=["backbone=dinov2_b14_ft_conv"])
    our_model = our_model.to("cuda")
    # print(cfg1)
    dataset = ScanNetPairsDataset()

    # extract features
    for i in tqdm(range(len(dataset))):
        instance = dataset.__getitem__(i)
        pair_list.append([f"{instance['sequence_id']}_{instance['frame_0']}", f"{instance['sequence_id']}_{instance['frame_1']}"])
        rgbs = torch.stack((instance["rgb_0"], instance["rgb_1"]), dim=0)

        feats = model(rgbs.cuda())
        our_feats = our_model(rgbs.cuda())

        vis_dir = f"feat_map_vis_scannet/{model_name}"
        os.makedirs(vis_dir, exist_ok=True)
        save_path_0 = os.path.join(vis_dir, f"{instance['sequence_id']}_{instance['frame_0']}_{instance['frame_1']}.png")
        # save_path_1 = os.path.join(vis_dir, f"{instance['sequence_id']}_{instance['frame_1']}.png")
        
        # ours_save_path_0 = os.path.join(vis_dir, f"{instance['sequence_id']}_{instance['frame_0']}_our.png")
        # ours_save_path_1 = os.path.join(vis_dir, f"{instance['sequence_id']}_{instance['frame_1']}_our.png")
        
        feats_0 = nn_F.normalize(nn_F.interpolate(feats[0:1].detach(), scale_factor=4), dim=1)
        feats_1 = nn_F.normalize(nn_F.interpolate(feats[1:2].detach(), scale_factor=4), dim=1)
        
        feats_2 = nn_F.normalize(nn_F.interpolate(our_feats[0:1].detach(), scale_factor=4), dim=1)
        feats_3 = nn_F.normalize(nn_F.interpolate(our_feats[1:2].detach(), scale_factor=4), dim=1)
        feats_combined = torch.cat([feats_0, feats_1], dim=0)
        our_feats_combined = torch.cat([feats_2, feats_3], dim=0)
        print(feats_combined.shape)
        if not os.path.exists(save_path_0):
            viz_feat(feats_combined, our_feats_combined, save_path_0)
            # viz_feat(, ours_save_path_0, ours_save_path_1)
        # assert False
    with open(f"/home/fanry/Desktop/fmft/probe3d/{vis_dir}/pair_list", 'w') as f:
        json.dump(pair_list, f)

if __name__ == "__main__":
    main()
