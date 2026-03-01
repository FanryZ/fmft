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

from evals.datasets.builder import build_loader

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
        if B == 2:
            return [img3chw[0], img3chw[1]]
        return [img3chw[i] for i in range(B)]

    ff = take_two(feat_rgb)
    oo = take_two(our_feat_rgb)

    # --- 转 PIL.Image ---
    def to_pil(x3hw: torch.Tensor) -> Image.Image:
        x = x3hw.permute(1, 2, 0).cpu().numpy()  # [H,W,3]
        return Image.fromarray(x.astype(np.uint8), mode="RGB")

    ff_pil = [to_pil(_) for _ in ff]
    oo_pil = [to_pil(_) for _ in oo]

    # --- 拼接 2x2 画布 ---
    W_img, H_img = ff_pil[0].size
    canvas = Image.new("RGB", (W_img * 4, H_img * 2), color=(0, 0, 0))
    for _ in range(len(ff_pil)):
        canvas.paste(ff_pil[_], (_ * W_img, 0))             
        canvas.paste(oo_pil[_], (_ * W_img, H_img))

    file_path_0.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(file_path_0)
    print(f"[Saved] {file_path_0.resolve()}")

def main():
    features = []
    pair_list = []

    # ===== Get model and dataset ====
    model_name = "dinov2_b14"
    model, cfg = get_model(cfg_path="./configs", cfg_name="feats_correspondence_vis",
                      overrides=[f"backbone={model_name}"])
    model = model.to("cuda")
    # print(cfg)
    
    our_model, cfg1 = get_model(cfg_path="./configs", cfg_name="feats_correspondence_vis",
                      overrides=["backbone=dinov2_b14_ft_conv"])
    our_model = our_model.to("cuda")
    # print(cfg1)
    loader = build_loader(cfg.dataset, "test", 1, 1, pair_dataset=True)
    _ = loader.dataset.__getitem__(0)

    # extract features
    vis_dir = f"single_feat_map_vis_navi/{model_name}"
    os.makedirs(vis_dir, exist_ok=True)
    counter = 0
    feat_buffer = []
    our_feat_buffer = []
    for batch in tqdm(loader):
        if batch['obj_id_0'][0] != "hand_drill_cordless_blue_black":
            continue
        counter += 2
        feat_0 = model(batch["image_0"].cuda())
        feat_1 = model(batch["image_1"].cuda())
        feat_buffer.append(feat_0)
        feat_buffer.append(feat_1)
        our_feat_0 = our_model(batch["image_0"].cuda())
        our_feat_1 = our_model(batch["image_1"].cuda())
        our_feat_buffer.append(our_feat_0)
        our_feat_buffer.append(our_feat_1)
        # name_buffer.append([f"{batch['obj_id_0'][0]}_{batch['scene_id_0'][0]}_{batch['img_id_0'][0]}.png", f"{batch['obj_id_1'][0]}_{batch['scene_id_1'][0]}_{batch['img_id_1'][0]}.png"])
        pair_list.append([f"{batch['obj_id_0'][0]}_{batch['scene_id_0'][0]}_{batch['img_id_0'][0]}.png", f"{batch['obj_id_1'][0]}_{batch['scene_id_1'][0]}_{batch['img_id_1'][0]}.png"])

        if counter!=4: 
            continue
        else:
            counter = 0
        
        save_path_0 = os.path.join(vis_dir, f"{batch['obj_id_0'][0]}_{batch['img_id_0'][0]}_{batch['img_id_1'][0]}.png")
        # save_path_1 = os.path.join(vis_dir, f"{batch['obj_id_1'][0]}_{batch['scene_id_1'][0]}_{batch['img_id_1'][0]}.png")
        # ours_save_path_0 =  os.path.join(vis_dir, f"{batch['obj_id_0'][0]}_{batch['scene_id_0'][0]}_{batch['img_id_0'][0]}_our.png")
        # ours_save_path_1 = os.path.join(vis_dir, f"{batch['obj_id_1'][0]}_{batch['scene_id_1'][0]}_{batch['img_id_1'][0]}_our.png")
        
        # feats_0 = nn_F.normalize(nn_F.interpolate(feat_0.detach(), scale_factor=4), dim=1)
        # feats_1 = nn_F.normalize(nn_F.interpolate(feat_1.detach(), scale_factor=4), dim=1)
        
        # feats_2 = nn_F.normalize(nn_F.interpolate(our_feat_0.detach(), scale_factor=4), dim=1)
        # feats_3 = nn_F.normalize(nn_F.interpolate(our_feat_1.detach(), scale_factor=4), dim=1)
        # feats_combined = torch.cat([feats_0, feats_1], dim=0)
        # our_feats_combined = torch.cat([feats_2, feats_3], dim=0)
        feats_combined = nn_F.normalize(nn_F.interpolate(feat_buffer[0].detach(), scale_factor=4), dim=1)
        for _ in feat_buffer[1:]:
            _ =  nn_F.normalize(nn_F.interpolate(_.detach(), scale_factor=4), dim=1)
            feats_combined = torch.cat([feats_combined, _], dim=0)
        our_feats_combined = nn_F.normalize(nn_F.interpolate(our_feat_buffer[0].detach(), scale_factor=4), dim=1)
        for _ in our_feat_buffer[1:]:
            _ =  nn_F.normalize(nn_F.interpolate(_.detach(), scale_factor=4), dim=1)
            our_feats_combined = torch.cat([our_feats_combined, _], dim=0)
        
        # print(feats_combined.shape)
        if not os.path.exists(save_path_0):
            viz_feat(feats_combined, our_feats_combined, save_path_0)
            # viz_feat(our_feats_combined, ours_save_path_0, ours_save_path_1)
        # assert False
        feat_buffer = []
        our_feat_buffer = []
    with open(f"/home/fanry/Desktop/fmft/probe3d/{vis_dir}/pair_list", 'w') as f:
        json.dump(pair_list, f)

if __name__ == "__main__":
    main()
