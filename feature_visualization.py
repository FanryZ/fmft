from datetime import datetime
import argparse
import sys
import os
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from evals.datasets.scannet_pairs import ScanNetPairsDataset
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def pca_to_2d_image(feat: torch.Tensor, file_path: Path):
    """
    Args:
        feat: [N, C, H, W]
        file_path: str
    """
    _, _, h, w = feat.shape
    feat = feat.squeeze(0).permute((1,2,0))
    projected_featmap = feat.reshape(-1, feat.shape[-1]).cpu()

    pca = PCA(n_components=2)
    pca.fit(projected_featmap)
    pca_features = pca.transform(projected_featmap)
    pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
    pca_features = pca_features * 255
    
    plt.figure(figsize=(6,6))
    plt.scatter(pca_features[:,0], pca_features[:,1], s=1, alpha=0.5)
    plt.title("PCA 2D projection")
    plt.savefig(file_path, dpi=300)
    plt.close()
    print("... saved PCA scatter plot to:", file_path)
    # res_pred = Image.fromarray(pca_features.reshape(h, w, 2).astype(np.uint8))
    # res_pred.save(file_path)
    # print("... saved to: ", file_path)


@hydra.main("./configs", "scannet_feat_map", None)
def main(cfg: DictConfig):
    print(f"Config: \n{OmegaConf.to_yaml(cfg)}")

    # ===== Get model and dataset ====
    model = instantiate(cfg.backbone, output="dense", return_multilayer=cfg.multilayer)
    model = model.to("cuda").eval()

    dataset = ScanNetPairsDataset()

    save_root = Path("feat_map_vis")
    save_root.mkdir(parents=True, exist_ok=True)

    # extract & visualize features
    with torch.inference_mode():
        for i in tqdm(range(len(dataset))):
            instance = dataset.__getitem__(i)
            rgbs = torch.stack((instance["rgb_0"], instance["rgb_1"]), dim=0)  # (2, C, H, W)

            feats = model(rgbs.cuda())
            feats_0 = F.normalize(F.interpolate(feats[0:1].detach(), scale_factor=4), dim=1)
            feats_1 = F.normalize(F.interpolate(feats[1:2].detach(), scale_factor=4), dim=1)
            pca_to_2d_image(feats_0, os.path.join(save_root, "sample.png"))
            break

    print(f"[Done] PCA visualizations saved to: {save_root.resolve()}")


if __name__ == "__main__":
    main()
