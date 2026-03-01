import hydra
import torch
import pandas as pd
import torch.nn.modules.utils as nn_utils
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from pathlib import Path
import time
import torch.nn.modules.utils as nn_utils
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
import cv2
import json
import torch.nn.functional as F
import torchvision.transforms.functional as VF

from evals.datasets.tapvid_video import TapVidDataset
from evals.utils.tracking_metrics import compute_tapvid_metrics_for_video
from evals.utils.tracking_model import ModelInference, Tracker
from datetime import datetime

device = 'cuda'

def tracking_with_feats(feats, video_config, images, patch_size=14, stride=7, device="cuda:1"):
    tracker = Tracker(feats, images, dino_patch_size=patch_size, stride=stride, device=device)
    model_inference = ModelInference(
        model=tracker,
        range_normalizer=tracker.range_normalizer,
        anchor_cosine_similarity_threshold=0.7,
        cosine_similarity_threshold=0.6,
    )
    # rescale sizes
    rescale_sizes = [tracker.video.shape[-1], tracker.video.shape[-2]]
    rescale_factor_x = rescale_sizes[0] / video_config["w"]
    rescale_factor_y = rescale_sizes[1] / video_config["h"]
    # query points
    query_points_dict = {}
    for frame_idx, q_pts_at_frame in video_config["query_points"].items():
        query_points_dict[frame_idx] = [
            [rescale_factor_x * q[0], rescale_factor_y * q[1], frame_idx]
            for q in q_pts_at_frame
        ]
    trajectories_dict, occlusions_dict = {}, {}
    for frame_idx in tqdm(sorted(query_points_dict.keys()), desc="Tracking"):
        qpts = torch.tensor(query_points_dict[frame_idx], dtype=torch.float32, device=device)
        trajs, occs = model_inference.infer(query_points=qpts)
        trajectories_dict[frame_idx] = trajs[..., :2].cpu().detach().numpy()
        occlusions_dict[frame_idx] = occs.cpu().detach().numpy()
    # metrics
    metrics = compute_tapvid_metrics_for_video(
        trajectories_dict=trajectories_dict,
        occlusions_dict=occlusions_dict,
        video_idx=video_config["video_idx"],
        benchmark_data={"videos": [video_config]},
        pred_video_sizes=rescale_sizes,
    )
    metrics["video_idx"] = video_config["video_idx"]
    return metrics

@hydra.main("./configs", "tapvid_video", None)
def main(cfg: DictConfig):
    print(f"Config: \n {OmegaConf.to_yaml(cfg)}")
    # ===== Get model and dataset ====
    model = instantiate(cfg.backbone, output="dense", return_multilayer=cfg.multilayer)
    model = model.to("cuda:0").to(torch.float32).eval()
    patch_size = model.patch_size
    stride = patch_size // 2

    # image_size = (480, 854)
    # image_size = ((image_size[0] // patch_size), image_size[1] // patch_size)
    dataset = TapVidDataset(patch_size=patch_size, stride=stride, resize_hw = (400, 712))
    metrics_list = []
    # extract features

    range_num = len(dataset) if cfg.video_num <= 0 else min(cfg.video_num, len(dataset))
    for i in tqdm(range(range_num)):
        instance = dataset.__getitem__(i)
        rgbs = instance["rgbs"].to("cuda:0")  # (T,3,H,W)
        with torch.no_grad():
            feats = model(rgbs)
            H, W = rgbs.shape[2], rgbs.shape[3]
            ph = 1 + (H - patch_size) // stride
            pw = 1 + (W - patch_size) // stride
            if feats.dim() == 4:  # B x C x h x w
                if feats.shape[2] != ph or feats.shape[3] != pw:
                    feats = torch.nn.functional.interpolate(feats, size=(ph, pw), mode="bilinear", align_corners=False)

        metrics = tracking_with_feats(
            feats=feats,
            video_config=instance["video_config"],
            images=rgbs,
            patch_size=patch_size,
            stride=stride,
            device="cuda:0",
        )
        metrics_list.append(metrics)
        print(metrics)
    metrics_track = pd.DataFrame(metrics_list)
    metrics_track.set_index(['video_idx'], inplace=True)

    time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(f'./probeRes/video/{time}.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_track.to_csv(output_path)
    mean = metrics_track.mean()
    
    exp_info = cfg.backbone
    print(mean)
    results = ", ".join(mean.values.astype(str))
    log = f"{time}, {exp_info}, tapvid_video, {results} \n"
    with open(cfg.log_file, "a") as f:
        f.write(log)
    print(mean)

if __name__ == "__main__":
    with torch.no_grad():
        main()
