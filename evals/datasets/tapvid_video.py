import os
import pickle
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class TapVidDataset(Dataset):
    def __init__(self, root="/data/fanry/Desktop/fmft/probe3d/data/tapvid-davis",
                 benchmark_path="/data/fanry/Desktop/fmft/probe3d/data/tapvid_davis_data_strided.pkl",
                 patch_size=14, stride=7, resize_hw=(480, 854)):
        super().__init__()
        self.root = root
        self.patch_size = patch_size
        self.stride = stride
        self.resize_hw = list(resize_hw)

        h, w = self.resize_hw
        if h % patch_size != 0 or w % patch_size != 0:
            print(f"Warning: ({h}, {w}) not divisible by patch_size {patch_size}")
            self.resize_hw[0] = h // patch_size * patch_size
            self.resize_hw[1] = w // patch_size * patch_size
            print(f"New image size: {self.resize_hw}")
        # ground truth
        with open(benchmark_path, "rb") as f:
            self.benchmark_config = pickle.load(f)

        self.rgb_transform = T.Compose([
            T.Resize(self.resize_hw, interpolation=T.InterpolationMode.LANCZOS),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.instances = self.get_instances()

    def __len__(self):
        return len(self.instances)

    def get_instances(self):
        instances = []
        for vid in sorted(os.listdir(self.root)):
            video_root = os.path.join(self.root, vid, "video")
            if not os.path.isdir(video_root):
                continue
            img_files = sorted(os.listdir(video_root))
            instances.append((vid, img_files))
        return instances

    def __getitem__(self, idx):
        vid, img_files = self.instances[idx]
        rgbs = []
        for fn in img_files:
            img = Image.open(os.path.join(self.root, vid, "video", fn)).convert("RGB")
            rgbs.append(self.rgb_transform(img))
        rgbs = torch.stack(rgbs)  # (T, 3, H, W)
        # ground truth
        video_idx = int(vid)
        gt_config = None
        for vcfg in self.benchmark_config["videos"]:
            if vcfg["video_idx"] == video_idx:
                gt_config = vcfg
                break

        return {
            "video_idx": video_idx,
            "rgbs": rgbs,      
            "video_config": gt_config,
        }