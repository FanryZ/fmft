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
import glob
import json
import os
from pathlib import Path
import pickle

import numpy as np
import torch
import torchvision
from PIL import Image

from .utils import (
    bbox_crop,
    camera_matrices_from_annotation,
    compute_normal,
    get_grid,
    get_navi_transforms,
    read_depth,
    read_image,
)


class TapvidPair(torch.utils.data.Dataset):
    def __init__(
        self,
        image_mean="imagenet",
        patch_size=14,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.pair_root = Path("/home/fanry/Desktop/fmft/locft/tapvidPair")
        self.object_id = glob.glob(str(self.pair_root / "*"))
        self.object_id = [os.path.basename(_id) for _id in self.object_id]

        self.pairs = []
        for obj_id in self.object_id:
            obj_dir = self.pair_root / obj_id
            pairs = glob.glob(str(obj_dir / "*"))
            pairs = [os.path.basename(_id) for _id in pairs]
            for pair in pairs:
                pair_dir = obj_dir / pair
                images = glob.glob(str(pair_dir / "*.jpg"))
                image_ids = [os.path.basename(_id).split(".")[0] for _id in images]
                json_path = str(pair_dir / "corres" / "overlap_points.json")
                self.pairs.append({
                    "obj_id": obj_id,
                    "pair": pair,
                    "images": image_ids,
                    "corres": json_path,
                })
        # get transforms
        # resize_hw=(480, 854)
        resize_hw=(400, 712)
        self.resize_hw = list(resize_hw)
        h, w = self.resize_hw
        self.resize_hw[0] = h // patch_size * patch_size
        self.resize_hw[1] = w // patch_size * patch_size
        self.rescale_factor = [self.resize_hw[0] / h, self.resize_hw[1] / w]
        # self.transform = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        self.rgb_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.resize_hw, interpolation=torchvision.transforms.InterpolationMode.LANCZOS),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        # image_size = (512, 512)
        # t_fns = get_navi_transforms(
        #     image_mean,
        #     image_size=image_size
        # )
        # self.image_transform, self.target_transform, self.shared_transform = t_fns

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = self.pairs[index]
        img_id_0 = pair["images"][0]
        img_id_1 = pair["images"][1]
        image_path_0 = self.pair_root / pair["obj_id"] / pair["pair"] / f"{img_id_0}.jpg"
        image_path_1 = self.pair_root / pair["obj_id"] / pair["pair"] / f"{img_id_1}.jpg"

        inst_0 = self.get_single(image_path_0)
        inst_1 = self.get_single(image_path_1)
        json_data = json.load(open(pair["corres"]))
        corres = np.array([[x["image1"]["point"], x["image2"]["point"]] for x in json_data])
        corres = corres * np.array(self.rescale_factor)

        output = {}
        output["meta"] = pair
        output["rgb_0"] = inst_0
        output["rgb_1"] = inst_1
        output["rgb_path_0"] = image_path_0
        output["rgb_path_1"] = image_path_1
        output["corres"] = corres
        return output

    def get_single(self, image_path):
        patch_size = self.patch_size
        rgb_resized = self.rgb_transform(Image.open(image_path).convert("RGB"))
        return rgb_resized


if __name__ == "__main__":
    dataset = TapvidPair()
    