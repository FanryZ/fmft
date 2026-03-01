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

import numpy as np
import torch
from torchvision import transforms

from .utils import (
    bbox_crop,
    camera_matrices_from_annotation,
    compute_normal,
    get_grid,
    get_navi_transforms,
    read_depth,
    read_image,
)


class ScannetPair(torch.utils.data.Dataset):
    def __init__(
        self,
    ):
        super().__init__()
        self.pair_root = Path("/data/fanry/Desktop/fmft/locft/scannetPair2")
        self.object_id = glob.glob(str(self.pair_root / "*"))
        self.object_id = [os.path.basename(_id) for _id in self.object_id]

        self.pairs = []
        for obj_id in self.object_id:
            obj_dir = self.pair_root / obj_id
            angles = glob.glob(str(obj_dir / "*"))
            angles = [os.path.basename(_id) for _id in angles]
            for angle in angles:
                angle_dir = obj_dir / angle
                pairs = glob.glob(str(angle_dir / "*"))
                pairs = [os.path.basename(_id) for _id in pairs]
                for pair in pairs:
                    pair_dir = angle_dir / pair
                    images = glob.glob(str(pair_dir / "*.jpg"))
                    image_ids = [os.path.basename(_id).split(".")[0] for _id in images]
                    # scene_id = open(str(pair_dir / "scene.txt")).read().strip()
                    self.pairs.append({
                        "obj_id": obj_id,
                        "angle": angle,
                        "pair": pair,
                        "images": image_ids,
                        # "scene_id": scene_id
                    })

        # get transforms
        self.transform = transforms.Compose(
            [
                transforms.Resize((480, 640)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = self.pairs[index]
        img_id_0 = pair["images"][0]
        img_id_1 = pair["images"][1]
        image_path_0 = self.pair_root / pair["obj_id"] / pair["angle"] / pair["pair"] / f"{img_id_0}.jpg"
        image_path_1 = self.pair_root / pair["obj_id"] / pair["angle"] / pair["pair"] / f"{img_id_1}.jpg"

        inst_0 = self.get_single(image_path_0)
        inst_1 = self.get_single(image_path_1)

        output = {}
        output["meta"] = pair
        output["rgb_0"] = inst_0
        output["rgb_1"] = inst_1
        output["rgb_path_0"] = image_path_0
        output["rgb_path_1"] = image_path_1
        return output

    def get_single(self, image_path):
        rgb_0 = read_image(image_path, exif_transpose=False)
        rgb_0 = self.transform(rgb_0)
        return rgb_0


if __name__ == "__main__":
    # Path to the JSON file
    import json
    import cv2
    import numpy as np
    import os
    from matplotlib import pyplot as plt

    dataset = ScannetPair()
    sample = dataset[0]
    
