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
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as transforms

from .utils import read_image


class ScanNetPairBase(torch.utils.data.Dataset):
    def __init__(self, pair_file = "pair_files/test.txt"):
        super().__init__()

        # Some defaults for consistency.
        self.name = "ScanNet-pairs"
        self.root = "data/scannet_test_1500"
        self.split = "test"
        self.num_views = 2

        self.rgb_transform = transforms.Compose(
            [
                transforms.Resize((480, 640)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        # parse files for data
        self.instances = self.get_instances(pair_file)

        # Print out dataset stats
        print(f"{self.name} | {len(self.instances)} pairs")

    def get_instances(self, pair_file):
        instances = []

        with open(pair_file, "r") as f:
            data = f.readlines()
            for line in data:
                image1, image2 = line.split()
                instances.append((image1, image2))

        return instances

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        rgb_path_0, rgb_path_1 = self.instances[index]

        # paths
        # rgb_path_0 = os.path.join(self.root, image1)
        # rgb_path_1 = os.path.join(self.root, image2)

        # get rgb
        rgb_0 = read_image(rgb_path_0, exif_transpose=False)
        rgb_1 = read_image(rgb_path_1, exif_transpose=False)
        rgb_0 = self.rgb_transform(rgb_0)
        rgb_1 = self.rgb_transform(rgb_1)

        return {
            "uid": index,
            "rgb_0": rgb_0,
            "rgb_1": rgb_1,
            "rgb_path_0": rgb_path_0,
            "rgb_path_1": rgb_path_1,
        }
        