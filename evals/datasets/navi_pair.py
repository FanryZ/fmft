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

from .utils import (
    bbox_crop,
    camera_matrices_from_annotation,
    compute_normal,
    get_grid,
    get_navi_transforms,
    read_depth,
    read_image,
)


class NAVIPair(torch.utils.data.Dataset):
    def __init__(
        self,
        image_mean="imagenet",
        bbox_crop=True,
    ):
        super().__init__()
        self.bbox_crop = bbox_crop
        self.data_root = Path("data/navi_v1")
        self.pair_root = Path("/data/fanry/Desktop/fmft/locft/naviPair2")
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
                    images = glob.glob(str(pair_dir / "downsampled_*"))
                    image_ids = [os.path.basename(_id).split(".")[0].split("_")[1]
                                 for _id in images]
                    scene_id = open(str(pair_dir / "scene.txt")).read().strip()
                    self.pairs.append({
                        "obj_id": obj_id,
                        "angle": angle,
                        "pair": pair,
                        "images": image_ids,
                        "scene_id": scene_id
                    })
        # get transforms
        image_size = (512, 512)
        t_fns = get_navi_transforms(
            image_mean,
            image_size=image_size
        )
        self.image_transform, self.target_transform, self.shared_transform = t_fns

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = self.pairs[index]
        img_id_0 = pair["images"][0]
        img_id_1 = pair["images"][1]

        inst_0 = self.get_single(pair["obj_id"], pair["scene_id"], img_id_0)
        inst_1 = self.get_single(pair["obj_id"], pair["scene_id"], img_id_1)

        output = {}
        output["meta"] = pair
        for key in inst_0:
            output[f"{key}_0"] = inst_0[key]
            output[f"{key}_1"] = inst_1[key]
        return output

    def get_single(self, obj_id, scene_id, img_id):
        prefix = "downsampled_"
        scene_path = self.data_root / obj_id / scene_id
        image_path = scene_path / f"images/{prefix}{img_id}.jpg"
        depth_path = scene_path / f"depth/{prefix}{img_id}.png"
        with open(os.path.join(scene_path, "annotations.json")) as f:
            annotations = json.load(f)
            anno = annotations[0]

        # get image
        image = read_image(image_path)
        image = self.image_transform(image)

        # get depth -- move from millimeter to meters
        depth = read_depth(str(depth_path)) / 1000
        min_depth = depth[depth > 0].min()
        depth = self.target_transform(depth)

        #  === construct xyz at full image size and apply all transformations ===
        orig_h, orig_w = anno["image_size"]
        image_h, image_w = image.shape[1:]
        orig_fx = anno["camera"]["focal_length"]
        aug_fx = orig_fx * min(image_h, image_w) / min(orig_h, orig_w)

        # intrnsics for augmented image
        intrinsics = torch.eye(3)
        intrinsics[0, 0] = aug_fx
        intrinsics[1, 1] = aug_fx
        intrinsics[0, 2] = 0.5 * image_h  # assume offset is at the center
        intrinsics[1, 2] = 0.5 * image_w  # assume offset is at the center

        # make grid
        grid = get_grid(image_h, image_w)
        uvd_grid = depth * grid
        xyz = intrinsics.inverse() @ uvd_grid.view(3, image_h * image_w)
        xyz_grid = xyz.view(3, image_h, image_w)

        if self.bbox_crop:
            image, depth, xyz_grid = bbox_crop(image, depth, xyz_grid)

        bbox_h, bbox_w = image.shape[1:]
        snorm = compute_normal(depth.clone(), aug_fx)

        if self.shared_transform is not None:
            transformed = self.shared_transform(
                image=image.permute(1, 2, 0).numpy(),
                depth=depth.permute(1, 2, 0).numpy(),
                snorm=snorm.permute(1, 2, 0).numpy(),
                xyz_grid=xyz_grid.permute(1, 2, 0).numpy(),
            )

            image = torch.tensor(transformed["image"]).float().permute(2, 0, 1)
            depth = torch.tensor(transformed["depth"]).float()[None, :, :, 0]
            snorm = torch.tensor(transformed["snorm"]).float().permute(2, 0, 1)
            xyz_grid = torch.tensor(transformed["xyz_grid"]).float().permute(2, 0, 1)

        # -- use min() to handle center cropping
        final_h, final_w = image.shape[1:]
        final_fx = aug_fx * min(final_h, final_w) / min(bbox_h, bbox_w)
        intrinsics = torch.eye(3)
        intrinsics[:2] = final_fx * intrinsics[:2]

        # remove weird depth artifacts from averaging
        depth[depth < min_depth] = 0

        return {
            "rgb": image,
            "rgb_path": image_path,
            "depth": depth,
            "class_id": obj_id,
            "intrinsics": intrinsics,
            "snorm": snorm,
            "xyz_grid": xyz_grid,
        }


if __name__ == "__main__":
    # Path to the JSON file
    import json
    import cv2
    import numpy as np
    import os
    from matplotlib import pyplot as plt

    dataset = NAVIPair()
    exit()
    
    # Paths
    base_dir = '/data/fanry/Desktop/fmft/locft/naviPair/3d_dollhouse_sink/45/pair2'
    json_path = os.path.join(base_dir, 'overlap_points.json')

    # Read the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    start_idx = 0  # Starting from the first point
    end_idx = 100   # Get 100 points
    data = data[start_idx:end_idx]

    # Load images
    img1 = cv2.imread(os.path.join(base_dir, '004.jpg'))
    img2 = cv2.imread(os.path.join(base_dir, '018.jpg'))
    # Extract points
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # Draw points on images
    for item in data:
        # Draw points on image1
        x1, y1 = map(int, item['image1']['point'])
        cv2.circle(img1_rgb, (x1, y1), 10, (255, 0, 0), -1)  # Red points for image1
        
        # Draw points on image2
        x2, y2 = map(int, item['image2']['point'])
        cv2.circle(img2_rgb, (x2, y2), 10, (0, 0, 255), -1)  # Blue points for image2

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Display image1 with points
    ax1.imshow(img1_rgb)
    ax1.set_title('Image 013.jpg with Points')
    ax1.axis('off')

    # Display image2 with points
    ax2.imshow(img2_rgb)
    ax2.set_title('Image 016.jpg with Points')
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig("test.png")

    # Save the results
    # output_path1 = os.path.join(base_dir, '004_with_points.jpg')
    # output_path2 = os.path.join(base_dir, '018_with_points.jpg')
    # cv2.imwrite(output_path1, cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2BGR))
    # cv2.imwrite(output_path2, cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2BGR))
    # print(f"Images with points saved to:\n{output_path1}\n{output_path2}")

    