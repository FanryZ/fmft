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

from evals.datasets.utils import (
    read_image,
)


class RealWorldPair(torch.utils.data.Dataset):
    def __init__(
        self,
    ):
        super().__init__()
        self.pair_root = Path("/data/fanry/Desktop/fmft/realworld/selfDataPair")
        self.object_id = glob.glob(str(self.pair_root / "*"))
        self.object_id = [os.path.basename(_id) for _id in self.object_id]

        self.pairs = []
        for obj_id in self.object_id:
            obj_dir = self.pair_root / obj_id
            angles = glob.glob(str(obj_dir / "*"))
            angles = [os.path.basename(_id) for _id in angles]
            for angle in angles:
                angle_dir = obj_dir / angle
                images = glob.glob(str(angle_dir / "*.png"))
                image_ids = [os.path.basename(_id)[:-4] for _id in images]
                self.pairs.append({
                    "obj_id": obj_id,
                    "angle": angle,
                    "images": image_ids
                })

        # get transforms
        self.transform = transforms.Compose(
            [
                transforms.Resize((360, 640)),
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
        image_path_0 = self.pair_root / pair["obj_id"] / pair["angle"] / f"{img_id_0}.png"
        image_path_1 = self.pair_root / pair["obj_id"] / pair["angle"] / f"{img_id_1}.png"

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

import hydra
import numpy as np
import torch
import torch.nn.functional as nn_F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from evals.utils.vggt_utils import patch_track, track_point_filter, vis_track_points
from metric_learning import metric_learning_finetune_outlier, interpolate_features, outliers, conv_training, viz_feat


@hydra.main("../configs", "scannet_finetune", None)
def main(cfg: DictConfig):
    print(f"Config: \n {OmegaConf.to_yaml(cfg)}")

    # ===== Get model and dataset ====
    model = instantiate(cfg.backbone, output="dense", return_multilayer=cfg.multilayer)
    model = model.to("cuda")
    dataset = RealWorldPair()
    print(dataset.pairs)

    train_range = range(len(dataset)) if cfg.pair_num <= 0 else range(min(cfg.pair_num, len(dataset)))
    if cfg.select_idx >= 0:
        train_range = [cfg.select_idx]
    for i in tqdm(train_range):
        instance = dataset.__getitem__(i)
        rgbs = torch.stack((instance["rgb_0"], instance["rgb_1"]), dim=0)
        rgb_path_0 = instance["rgb_path_0"]
        rgb_path_1 = instance["rgb_path_1"]

        feats = model(rgbs.cuda())
        feats_0, feats_1 = torch.split(feats, [1, 1], dim=0)
        patch_size = model.patch_size
        
        images, track_list, vis_score, conf_score = patch_track([rgb_path_0, rgb_path_1], patch_size)
        points_masked0_1, points_masked1_1 = track_point_filter(track_list, vis_score, conf_score, 0.5)
        images, track_list, vis_score, conf_score = patch_track([rgb_path_1, rgb_path_0], patch_size)
        points_masked1_2, points_masked0_2 = track_point_filter(track_list, vis_score, conf_score, 0.5)
        points_masked0 = torch.cat((points_masked0_1, points_masked0_2), dim=0)
        points_masked1 = torch.cat((points_masked1_1, points_masked1_2), dim=0)

        # vis_track_points(images, points_masked1, points_masked0)
        # _, n, c, vh, vw = images.shape
        # if cfg.normalize:
        #     feats_0 = nn_F.normalize(feats_0, dim=2)
        #     feats_1 = nn_F.normalize(feats_1, dim=2)

        outlier_points_0 = outliers(points_masked0, images.shape[-2:], patch_size, margin=2)
        outlier_points_1 = outliers(points_masked1, images.shape[-2:], patch_size, margin=2)
        conv_layer = conv_training(feats_0, feats_1, points_masked0[None, ...], points_masked1[None, ...], 
            outlier_points_0[None, ...], outlier_points_1[None, ...], images.shape[-2:], 
            cfg.neg_pair_num, cfg.iter_num, cfg.temperature, cfg.weight_decay)

        feats_0 = conv_layer(feats_0)
        feats_1 = conv_layer(feats_1)
        
        interpolated_feats_0 = interpolate_features(feats_0, points_masked0[None, ...], images.shape[-2:])
        interpolated_feats_1 = interpolate_features(feats_1, points_masked1[None, ...], images.shape[-2:])
        outlier_feats0 = interpolate_features(feats_0, outlier_points_0[None, ...], images.shape[-2:])
        outlier_feats1 = interpolate_features(feats_1, outlier_points_1[None, ...], images.shape[-2:])

        np_feats_0 = interpolated_feats_0[0].detach().cpu().numpy()
        np_feats_1 = interpolated_feats_1[0].detach().cpu().numpy()
        np_outlier_feats0 = outlier_feats0[0].detach().cpu().numpy()
        np_outlier_feats1 = outlier_feats1[0].detach().cpu().numpy()

        metric_model, metric_model_score = metric_learning_finetune_outlier(
            np_feats_0, np_feats_1, np_outlier_feats0, np_outlier_feats1, cfg.neg_pair_num)
        transform_mat = torch.from_numpy(metric_model.components_.T).to(feats_0)
        # feats_0 = (feats_0.permute(0, 2, 3, 1) @ transform_mat).permute(0, 3, 1, 2)
        # feats_1 = (feats_1.permute(0, 2, 3, 1) @ transform_mat).permute(0, 3, 1, 2)

        # conv_feats_0 = conv_layer(feats_0)
        # conv_feats_1 = conv_layer(feats_1)   
        # if cfg.multilayer:
        #     feats = torch.cat(feats, dim=1)
        
        output_path = f"./exp/realworld_finetune"
        os.makedirs(output_path, exist_ok=True)
        if cfg.select_idx >= 0:
            torch.save(conv_layer.state_dict(), os.path.join(output_path, f"conv_layer_select.pth"))
            torch.save(transform_mat, os.path.join(output_path, f"proj_layer_select.pth"))
        else:
            torch.save(conv_layer.state_dict(), os.path.join(output_path, f"conv_layer_{i}.pth"))
            torch.save(transform_mat, os.path.join(output_path, f"proj_layer_{i}.pth"))

        # feats_0 = (feats_0.permute(0, 2, 3, 1) @ transform_mat).permute(0, 3, 1, 2)
        # feats_1 = (feats_1.permute(0, 2, 3, 1) @ transform_mat).permute(0, 3, 1, 2)
        # feats_0 = nn_F.normalize(nn_F.interpolate(feats_0, scale_factor=4), dim=1)
        # feats_1 = nn_F.normalize(nn_F.interpolate(feats_1, scale_factor=4), dim=1)

        # viz_feat(feats_0.detach(), os.path.join(output_path, f"feats_0_{i}.png"))
        # viz_feat(feats_1.detach(), os.path.join(output_path, f"feats_1_{i}.png")) 
        # viz_feat(torch.cat((feats_0.detach(), feats_1.detach()), dim=3), os.path.join(output_path, f"feats.png")) 
        # viz_feat(interpolated_feats_0, os.path.join(output_path, f"interpolated_feats_0_{i}.png"))
        # viz_feat(interpolated_feats_1, os.path.join(output_path, f"interpolated_feats_1_{i}.png"))
        # viz_feat(outlier_feats0, os.path.join(output_path, f"outlier_feats0_{i}.png"))
        # viz_feat(outlier_feats1, os.path.join(output_path, f"outlier_feats1_{i}.png"))

    print(f"Features saved to {output_path}")


if __name__ == "__main__":
    main()
