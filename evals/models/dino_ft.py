import torch
import torch.nn.functional as F
import numpy as np
import pickle

from .utils import center_padding, tokens_to_output
meta_list = pickle.load(open("meta_navi.pkl", "rb"))
# meta_list = pickle.load(open("meta_onepose.pkl", "rb"))

class DINO(torch.nn.Module):
    def __init__(
        self,
        dino_name="dino",
        model_name="vitb16",
        output="dense",
        layer=-1,
        return_multilayer=False,
        cache_feat=None,

        transform_mat=None,
        lambda_ft=0.5,
        adapt=False,
        mat_idx=0
    ):
        super().__init__()
        feat_dims = {
            "vitb8": 768,
            "vitb16": 768,
            "vitb14": 768,
            "vitb14_reg": 768,
            "vitl14": 1024,
            "vitg14": 1536,
        }

        # get model
        self.model_name = dino_name
        self.checkpoint_name = f"{dino_name}_{model_name}"
        dino_vit = torch.hub.load(f"facebookresearch/{dino_name}", self.checkpoint_name)
        self.vit = dino_vit.eval().to(torch.float32)
        self.has_registers = "_reg" in model_name

        assert output in ["cls", "gap", "dense", "dense-cls"]
        self.output = output
        self.patch_size = self.vit.patch_embed.proj.kernel_size[0]

        feat_dim = feat_dims[model_name]
        feat_dim = feat_dim * 2 if output == "dense-cls" else feat_dim

        num_layers = len(self.vit.blocks)
        multilayers = [
            num_layers // 4 - 1,
            num_layers // 2 - 1,
            num_layers // 4 * 3 - 1,
            num_layers - 1,
        ]

        if return_multilayer:
            self.feat_dim = [feat_dim, feat_dim, feat_dim, feat_dim]
            self.multilayers = multilayers
        else:
            self.feat_dim = feat_dim
            layer = multilayers[-1] if layer == -1 else layer
            self.multilayers = [layer]

        # define layer name (for logging)
        self.layer = "-".join(str(_x) for _x in self.multilayers)

        transform_mat_data = np.load(transform_mat)
        meta_data = meta_list[mat_idx]
        trans_id = f"{meta_data['obj_id']}/{meta_data['angle']}/{meta_data['images'][0]}_{meta_data['images'][1]}"
        transform_mat = torch.tensor(transform_mat_data[trans_id].T)
        # transform_mat = torch.tensor(transform_mat["transform_mats"][mat_idx])
        self.transform_mat = torch.nn.Parameter(transform_mat.to(torch.float32))
        self.refine_conv = torch.nn.Conv2d(feat_dim, feat_dim, kernel_size=3, stride=1, padding=1)

        feats_org = torch.tensor(transform_mat_data["feats_org"][mat_idx])
        feat0, feat1 = feats_org[0], feats_org[1]
        feat0m = torch.mean(feat0.reshape(feat0.shape[0], -1), dim=1)
        feat1m = torch.mean(feat1.reshape(feat1.shape[0], -1), dim=1)
        # self.feat_m = torch.nn.Parameter((feat0m + feat1m) / 2)
        # self.feat_m = torch.nn.Parameter((feat0m + feat1m) / 2)
        self.feat0m = torch.nn.Parameter(feat0m)
        self.feat1m = torch.nn.Parameter(feat1m)
        self.paired_dist = torch.linalg.norm(feat0m - feat1m)
        # self.transform_mat = torch.nn.Parameter(torch.tensor(np.load("./temp/test.npy").T).to(torch.float32))
        self.lambda_ft = lambda_ft
        self.adapt = adapt

    def transform_softmax(self, input_featmap, temp=1.0):
        cost_volume = input_featmap @ self.src_feats.permute(1, 0)
        cost_volume = cost_volume * temp
        cost_volume = F.softmax(cost_volume, dim=3)
        return cost_volume @ self.tar_feats

    def forward(self, images):
        # pad images (if needed) to ensure it matches patch_size
        images = center_padding(images, self.patch_size)
        h, w = images.shape[-2:]
        h, w = h // self.patch_size, w // self.patch_size

        if self.model_name == "dinov2":
            x = self.vit.prepare_tokens_with_masks(images, None)
        else:
            x = self.vit.prepare_tokens(images)

        embeds = []
        for i, blk in enumerate(self.vit.blocks):
            x = blk(x)
            if i in self.multilayers:
                embeds.append(x)
                if len(embeds) == len(self.multilayers):
                    break

        num_spatial = h * w
        outputs = []
        for i, x_i in enumerate(embeds):
            cls_tok = x_i[:, 0]
            # ignoring register tokens
            spatial = x_i[:, -1 * num_spatial :]
            x_i = tokens_to_output(self.output, spatial, cls_tok, (h, w))
            outputs.append(x_i)

        # feat_ori = outputs[0].permute(0, 2, 3, 1)
        feat_ft = self.refine_conv(outputs[0])
        return feat_ft
        # feat_ft = feat_ori @ self.transform_mat
        # alpha = 0.0

        if self.adapt:
            # beta = feat_ft * feat_ori_norm_m / feat_ft_norm_m
            feat_ft_norm_m = torch.mean(feat_ft.norm(dim=1, keepdim=True))
            feat_ori_norm_m = torch.mean(feat_ori.norm(dim=1, keepdim=True))
            beta = feat_ori_norm_m / feat_ft_norm_m

            feat_m = torch.mean(feat_ori, dim=[0, 1, 2])
            feat_dist = torch.linalg.norm(feat_m - self.feat0m) + torch.linalg.norm(feat_m - self.feat1m)
            # print(feat_dist)
            # alpha = min(0.95, self.paired_dist / feat_dist) 
            alpha = 1.0
            
        return feat_ft.permute(0, 3, 1, 2)
        # return (feat_ori + self.lambda_ft * alpha * feat_ft).permute(0, 3, 1, 2)
        # return ((1 - alpha) * feat_ori + alpha * self.lambda_ft * beta * feat_ft).permute(0, 3, 1, 2)
        # return feat_ori.permute(0, 3, 1, 2)
        # feat_ft = self.transform_softmax(feat_ori, temp=100.0)

# 17.81, 27.05, 38.61, 36.97, 27.52, 19.73, 11.15 