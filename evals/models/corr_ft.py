import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.nn.parameter import Parameter
from torchvision.transforms import functional


class _LoRA_qkv(nn.Module):
    """
    In Dinov2 it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim:] += new_v
        return qkv


class FinetuneDINO(nn.Module):
    def __init__(self, r, backbone_size, reg=False, datasets=None):
        super().__init__()
        assert r > 0
        self.backbone_size = backbone_size
        self.backbone_archs = {
            "small": "vits14",
            "base": "vitb14",
            "large": "vitl14",
            "giant": "vitg14",
        }
        self.embedding_dims = {
            "small": 384,
            "base": 768,
            "large": 1024,
            "giant": 1536,
        }
        self.backbone_arch = self.backbone_archs[self.backbone_size]
        if reg:
            self.backbone_arch = f"{self.backbone_arch}_reg"
        self.embedding_dim = self.embedding_dims[self.backbone_size]
        
        self.backbone_name = f"dinov2_{self.backbone_arch}"
        dinov2 = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=self.backbone_name)
        self.datasets = datasets
        
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []
        # freeze first
        for param in dinov2.parameters():
            param.requires_grad = False

        # finetune the last 4 blocks
        for _, blk in enumerate(dinov2.blocks[-4:]):
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )
        self.reset_parameters()

        self.dinov2 = dinov2
        self.downsample_factor = 8

        self.refine_conv = nn.Conv2d(self.embedding_dim, self.embedding_dim, kernel_size=3, stride=1, padding=1)

        self.thresh3d_pos = 5e-3
        self.thres3d_neg = 0.1
        
        self.patch_size = 14
        self.target_res = 640
        
        self.input_transform = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        
    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        pass

    def load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # print(checkpoint.keys())

        self.refine_conv.load_state_dict(checkpoint['state_dict']['refine_conv'])
        for i, w_A_linear in enumerate(self.w_As):
            saved_key = f"w_a_{i:03d}"
            saved_tensor = checkpoint[saved_key]
            w_A_linear.weight = Parameter(saved_tensor)

        for i, w_B_linear in enumerate(self.w_Bs):
            saved_key = f"w_b_{i:03d}"
            saved_tensor = checkpoint[saved_key]
            w_B_linear.weight = Parameter(saved_tensor)
        self.loaded = True
                
    def get_feature_wo_kp(self, rgbs, normalize=True):
        # tgt_size = (int(rgbs.shape[-2] * self.target_res / rgbs.shape[-1]), self.target_res)
        # if rgbs.shape[-2] > rgbs.shape[-1]:
        #     tgt_size = (self.target_res, int(rgbs.shape[-1] * self.target_res / rgbs.shape[-2]))
        
        # patch_h, patch_w = tgt_size[0] // self.downsample_factor, tgt_size[1] // self.downsample_factor
        n, c, h, w = rgbs.shape
        patch_h, patch_w = h // self.patch_size, w // self.patch_size
        rgb_resized = functional.resize(rgbs, (patch_h * self.patch_size, patch_w * self.patch_size))
        
        result = self.dinov2.forward_features(rgb_resized)
        feature = result['x_norm_patchtokens'].reshape(n, patch_h, patch_w, -1).permute(0, 3, 1, 2)
        feature = self.refine_conv(feature)
        # feature = functional.resize(feature, (rgbs.shape[-2], rgbs.shape[-1])).permute(0, 2, 3, 1)
        if normalize:
            feature = F.normalize(feature, p=2, dim=1)
        return feature
    # def get_feature(self, rgbs, normalize=True):
    #     tgt_size = (int(rgbs.shape[-2] * self.target_res / rgbs.shape[-1]), self.target_res)
    #     if rgbs.shape[-2] > rgbs.shape[-1]:
    #         tgt_size = (self.target_res, int(rgbs.shape[-1] * self.target_res / rgbs.shape[-2]))
        
    #     patch_h, patch_w = tgt_size[0] // self.downsample_factor, tgt_size[1] // self.downsample_factor
    #     rgb_resized = functional.resize(rgbs, (patch_h * self.patch_size, patch_w * self.patch_size))
        
    #     result = self.dinov2.forward_features(self.input_transform(rgb_resized))        
    #     feature = result['x_norm_patchtokens'].reshape(rgb_resized.shape[0], patch_h, patch_w, -1).permute(0, 3, 1, 2)
    #     feature = self.refine_conv(feature)
    #     if normalize:
    #         feature = F.normalize(feature, p=2, dim=1)
    #     return feature


class CorrFT(nn.Module):

    ckpt_dict = {
        "giant": "https://huggingface.co/qq456cvb/3DCorrEnhance/resolve/main/dinov2_giant.ckpt",
        "large": "https://huggingface.co/qq456cvb/3DCorrEnhance/resolve/main/dinov2_large.ckpt",
        "base": "https://huggingface.co/qq456cvb/3DCorrEnhance/resolve/main/dinov2_base.ckpt",
        "small": "https://huggingface.co/qq456cvb/3DCorrEnhance/resolve/main/dinov2_small.ckpt",
        "reg_giant": "https://huggingface.co/qq456cvb/3DCorrEnhance/resolve/main/dinov2_reg_giant.ckpt",
        "reg_large": "https://huggingface.co/qq456cvb/3DCorrEnhance/resolve/main/dinov2_reg_large.ckpt",
        "reg_base": "https://huggingface.co/qq456cvb/3DCorrEnhance/resolve/main/dinov2_reg_base.ckpt",
        "reg_small": "https://huggingface.co/qq456cvb/3DCorrEnhance/resolve/main/dinov2_reg_small.ckpt",
        "clip": "https://huggingface.co/qq456cvb/3DCorrEnhance/resolve/main/clip.ckpt",
        "mae": "https://huggingface.co/qq456cvb/3DCorrEnhance/resolve/main/mae.ckpt",
        "deit": "https://huggingface.co/qq456cvb/3DCorrEnhance/resolve/main/deit.ckpt",
    }

    def __init__(self, 
                 backbone_size="base", 
                 rank=4, 
                 reg=False, 
                 output=None,
                 return_multilayer=False
                 ):
        super().__init__()
        assert backbone_size in ["small", "base", "large", "giant"]
        if reg:
            backbone_size = f"reg_{backbone_size}"
        self.backbone_size = backbone_size
        ckpt_path = self.ckpt_dict[backbone_size]

        ckpt = torch.hub.load_state_dict_from_url(ckpt_path)
        self.finetuned_dino = FinetuneDINO(rank, backbone_size, reg, None)
        self.finetuned_dino.load_checkpoint(ckpt)
        self.patch_size = self.finetuned_dino.patch_size

    def forward(self, rgbs, normalize=False):
        output = self.finetuned_dino.get_feature_wo_kp(rgbs, normalize)
        # return output.permute(0, 3, 1, 2)
        return output
    

# Test_model
if __name__ == "__main__":
    model = CorrFT(backbone_size="base", rank=4, reg=False, output=None, return_multilayer=False)
    model = model.to("cuda")
    model.eval()
    
    rgbs = torch.randn(1, 3, 640, 640).to("cuda")
    feats = model(rgbs)
    print(feats.shape)
    