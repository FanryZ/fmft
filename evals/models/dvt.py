import logging
import re
import types
from typing import List, Optional, Tuple, Union

import timm
import timm.data
import torch
from timm.models.eva import Eva
from timm.models.vision_transformer import VisionTransformer
from torch import nn
from torchvision import transforms

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import resample_abs_pos_embed
from timm.models.vision_transformer import Block, Mlp
from torch import Tensor

# We have played with these models, Feel free to add more models to the list.
MODEL_LIST = [
    # DINOv1
    "vit_small_patch8_224.dino",
    "vit_small_patch16_224.dino",
    "vit_base_patch8_224.dino",
    "vit_base_patch16_224.dino",
    # DINOv2
    "vit_small_patch14_dinov2.lvd142m",
    "vit_base_patch14_dinov2.lvd142m",
    "vit_large_patch14_dinov2.lvd142m",
    "vit_giant_patch14_dinov2.lvd142m",
    # DINOv2 + register
    "vit_small_patch14_reg4_dinov2.lvd142m",
    "vit_base_patch14_reg4_dinov2.lvd142m",
    "vit_large_patch14_reg4_dinov2.lvd142m",
    "vit_giant_patch14_reg4_dinov2.lvd142m",
    # MAE
    "vit_base_patch16_224.mae",
    "vit_large_patch16_224.mae",
    "vit_huge_patch14_224.mae",
    # CLIP
    "vit_base_patch16_clip_384.laion2b_ft_in12k_in1k",
    "vit_base_patch16_clip_224.openai",
    # EVA
    "eva02_base_patch16_clip_224.merged2b",
    # DEiT-III
    "deit3_base_patch16_224.fb_in1k",
    # Auto-auged supervised ViT:
    "vit_base_patch16_384.augreg_in21k_ft_in1k",
    # commented out for simplicity. Do not use these models for now
    # it's a bit annoying to hack the intermediate layers for these models
    # however, in our informal early experiments, these models all exhibit
    # similar artifacts as the models above.
    # the artifacts in SAM are similar to the ones in MAE, while the
    # artifacts in iJEPGA are similar to the ones in DeiT.
    # SAM
    # "samvit_base_patch16.sa1b",
    # "samvit_large_patch16.sa1b",
    # "samvit_huge_patch16.sa1b",
    # ijepga
    # "vit_huge_patch14_224_ijepa.in1k",
]


class PretrainedViTWrapper(nn.Module):
    def __init__(
        self,
        model_identifier: str = "vit_base_patch14_dinov2.lvd142m",
        stride: int = 7,
        dynamic_img_size: bool = True,
        dynamic_img_pad: bool = False,
        **kwargs,
    ):
        super().__init__()
        # comment out the following line to test the models not in the list
        assert model_identifier in MODEL_LIST, f"Model type {model_identifier} not tested yet."
        self.model_identifier = model_identifier
        self.stride = stride
        self.patch_size = int(re.search(r"patch(\d+)", model_identifier).group(1))
        self.dynamic_img_size = dynamic_img_size
        self.dynamic_img_pad = dynamic_img_pad
        self.model, self.transformation = self.create_model(model_identifier, **kwargs)
        # overwrite the stride size
        if stride != self.model.patch_embed.proj.stride[0]:
            self.model.patch_embed.proj.stride = [stride, stride]

            def dynamic_feat_size(self, img_size: Tuple[int, int]) -> Tuple[int, int]:
                """Get grid (feature) size for given image size taking account of dynamic padding.
                NOTE: must be torchscript compatible so using fixed tuple indexing
                """
                return (img_size[0] - self.patch_size[0]) // self.proj.stride[0] + 1, (
                    img_size[1] - self.patch_size[1]
                ) // self.proj.stride[1] + 1

            self.model.patch_embed.dynamic_feat_size = types.MethodType(
                dynamic_feat_size, self.model.patch_embed
            )

    @property
    def n_output_dims(self) -> int:
        return self.model.pos_embed.shape[-1]

    @property
    def num_blocks(self) -> int:
        return len(self.model.blocks)

    @property
    def last_layer_index(self) -> int:
        return self.num_blocks - 1

    def create_model(
        self, model_identifier: str, **kwargs
    ) -> Tuple[Union[VisionTransformer, Eva], transforms.Compose]:
        model = timm.create_model(
            model_identifier,
            pretrained=True,
            num_classes=0,
            dynamic_img_size=self.dynamic_img_size,
            dynamic_img_pad=self.dynamic_img_pad,
            **kwargs,
        )
        # Different models have different data configurations
        # e.g., their training resolution, normalization, etc, are different
        data_config = timm.data.resolve_model_data_config(model=model)
        transforms = timm.data.create_transform(**data_config, is_training=False)
        return model, transforms

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, List[int], Tuple[int]] = 1,
        reshape: bool = True,
        return_prefix_tokens: bool = False,
        norm: bool = True,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Intermediate layer accessor inspired by DINO / DINOv2 interface.
        Args:
            x: Input tensor.
            n: Take last n blocks if int, all if None, select matching indices if sequence
            reshape: Whether to reshape the output.
        """
        return self.model.forward_intermediates(
            x,
            n,
            return_prefix_tokens=return_prefix_tokens,
            norm=norm,
            output_fmt="NCHW" if reshape else "NLC",
            intermediates_only=True,
        )

    def forward(self, x: torch.Tensor):
        return self.model(x)

# "vitb8": 768,
#             "vitb16": 768,
#             "vitb14": 768,
#             "vitb14_reg": 768,
#             "vitl14": 1024,
#             "vitg14"
model_dict = {
    # "vit"vit_small_patch14_dinov2.lvd142m",
    "vitb14": "vit_base_patch14_dinov2.lvd142m",
    "vits14": "vit_small_patch14_dinov2.lvd142m",
    "vitl14": "vit_large_patch14_dinov2.lvd142m",
    "vitg14": "vit_giant_patch14_dinov2.lvd142m",
}


class Denoiser(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_blocks: int = 1,
        output="dense",
        return_multilayer=False,
    ):
        super().__init__()
        backbone_type = model_dict[model_name]
        stride = int(re.search(r"patch(\d+)", backbone_type).group(1))
        self.stride = stride
        self.patch_size = stride
        vit = PretrainedViTWrapper(
            model_identifier=backbone_type, 
            stride=stride, 
            dynamic_img_size=True, 
            dynamic_img_pad=True
        )
        self.model_name = model_name
        self.vit = vit
        self.feat_dim = vit.n_output_dims
        if vit.stride == 14:
            noise_map_width = 37
            noise_map_height = 37
        elif vit.stride == 16:
            noise_map_width = 32
            noise_map_height = 32

        self.denoiser = Block(
            dim=self.feat_dim,
            num_heads=self.feat_dim // 64,
            mlp_ratio=4,
            qkv_bias=True,
            qk_norm=False,
            init_values=None,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            mlp_layer=Mlp,
        )
        if num_blocks > 1:
            self.denoiser = nn.Sequential(
                *[
                    Block(
                        dim=self.feat_dim,
                        num_heads=self.feat_dim // 64,
                        mlp_ratio=4,
                        qkv_bias=True,
                        qk_norm=False,
                        init_values=None,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        act_layer=nn.GELU,
                        mlp_layer=Mlp,
                    )
                    for _ in range(num_blocks)
                ]
            )

        self.pos_embed = None
        enable_pe = True
        if enable_pe:
            seq_len = noise_map_height * noise_map_width
            self.pos_embed = nn.Parameter(torch.randn(1, seq_len, self.feat_dim) * 0.02)
        if self.vit is not None:
            for param in self.vit.parameters():
                param.requires_grad = False

    def forward(
        self,
        x,
        return_dict=False,
        return_channel_first=True,
        return_class_token=False,
        norm=True,
    ):
        class_tokens = None
        if self.vit is not None:
            with torch.no_grad():
                # (B, C, H, W)
                vit_outputs = self.vit.get_intermediate_layers(
                    x,
                    n=[self.vit.last_layer_index],
                    return_prefix_tokens=return_class_token,
                    norm=norm,
                )
                if return_class_token:
                    vit_outputs = vit_outputs[-1]
                    class_tokens = vit_outputs[1][:, 0]
                original_feats = vit_outputs[0].permute(0, 2, 3, 1)
                x = original_feats
        else:
            original_feats = x.clone()
        b, h, w, c = x.shape
        x = x.reshape(b, h * w, c)
        if self.pos_embed is not None:
            x = x + resample_abs_pos_embed(self.pos_embed, (h, w), num_prefix_tokens=0)
        x = self.denoiser(x)
        x = x.reshape(b, h, w, c)
        if return_channel_first:
            x = x.permute(0, 3, 1, 2)
        if return_dict:
            return {
                "denoised_feats": x,
                "original_feats": original_feats.detach(),
                "class_tokens": class_tokens.detach() if class_tokens is not None else None,
            }
        if return_class_token:
            assert class_tokens is not None
            return x, class_tokens
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vit = PretrainedViTWrapper(model_identifier="vit_base_patch14_dinov2.lvd142m", stride=14)
    pos_h = 37
    denoised_vit = Denoiser(
        noise_map_height=pos_h,
        noise_map_width=pos_h,
        feat_dim=vit.n_output_dims,
        vit=vit,
        num_blocks=1,
    )
    denoised_vit_ckpt = torch.load("/data/fanry/Desktop/fmft/Denoising-ViT/DVT/voc_denoised/vit_base_patch14_dinov2.lvd142m.pth")["denoiser"]
    msg = denoised_vit.load_state_dict(denoised_vit_ckpt, strict=False)
    missing_keys = {k for k in msg.missing_keys if "vit.model" not in k}
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {msg.unexpected_keys}")
    denoised_vit.to(device)
    backbone_model = denoised_vit

    image_input = torch.randn(1, 3, 224, 224).to(device)
    output = denoised_vit(image_input)
