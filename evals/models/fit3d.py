import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import types
from torch import Tensor
from .utils import center_padding, tokens_to_output


finetuned_checkpoints = {
    "dinov2_small": "https://huggingface.co/yuanwenyue/FiT3D/resolve/main/dinov2_small_finetuned.pth",
    "dinov2_base": "https://huggingface.co/yuanwenyue/FiT3D/resolve/main/dinov2_base_finetuned.pth",
    "dinov2_reg_small": "https://huggingface.co/yuanwenyue/FiT3D/resolve/main/dinov2_reg_small_finetuned.pth",
    "clip_base": "https://huggingface.co/yuanwenyue/FiT3D/resolve/main/clip_base_finetuned.pth",
    "mae_base": "https://huggingface.co/yuanwenyue/FiT3D/resolve/main/mae_base_finetuned.pth",
    "deit3_base": "https://huggingface.co/yuanwenyue/FiT3D/resolve/main/deit3_base_finetuned.pth"
}


def get_intermediate_layers(
    self,
    x: torch.Tensor,
    n=1,
    reshape: bool = False,
    return_prefix_tokens: bool = False,
    return_class_token: bool = False,
    norm: bool = True,
):
    outputs = self._intermediate_layers(x, n)
    if norm:
        outputs = [self.norm(out) for out in outputs]
    if return_class_token:
        prefix_tokens = [out[:, 0] for out in outputs]
    else:
        prefix_tokens = [out[:, 0 : self.num_prefix_tokens] for out in outputs]
    outputs = [out[:, self.num_prefix_tokens :] for out in outputs]

    if reshape:
        B, C, H, W = x.shape
        grid_size = (
            (H - self.patch_embed.patch_size[0])
            // self.patch_embed.proj.stride[0]
            + 1,
            (W - self.patch_embed.patch_size[1])
            // self.patch_embed.proj.stride[1]
            + 1,
        )
        outputs = [
            out.reshape(x.shape[0], grid_size[0], grid_size[1], -1)
            .permute(0, 3, 1, 2)
            .contiguous()
            for out in outputs
        ]

    if return_prefix_tokens or return_class_token:
        return tuple(zip(outputs, prefix_tokens))
    return tuple(outputs)


def build_2d_model(model_name="dinov2_small"):
    timm_model_card = {
        "dinov2_small": "vit_small_patch14_dinov2.lvd142m",
        "dinov2_base": "vit_base_patch14_dinov2.lvd142m",
        "dinov2_reg_small": "vit_small_patch14_reg4_dinov2.lvd142m",
        "clip_base": "vit_base_patch16_clip_384.laion2b_ft_in12k_in1k",
        "mae_base": "vit_base_patch16_224.mae",
        "deit3_base": "deit3_base_patch16_224.fb_in1k"
    }
    assert model_name in timm_model_card.keys(), "invalid model name"
    model = timm.create_model(
        timm_model_card[model_name],
        pretrained=True,
        num_classes=0,
        dynamic_img_size=True,
        dynamic_img_pad=False,
    )
    # model.get_intermediate_layers = types.MethodType(
    #     get_intermediate_layers,
    #     model,
    # )
    return model


class FiT3D(nn.Module):
    def __init__(
        self,
        fit_type="dinov2_base",
        output="dense-cls",
        return_multilayer=False,
    ):
        super().__init__()

        self.vit = build_2d_model(fit_type)
        self.finetuned_model = build_2d_model(fit_type)
        fine_ckpt = torch.hub.load_state_dict_from_url(finetuned_checkpoints[fit_type], map_location='cpu')
        msg = self.finetuned_model.load_state_dict(fine_ckpt, strict=False)
        print(msg)

        self.model_name = f"fit3d_{fit_type}"
        self.patch_size = self.vit.patch_embed.proj.kernel_size[0]

        # self.fuse_layer = nn.Linear(self.vit.embed_dim*2, self.vit.embed_dim)

    def forward(
        self,
        x: Tensor,
        norm=True,
    ) -> Tensor:
        # run backbone if backbone is there
        x = center_padding(x, self.patch_size)
        if self.vit is not None:
            with torch.no_grad():
                vit_outputs = self.vit.get_intermediate_layers(
                    x,
                    n=[len(self.vit.blocks) - 1],
                    reshape=True,
                    norm=norm,
                )[0]
                vit_outputs_fine = self.finetuned_model.get_intermediate_layers(
                    x,
                    n=[len(self.finetuned_model.blocks) - 1],
                    reshape=True,
                    norm=norm,
                )[0]

                ## strategy 1: concatenate
                # x = torch.cat([raw_vit_feats, raw_vit_feats_fine], -1)
                ## strategy 2: adding
                output = vit_outputs + vit_outputs_fine
                ## strategy 3: linear fusion
                # x = self.fuse_layer(x)
                
        # out_feat = x
        # out_feat = out_feat.permute(0, 3, 1, 2)
        return output
