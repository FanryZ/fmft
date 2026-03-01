import torch
import numpy as np

from .deit_utils import deit_base_patch16_LS, deit_large_patch16_LS
from .utils import resize_pos_embed, tokens_to_output


class DeIT(torch.nn.Module):
    def __init__(
        self,
        model_size="base",
        img_size=384,
        patch_size=16,
        output="dense",
        layer=-1,
        return_multilayer=False,
        conv_layer=None,
        mode="finetune",
    ):
        super().__init__()

        assert output in ["cls", "gap", "dense"], "Options: [cls, gap, dense]"
        self.output = output

        self.checkpoint_name = f"deit3_{model_size}-{patch_size}_{img_size}"
        if model_size == "base":
            vit = deit_base_patch16_LS(True, img_size, True)
        elif model_size == "large":
            vit = deit_large_patch16_LS(True, img_size, True)

        self.vit = vit.eval()
        self.patch_size = patch_size
        self.embed_size = (img_size / self.patch_size, img_size, self.patch_size)
        # deactivate strict image size for positional embeding resizing
        self.vit.patch_embed.strict_img_size = False

        num_layers = len(self.vit.blocks)
        feat_dim = self.vit.num_features

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

        assert mode in ["finetune", "add", "norm_add", "conv"]
        self.mode = mode
        # define layer name (for logging)
        self.layer = "-".join(str(_x) for _x in self.multilayers)
        self.refine_conv = torch.nn.Conv2d(feat_dim, feat_dim, kernel_size=3, stride=1, padding=1, bias=False)
        if conv_layer is not None:
            self.refine_conv.load_state_dict(torch.load(conv_layer))
            self.refine_proj = torch.nn.Parameter(torch.load(conv_layer.replace("conv", "proj")))


    def forward(self, images):
        B, _, h, w = images.shape
        h, w = h // self.patch_size, w // self.patch_size

        if (h, w) != self.embed_size:
            self.embed_size = (h, w)
            self.vit.pos_embed.data = resize_pos_embed(
                self.vit.pos_embed[0], self.embed_size, False
            )[None, :, :]

        x = self.vit.patch_embed(images)
        cls_tokens = self.vit.cls_token.expand(B, -1, -1)
        x = x + self.vit.pos_embed
        x = torch.cat((cls_tokens, x), dim=1)

        embeds = []
        for i, blk in enumerate(self.vit.blocks):
            x = blk(x)
            if i in self.multilayers:
                embeds.append(x)
                if len(embeds) == len(self.multilayers):
                    break

        outputs = []
        for i, x_i in enumerate(embeds):
            x_i = tokens_to_output(self.output, x_i[:, 1:], x_i[:, 0], (h, w))
            outputs.append(x_i)

        feat_ori = outputs[0]
        feat_ft = self.refine_conv(feat_ori)
        if self.mode == "norm_add":
            feat_ft = F.normalize(feat_ft, dim=1) + F.normalize(feat_ori, dim=1)
        elif self.mode == "add":
            feat_ft = feat_ft + feat_ori
        elif self.mode == "finetune":
            feat_ft = (feat_ft.permute(0, 2, 3, 1) @ self.refine_proj).permute(0, 3, 1, 2)
        return feat_ft
        # return outputs[0] if len(outputs) == 1 else outputs
