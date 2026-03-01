import torch
import torch.nn.functional as F

from .utils import center_padding, tokens_to_output

class DINO(torch.nn.Module):
    def __init__(
        self,
        dino_name="dino",
        model_name="vitb16",
        output="dense",
        layer=-1,
        return_multilayer=False,
        conv_layer=None,
        mode="finetune",
    ):
        super().__init__()
        feat_dims = {
            "vitb8": 768,
            "vitb16": 768,
            "vitb14": 768,
            "vits14": 384,
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

        assert mode in ["finetune", "add", "norm_add", "conv"]
        self.mode = mode
        # define layer name (for logging)
        self.layer = "-".join(str(_x) for _x in self.multilayers)
        self.refine_conv = torch.nn.Conv2d(feat_dim, feat_dim, kernel_size=3, stride=1, padding=1, bias=False)
        # self.refine_proj = torch.nn.Linear(feat_dim, feat_dim, bias=False)
        if conv_layer is not None:
            self.refine_conv.load_state_dict(torch.load(conv_layer))
            self.refine_proj = torch.nn.Parameter(torch.load(conv_layer.replace("conv", "proj")))

    def parameters(self, recurse: bool = True):
        return self.refine_conv.parameters()

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

        feat_ori = outputs[0]
        feat_ft = self.refine_conv(feat_ori)
        if self.mode == "norm_add":
            feat_ft = F.normalize(feat_ft, dim=1) + F.normalize(feat_ori, dim=1)
        elif self.mode == "add":
            feat_ft = feat_ft + feat_ori
        elif self.mode == "finetune":
            feat_ft = (feat_ft.permute(0, 2, 3, 1) @ self.refine_proj).permute(0, 3, 1, 2)
        return feat_ft

"""
2025-08-26 08:42:15,344 - mmseg - INFO - Validation metrics: {'a1': 0.8117330832600085, 'a2': 0.9758387609038532, 'a3': 0.9962063602307014, 'abs_rel': 0.14518696, 'rmse': 0.48781818, 'log_10': 0.060043737, 'rmse_log': 0.17198631, 'silog': 13.198341140813666, 'sq_rel': 0.09064789}
2025-08-26 08:41:44,993 - mmseg - INFO - Validation metrics: {'a1': 0.8007200119678315, 'a2': 0.9742471841753886, 'a3': 0.9959919878598811, 'abs_rel': 0.1489955, 'rmse': 0.4995742, 'log_10': 0.06170747, 'rmse_log': 0.17588475, 'silog': 13.324513427535312, 'sq_rel': 0.09496744}
"""
