import torch
import torch.nn.functional as F
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import torch.distributed as dist

from ...datasets.nyuv2_utils import resize
from ..utils import center_padding

class BaseDepther(torch.nn.Module, metaclass=ABCMeta):
    """Base class for depther."""
    def __init__(self, init_cfg=None):
        super(BaseDepther, self).__init__()
        self.fp16_enabled = False
        self.patch_size = 14

    @property
    def with_neck(self):
        """bool: whether the depther has neck"""
        return hasattr(self, "neck") and self.neck is not None

    @property
    def with_auxiliary_head(self):
        """bool: whether the depther has auxiliary head"""
        return hasattr(self, "auxiliary_head") and self.auxiliary_head is not None

    @property
    def with_decode_head(self):
        """bool: whether the depther has decode head"""
        return hasattr(self, "decode_head") and self.decode_head is not None

    @abstractmethod
    def extract_feat(self, imgs):
        """Placeholder for extract features from images."""
        pass

    @abstractmethod
    def encode_decode(self, img, img_metas):
        """Placeholder for encode images with backbone and decode into a
        semantic depth map of the same size as input."""
        pass

    @abstractmethod
    def forward_train(self, imgs, img_metas, **kwargs):
        """Placeholder for Forward function for training."""
        pass

    @abstractmethod
    def simple_test(self, img, img_meta, **kwargs):
        """Placeholder for single image test."""
        pass

    @abstractmethod
    def aug_test(self, imgs, img_metas, **kwargs):
        """Placeholder for augmentation test."""
        pass

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, "imgs"), (img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError(f"{name} must be a list, but got " f"{type(var)}")
        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(
                f"num of augmentations ({len(imgs)}) != "
                f"num of image meta ({len(img_metas)})"
            )
        if not isinstance(img_metas[0], list):
            # is a datacontainer, need to unzip
            img_metas = [[img_meta] for img_meta in img_metas]

        # all images in the same aug batch all of the same ori_shape and pad
        # shape
        for img_meta in img_metas:
            ori_shapes = [_["ori_shape"] for _ in img_meta]
            assert all(shape == ori_shapes[0] for shape in ori_shapes)
            img_shapes = [_["img_shape"] for _ in img_meta]
            assert all(shape == img_shapes[0] for shape in img_shapes)
            pad_shapes = [_["pad_shape"] for _ in img_meta]
            assert all(shape == pad_shapes[0] for shape in pad_shapes)

        # if num_augs == 1:
        #     return self.simple_test(imgs[0], img_metas[0], **kwargs)
        # else:
        return self.aug_test(imgs, img_metas, **kwargs)

    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if not isinstance(img, list):
            img = center_padding(img, self.patch_size)
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        losses = self(**data_batch)

        # split losses and images
        real_losses = {}
        log_imgs = {}
        for k, v in losses.items():
            if "img" in k:
                log_imgs[k] = v
            else:
                real_losses[k] = v

        loss, log_vars = self._parse_losses(real_losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data_batch["img_metas"]),
            log_imgs=log_imgs,
        )

        return outputs

    def val_step(self, data_batch, **kwargs):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        output = self(**data_batch, **kwargs)
        return output

    @staticmethod
    def _parse_losses(losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(f"{loss_name} is not a tensor or list of tensors")

        loss = sum(_value for _key, _value in log_vars.items() if "loss" in _key)

        log_vars["loss"] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars


def add_prefix(inputs, prefix):
    """Add prefix for dict.

    Args:
        inputs (dict): The input dict with str keys.
        prefix (str): The prefix to add.

    Returns:

        dict: The dict with keys updated with ``prefix``.
    """

    outputs = dict()
    for name, value in inputs.items():
        outputs[f"{prefix}.{name}"] = value

    return outputs

class DepthEncoderDecoder(BaseDepther):
    """Encoder Decoder depther.

    EncoderDecoder typically consists of backbone, (neck) and decode_head.
    """
    def __init__(
        self,
        backbone,
        decode_head,
        train_cfg=None,
        test_cfg=None,
    ):
        super(DepthEncoderDecoder, self).__init__()
        self.backbone = backbone
        self._init_decode_head(decode_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        assert self.with_decode_head

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = decode_head
        self.align_corners = self.decode_head.align_corners

    def extract_feat(self, img):
        """Extract features from images."""
        # x = self.backbone.extract_depth_feat(img)
        with torch.no_grad():
            x = self.backbone(img)
            if x.dim() == 4:
                out_feat = x
            elif x.dim() == 3:
                # (B, N, C) → (B, C, H, W)，需要知道 grid_size
                B, N, C = x.shape
                H = W = int(N ** 0.5)
                out_feat = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return [out_feat]

    def encode_decode(self, img, img_metas, rescale=True, size=None):
        """Encode images with backbone and decode into a depth estimation
        map of the same size as input."""
        x = self.extract_feat(img)
        out = self._decode_head_forward_test(x, img_metas)
        # crop the pred depth to the certain range.
        out = torch.clamp(
            out, min=self.decode_head.min_depth, max=self.decode_head.max_depth
        )
        if rescale:
            if size is None:
                if img_metas is not None:
                    size = img_metas[0]["ori_shape"][:2]
                else:
                    size = img.shape[2:]
            out = resize(
                input=out, size=size, mode="bilinear", align_corners=self.align_corners
            )
        return out

    def _decode_head_forward_train(self, img, x, img_metas, depth_gt, **kwargs):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(
            img, x, img_metas, depth_gt, self.train_cfg, **kwargs
        )
        losses.update(add_prefix(loss_decode, "decode"))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        depth_pred = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return depth_pred

    def forward_dummy(self, img):
        """Dummy forward function."""
        depth = self.encode_decode(img, None)

        return depth

    def forward_train(self, img, img_metas, depth_gt, **kwargs):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `depth/datasets/pipelines/formatting.py:Collect`.
            depth_gt (Tensor): Depth gt
                used if the architecture supports depth estimation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.extract_feat(img)

        losses = dict()
        # the last of x saves the info from neck
        loss_decode = self._decode_head_forward_train(
            img, x, img_metas, depth_gt, **kwargs
        )

        losses.update(loss_decode)

        return losses

    def whole_inference(self, img, img_meta, rescale, size=None):
        """Inference with full image."""
        # size = (480, 640)
        depth_pred = self.encode_decode(img, img_meta, rescale, size=size)

        return depth_pred

    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg["stride"]
        h_crop, w_crop = self.test_cfg["crop_size"]
        batch_size, _, h_img, w_img = img.size()
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, 1, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                depth_pred = self.encode_decode(crop_img, img_meta, rescale)
                preds += F.pad(
                    depth_pred,
                    (
                        int(x1),
                        int(preds.shape[3] - x2),
                        int(y1),
                        int(preds.shape[2] - y2),
                    ),
                )

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(count_mat.cpu().detach().numpy()).to(
                device=img.device
            )
        preds = preds / count_mat
        return preds

    def inference(self, img, img_meta, rescale, size=None):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `depth/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output depth map.
        """

        assert self.test_cfg["mode"] in ["slide", "whole"]
        img = center_padding(img, self.patch_size)
        ori_shape = img_meta[0]["ori_shape"]
        assert all(_["ori_shape"] == ori_shape for _ in img_meta)
        if self.test_cfg["mode"] == "slide":
            depth_pred = self.slide_inference(img, img_meta, rescale)
        else:
            depth_pred = self.whole_inference(img, img_meta, rescale, size=size)
        output = depth_pred
        flip = img_meta[0]["flip"] if "flip" in img_meta[0] else False
        if flip:
            flip_direction = img_meta[0]["flip_direction"][0]
            assert flip_direction in ["horizontal", "vertical"]
            if flip_direction == "horizontal":
                output = output.flip(dims=(3,))
            elif flip_direction == "vertical":
                output = output.flip(dims=(2,))

        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        depth_pred = self.inference(img, img_meta, rescale)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            depth_pred = depth_pred.unsqueeze(0)
            return depth_pred
        depth_pred = depth_pred.cpu().numpy()
        # unravel batch dim
        depth_pred = list(depth_pred)
        return depth_pred

    def aug_test(self, imgs, img_metas, size, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented depth logit inplace
        depth_pred = self.inference(imgs[0], img_metas[0], rescale, size=size)
        for i in range(1, len(imgs)):
            cur_depth_pred = self.inference(
                imgs[i], img_metas[i], rescale, size=depth_pred.shape[-2:]
            )
            depth_pred += cur_depth_pred
        depth_pred /= len(imgs)
        depth_pred = depth_pred.cpu().numpy()
        # unravel batch dim
        depth_pred = list(depth_pred)
        return depth_pred