import os.path as osp
import numpy as np
import torch
from numpy.core.fromnumeric import shape
import warnings
import torch.nn.functional as F
from PIL import Image
import json
from collections.abc import Sequence
from .mmcv import imflip, imrescale, imresize, is_list_of, imnormalize, imrotate, imfrombytes, FileClient
import functools


def assert_tensor_type(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not isinstance(args[0].data, torch.Tensor):
            raise AttributeError(
                f'{args[0].__class__.__name__} has no attribute '
                f'{func.__name__} for type {args[0].datatype}')
        return func(*args, **kwargs)

    return wrapper

def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not isinstance(data, str):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f"type {type(data)} cannot be converted to tensor.")

class ToTensor(object):
    """Convert some results to :obj:`torch.Tensor` by given keys.

    Args:
        keys (Sequence[str]): Keys that need to be converted to Tensor.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        """Call function to convert data in results to :obj:`torch.Tensor`.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data converted
                to :obj:`torch.Tensor`.
        """

        for key in self.keys:
            results[key] = to_tensor(results[key])
        return results

    def __repr__(self):
        return self.__class__.__name__ + f"(keys={self.keys})"

class DataContainer:
    """A container for any type of objects.

    Typically tensors will be stacked in the collate function and sliced along
    some dimension in the scatter function. This behavior has some limitations.
    1. All tensors have to be the same size.
    2. Types are limited (numpy array or Tensor).

    We design `DataContainer` and `MMDataParallel` to overcome these
    limitations. The behavior can be either of the following.

    - copy to GPU, pad all tensors to the same size and stack them
    - copy to GPU without stacking
    - leave the objects as is and pass it to the model
    - pad_dims specifies the number of last few dimensions to do padding
    """

    def __init__(self,
                 data,
                 stack=False,
                 padding_value=0,
                 cpu_only=False,
                 pad_dims=2):
        self._data = data
        self._cpu_only = cpu_only
        self._stack = stack
        self._padding_value = padding_value
        assert pad_dims in [None, 1, 2, 3]
        self._pad_dims = pad_dims

    def __repr__(self):
        return f'{self.__class__.__name__}({repr(self.data)})'

    def __len__(self):
        return len(self._data)

    @property
    def data(self):
        return self._data

    @property
    def datatype(self):
        if isinstance(self.data, torch.Tensor):
            return self.data.type()
        else:
            return type(self.data)

    @property
    def cpu_only(self):
        return self._cpu_only

    @property
    def stack(self):
        return self._stack

    @property
    def padding_value(self):
        return self._padding_value

    @property
    def pad_dims(self):
        return self._pad_dims

    @assert_tensor_type
    def size(self, *args, **kwargs):
        return self.data.size(*args, **kwargs)

    @assert_tensor_type
    def dim(self):
        return self.data.dim()

class ResizeToDepth:
    def __init__(self, size=(480, 640)):
        self.size = size

    def __call__(self, results):
        for key in ["img", "depth_gt"]:
            if key in results and isinstance(results[key], torch.Tensor):
                if results[key].ndim == 3:  # [C,H,W]
                    results[key] = results[key].unsqueeze(0)
                    results[key] = resize(results[key], size=self.size, mode="bilinear" if key=="img" else "nearest", align_corners=False)
                    results[key] = results[key].squeeze(0)
        return results

def resize(
    input,
    size=None,
    scale_factor=None,
    mode="nearest",
    align_corners=None,
    warning=False,
):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if (
                    (output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1)
                    and (output_h - 1) % (input_h - 1)
                    and (output_w - 1) % (input_w - 1)
                ):
                    warnings.warn(
                        f"When align_corners={align_corners}, "
                        "the output would more aligned if "
                        f"input size {(input_h, input_w)} is `x+1` and "
                        f"out size {(output_h, output_w)} is `nx+1`"
                    )
    return F.interpolate(input, size, scale_factor, mode, align_corners)

class Compose(object):
    """Compose multiple transforms sequentially.

    Args:
        transforms (Sequence[dict | callable]): Sequence of transform object or
            config dict to be composed.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, Sequence)
        self.transforms = transforms

    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """

        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string

class LoadImageFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(
        self,
        to_float32=False,
        color_type="color",
        file_client_args=dict(backend="disk"),
        imdecode_backend="cv2",
    ):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = FileClient(**self.file_client_args)

        if results.get("img_prefix") is not None:
            filename = osp.join(results["img_prefix"], results["img_info"]["filename"])
        else:
            filename = results["img_info"]["filename"]
        img_bytes = self.file_client.get(filename)
        img = imfrombytes(
            img_bytes, flag=self.color_type, backend=self.imdecode_backend
        )
        if self.to_float32:
            img = img.astype(np.float32)

        results["filename"] = filename
        results["ori_filename"] = results["img_info"]["filename"]
        results["img"] = img
        results["img_shape"] = img.shape
        results["ori_shape"] = img.shape
        # Set initial values for default meta_keys
        results["pad_shape"] = img.shape
        results["scale_factor"] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results["img_norm_cfg"] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False,
        )
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(to_float32={self.to_float32},"
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str

class DepthLoadAnnotations(object):
    """Load annotations for depth estimation.

    Args:
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(
        self, file_client_args=dict(backend="disk"), imdecode_backend="pillow"
    ):
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`depth.CustomDataset`.

        Returns:
            dict: The dict contains loaded depth estimation annotations.
        """

        if self.file_client is None:
            self.file_client = FileClient(**self.file_client_args)

        if results.get("depth_prefix", None) is not None:
            filename = osp.join(
                results["depth_prefix"], results["ann_info"]["depth_map"]
            )
        else:
            filename = results["ann_info"]["depth_map"]

        depth_gt = (
            np.asarray(Image.open(filename), dtype=np.float32) / results["depth_scale"]
        )
        results["depth_gt"] = depth_gt
        results["depth_ori_shape"] = depth_gt.shape

        results["depth_fields"].append("depth_gt")
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str

class NYUCrop(object):
    """NYU standard krop when training monocular depth estimation on NYU dataset.

    Args:
        depth (bool): Whether apply NYUCrop on depth map. Default: False.
    """

    def __init__(self, depth=False):
        self.depth = depth

    def __call__(self, results):
        """Call function to apply NYUCrop on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Croped results.
        """

        if self.depth:
            depth_cropped = results["depth_gt"][45:472, 43:608]
            results["depth_gt"] = depth_cropped
            results["depth_shape"] = results["depth_gt"].shape

        img_cropped = results["img"][45:472, 43:608, :]
        results["img"] = img_cropped
        results["ori_shape"] = img_cropped.shape
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str

class RandomRotate(object):
    """Rotate the image & depth.

    Args:
        prob (float): The rotation probability.
        degree (float, tuple[float]): Range of degrees to select from. If
            degree is a number instead of tuple like (min, max),
            the range of degree will be (``-degree``, ``+degree``)
        pad_val (float, optional): Padding value of image. Default: 0.
        depth_pad_val (float, optional): Padding value of depth map.
            Default: 255.
        center (tuple[float], optional): Center point (w, h) of the rotation in
            the source image. If not specified, the center of the image will be
            used. Default: None.
        auto_bound (bool): Whether to adjust the image size to cover the whole
            rotated image. Default: False
    """

    def __init__(
        self, prob, degree, pad_val=0, depth_pad_val=0, center=None, auto_bound=False
    ):
        self.prob = prob
        assert prob >= 0 and prob <= 1
        if isinstance(degree, (float, int)):
            assert degree > 0, f"degree {degree} should be positive"
            self.degree = (-degree, degree)
        else:
            self.degree = degree
        assert len(self.degree) == 2, (
            f"degree {self.degree} should be a " f"tuple of (min, max)"
        )
        self.pal_val = pad_val
        self.depth_pad_val = depth_pad_val
        self.center = center
        self.auto_bound = auto_bound

    def __call__(self, results):
        """Call function to rotate image, depth estimation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Rotated results.
        """

        rotate = True if np.random.rand() < self.prob else False
        degree = np.random.uniform(min(*self.degree), max(*self.degree))
        if rotate:
            # rotate image
            results["img"] = imrotate(
                results["img"],
                angle=degree,
                border_value=self.pal_val,
                center=self.center,
                auto_bound=self.auto_bound,
            )

            # rotate depth
            for key in results.get("depth_fields", []):
                results[key] = imrotate(
                    results[key],
                    angle=degree,
                    border_value=self.depth_pad_val,
                    center=self.center,
                    auto_bound=self.auto_bound,
                    interpolation="nearest",
                )

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            f"(prob={self.prob}, "
            f"degree={self.degree}, "
            f"pad_val={self.pal_val}, "
            f"depth_pad_val={self.depth_pad_val}, "
            f"center={self.center}, "
            f"auto_bound={self.auto_bound})"
        )
        return repr_str

class RandomFlip(object):
    """Flip the image & depth.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        prob (float, optional): The flipping probability. Default: None.
        direction(str, optional): The flipping direction. Options are
            'horizontal' and 'vertical'. Default: 'horizontal'.
    """

    def __init__(self, prob=None, direction="horizontal"):
        self.prob = prob
        self.direction = direction
        if prob is not None:
            assert prob >= 0 and prob <= 1
        assert direction in ["horizontal", "vertical"]

    def __call__(self, results):
        """Call function to flip bounding boxes, masks, depth estimation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added into
                result dict.
        """

        if "flip" not in results:
            flip = True if np.random.rand() < self.prob else False
            results["flip"] = flip
        if "flip_direction" not in results:
            results["flip_direction"] = self.direction
        if results["flip"]:
            # flip image
            results["img"] = imflip(
                results["img"], direction=results["flip_direction"]
            )

            # flip depth
            for key in results.get("depth_fields", []):
                # use copy() to make numpy stride positive
                results[key] = imflip(
                    results[key], direction=results["flip_direction"]
                ).copy()
        return results

    def __repr__(self):
        return self.__class__.__name__ + f"(prob={self.prob})"

class RandomCrop(object):
    """Random crop the image & depth.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
    """

    def __init__(self, crop_size):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size

    def get_crop_bbox(self, img):
        """Randomly get a crop bounding box."""
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def __call__(self, results):
        """Call function to randomly crop images, depth estimation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        img = results["img"]
        crop_bbox = self.get_crop_bbox(img)

        # crop the image
        img = self.crop(img, crop_bbox)
        img_shape = img.shape
        results["img"] = img
        results["img_shape"] = img_shape
        # crop depth
        for key in results.get("depth_fields", []):
            results[key] = self.crop(results[key], crop_bbox)

        results["depth_shape"] = img_shape
        return results

    def __repr__(self):
        return self.__class__.__name__ + f"(crop_size={self.crop_size})"

class ColorAug(object):
    """Color augmentation used in depth estimation

    Args:
        prob (float, optional): The color augmentation probability. Default: None.
        gamma_range(list[int], optional): Gammar range for augmentation. Default: [0.9, 1.1].
        brightness_range(list[int], optional): Brightness range for augmentation. Default: [0.9, 1.1].
        color_range(list[int], optional): Color range for augmentation. Default: [0.9, 1.1].
    """

    def __init__(
        self,
        prob=None,
        gamma_range=[0.9, 1.1],
        brightness_range=[0.9, 1.1],
        color_range=[0.9, 1.1],
    ):
        self.prob = prob
        self.gamma_range = gamma_range
        self.brightness_range = brightness_range
        self.color_range = color_range
        if prob is not None:
            assert prob >= 0 and prob <= 1

    def __call__(self, results):
        """Call function to apply color augmentation.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly colored results.
        """
        aug = True if np.random.rand() < self.prob else False

        if aug:
            image = results["img"]

            # gamma augmentation
            gamma = np.random.uniform(min(*self.gamma_range), max(*self.gamma_range))
            image_aug = image**gamma

            # brightness augmentation
            brightness = np.random.uniform(
                min(*self.brightness_range), max(*self.brightness_range)
            )
            image_aug = image_aug * brightness

            # color augmentation
            colors = np.random.uniform(
                min(*self.color_range), max(*self.color_range), size=3
            )
            white = np.ones((image.shape[0], image.shape[1]))
            color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
            image_aug *= color_image
            image_aug = np.clip(image_aug, 0, 255)

            results["img"] = image_aug

        return results

    def __repr__(self):
        return self.__class__.__name__ + f"(prob={self.prob})"

class Normalize(object):
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """

        results["img"] = imnormalize(
            results["img"], self.mean, self.std, self.to_rgb
        )
        results["img_norm_cfg"] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(mean={self.mean}, std={self.std}, to_rgb=" f"{self.to_rgb})"
        return repr_str

class DefaultFormatBundle(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img"
    and "depth_gt". These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - depth_gt: (1)unsqueeze dim-0, (2)to tensor, (3)to DataContainer (stack=True)
    """

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """

        if "img" in results:
            img = results["img"]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            # results["img"] = DataContainer(to_tensor(img), stack=True)
            results["img"] = to_tensor(img).float()
        if "depth_gt" in results:
            # unsqueeze here
            # results["depth_gt"] = DataContainer(
            #     to_tensor(results["depth_gt"][None, ...]), stack=True
            # )
            results["depth_gt"] = to_tensor(results["depth_gt"][None, ...]).float()
        return results

    def __repr__(self):
        return self.__class__.__name__

class Collect(object):
    """Collect data from the loader relevant to the specific task.

    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img", "depth_gt".

    The "img_meta" item is always populated.  The contents of the "img_meta"
    dictionary depends on "meta_keys". By default this includes:

        - "img_shape": shape of the image input to the network as a tuple
            (h, w, c).  Note that images may be zero padded on the bottom/right
            if the batch tensor is larger than this shape.

        - "scale_factor": a float indicating the preprocessing scale

        - "flip": a boolean indicating if image flip transform was used

        - "filename": path to the image file

        - "ori_shape": original shape of the image as a tuple (h, w, c)

        - "pad_shape": image shape after padding

        - "img_norm_cfg": a dict of normalization information:
            - mean - per channel mean subtraction
            - std - per channel std divisor
            - to_rgb - bool indicating if bgr was converted to rgb

    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ``('filename', 'ori_filename', 'ori_shape', 'img_shape',
            'pad_shape', 'scale_factor', 'flip', 'flip_direction',
            'img_norm_cfg')``
    """

    def __init__(
        self,
        keys,
        meta_keys=(
            "filename",
            "ori_filename",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "scale_factor",
            "flip",
            "flip_direction",
            "img_norm_cfg",
        ),
    ):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:mmcv.DataContainer.

        Args:
            results (dict): Result dict contains the data to collect.

        Returns:
            dict: The result dict contains the following keys
                - keys in``self.keys``
                - ``img_metas``
        """

        data = {}
        img_meta = {}
        for key in self.meta_keys:
            img_meta[key] = results[key]
        # data["img_metas"] = DataContainer(img_meta, cpu_only=True)
        data["img_metas"] = img_meta
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        return (
            self.__class__.__name__ + f"(keys={self.keys}, meta_keys={self.meta_keys})"
        )

class MultiScaleFlipAug(object):
    def __init__(
        self,
        transforms,
        img_scale,
        img_ratios=None,
        flip=False,
        flip_direction="horizontal",
    ):
        self.transforms = Compose(transforms)
        if img_ratios is not None:
            img_ratios = img_ratios if isinstance(img_ratios, list) else [img_ratios]
            assert is_list_of(img_ratios, float)
        if img_scale is None:
            # mode 1: given img_scale=None and a range of image ratio
            self.img_scale = None
            assert is_list_of(img_ratios, float)
        elif isinstance(img_scale, tuple) and is_list_of(img_ratios, float):
            assert len(img_scale) == 2
            # mode 2: given a scale and a range of image ratio
            self.img_scale = [
                (int(img_scale[0] * ratio), int(img_scale[1] * ratio))
                for ratio in img_ratios
            ]
        else:
            # mode 3: given multiple scales
            self.img_scale = img_scale if isinstance(img_scale, list) else [img_scale]
        assert is_list_of(self.img_scale, tuple) or self.img_scale is None
        self.flip = flip
        self.img_ratios = img_ratios
        self.flip_direction = (
            flip_direction if isinstance(flip_direction, list) else [flip_direction]
        )
        assert is_list_of(self.flip_direction, str)

    def __call__(self, results):
        """Call function to apply test time augment transforms on results.

        Args:
            results (dict): Result dict contains the data to transform.

        Returns:
           dict[str: list]: The augmented data, where each value is wrapped
               into a list.
        """

        aug_data = []
        if self.img_scale is None and is_list_of(self.img_ratios, float):
            h, w = results["img"].shape[:2]
            img_scale = [(int(w * ratio), int(h * ratio)) for ratio in self.img_ratios]
        else:
            img_scale = self.img_scale
        flip_aug = [False, True] if self.flip else [False]
        for scale in img_scale:
            for flip in flip_aug:
                for direction in self.flip_direction:
                    _results = results.copy()
                    _results["scale"] = scale
                    _results["flip"] = flip
                    _results["flip_direction"] = direction
                    data = self.transforms(_results)
                    aug_data.append(data)
        # list of dict to dict of list
        aug_data_dict = {key: [] for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)

        return aug_data_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(transforms={self.transforms}, "
        repr_str += f"img_scale={self.img_scale}, flip={self.flip})"
        repr_str += f"flip_direction={self.flip_direction}"
        return repr_str

class ImageToTensor(object):
    """Convert image to :obj:`torch.Tensor` by given keys.

    The dimension order of input image is (H, W, C). The pipeline will convert
    it to (C, H, W). If only 2 dimension (H, W) is given, the output would be
    (1, H, W).

    Args:
        keys (Sequence[str]): Key of images to be converted to Tensor.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        """Call function to convert image in results to :obj:`torch.Tensor` and
        transpose the channel order.

        Args:
            results (dict): Result dict contains the image data to convert.

        Returns:
            dict: The result dict contains the image converted
                to :obj:`torch.Tensor` and transposed to (C, H, W) order.
        """

        for key in self.keys:
            img = results[key]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            results[key] = to_tensor(img.transpose(2, 0, 1))
        return results

    def __repr__(self):
        return self.__class__.__name__ + f"(keys={self.keys})"

class Resize(object):
    """Resize images & depth.

    This transform resizes the input image to some scale. If the input dict
    contains the key "scale", then the scale in the input dict is used,
    otherwise the specified scale in the init method is used.

    ``img_scale`` can be None, a tuple (single-scale) or a list of tuple
    (multi-scale). There are 4 multiscale modes:

    - ``ratio_range is not None``:
    1. When img_scale is None, img_scale is the shape of image in results
        (img_scale = results['img'].shape[:2]) and the image is resized based
        on the original size. (mode 1)
    2. When img_scale is a tuple (single-scale), randomly sample a ratio from
        the ratio range and multiply it with the image scale. (mode 2)

    - ``ratio_range is None and multiscale_mode == "range"``: randomly sample a
    scale from the a range. (mode 3)

    - ``ratio_range is None and multiscale_mode == "value"``: randomly sample a
    scale from multiple scales. (mode 4)

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
            Default:None.
        multiscale_mode (str): Either "range" or "value".
            Default: 'range'
        ratio_range (tuple[float]): (min_ratio, max_ratio).
            Default: None
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image. Default: True
    """

    def __init__(
        self, img_scale=None, multiscale_mode="range", ratio_range=None, keep_ratio=True
    ):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given img_scale=None and a range of image ratio
            # mode 2: given a scale and a range of image ratio
            assert self.img_scale is None or len(self.img_scale) == 1
        else:
            # mode 3 and 4: given multiple scales or a range of scales
            assert multiscale_mode in ["value", "range"]

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.

        Args:
            img_scales (list[tuple]): Images scales for selection.

        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``,
                where ``img_scale`` is the selected image scale and
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.

        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and upper bound of image scales.

        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where
                ``img_scale`` is sampled scale and None is just a placeholder
                to be consistent with :func:`random_select`.
        """

        assert is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(min(img_scale_long), max(img_scale_long) + 1)
        short_edge = np.random.randint(min(img_scale_short), max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.

        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.

        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where
                ``scale`` is sampled ratio multiplied with ``img_scale`` and
                None is just a placeholder to be consistent with
                :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into
                ``results``, which would be used by subsequent pipelines.
        """

        if self.ratio_range is not None:
            if self.img_scale is None:
                h, w = results["img"].shape[:2]
                scale, scale_idx = self.random_sample_ratio((w, h), self.ratio_range)
            else:
                scale, scale_idx = self.random_sample_ratio(
                    self.img_scale[0], self.ratio_range
                )
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == "range":
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == "value":
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results["scale"] = scale
        results["scale_idx"] = scale_idx

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        if self.keep_ratio:
            img, scale_factor = imrescale(
                results["img"], results["scale"], return_scale=True
            )
            # the w_scale and h_scale has minor difference
            # a real fix should be done in the mmcv.imrescale in the future
            new_h, new_w = img.shape[:2]
            h, w = results["img"].shape[:2]
            w_scale = new_w / w
            h_scale = new_h / h
        else:
            img, w_scale, h_scale = imresize(
                results["img"], results["scale"], return_scale=True
            )
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
        results["img"] = img
        results["img_shape"] = img.shape
        results["pad_shape"] = img.shape  # in case that there is no padding
        results["scale_factor"] = scale_factor
        results["keep_ratio"] = self.keep_ratio

    def _resize_depth(self, results):
        """Resize depth estimation map with ``results['scale']``."""
        for key in results.get("depth_fields", []):
            if self.keep_ratio:
                gt_depth = imrescale(
                    results[key], results["scale"], interpolation="nearest"
                )
            else:
                gt_depth = imresize(
                    results[key], results["scale"], interpolation="nearest"
                )
            results[key] = gt_depth

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, depth estimation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor',
                'keep_ratio' keys are added into result dict.
        """

        if "scale" not in results:
            self._random_scale(results)
        self._resize_img(results)
        self._resize_depth(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            f"(img_scale={self.img_scale}, "
            f"multiscale_mode={self.multiscale_mode}, "
            f"ratio_range={self.ratio_range}, "
            f"keep_ratio={self.keep_ratio})"
        )
        return repr_str