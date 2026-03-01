import pickle
import types
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.modules.utils as nn_utils
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
import cv2
import json
import torch.nn.functional as F
import torchvision.transforms.functional as VF

from .functions import query_pose_error, interpolate_features, preprocess_kps_pad, _fix_pos_enc
from .tracking_metrics import compute_tapvid_metrics_for_video
from .tracking_model import ModelInference, Tracker

imagenet_norm = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

# video tracking
def tracking_single(video_id, model, device):
    # for tracking we use stride 7
    stride = 7
    patch_size = 14
    h, w = 476, 854
    if h % patch_size != 0 or w % patch_size != 0:
        # 进行截断处理
        print(f'Warning: image size ({h}, {w}) is not divisible by patch size {patch_size}')
        h = h // patch_size * patch_size
        w = w // patch_size * patch_size
        print(f'New image size: {h}, {w}')
    video_root = Path(f'/home/liyh/3DCorrEnhance/data/tapvid-davis/{video_id}')
    images = []
    for img_fn in sorted((video_root / 'video').glob('*.jpg')):
        images.append(np.array(Image.open(img_fn).resize((w, h), Image.LANCZOS)))
    images = np.stack(images)
    # (N, channels, H, W)
    images = torch.from_numpy(images).permute(0, 3, 1, 2).float().to(device) / 255.0
    features = []
    for image in tqdm(images):
        # 特征 patch 网络的宽和高
        ph = 1 + (h - patch_size) // stride
        pw = 1 + (w - patch_size) // stride
        # fix the stride
        # _pair(stride) 将单个整数转换为 (stride, stride)。
        stride_pair = nn_utils._pair(stride)
        # 手动更改 DINOv2 的 patch embedding 卷积的 stride，从默认 14 改成 7，达到更密集的特征提取（提高空间分辨率）
        model.vit.patch_embed.proj.stride = stride_pair
        # fix the positional encoding code
        # 由于更改了 stride，原本位置编码（positional encoding）尺寸不再匹配，因此需要手动替换 DINOv2 的位置编码插值函数
        model.vit.interpolate_pos_encoding = types.MethodType(_fix_pos_enc(patch_size, stride_pair), model.vit)
        feature = model.vit.forward_features(imagenet_norm(image[None].to(device)))["x_prenorm"]
        # 去除非图像区域的 token
        feature = feature[:, 1 + model.vit.num_register_tokens:]
        feature = feature.reshape(-1, ph, pw, feature.shape[-1]).permute(0, 3, 1, 2)
        feature = model.refine_conv(feature)
        features.append(feature)
    features = torch.cat(features)
    dino_tracker = Tracker(features, images, dino_patch_size=patch_size, stride=stride)
    # 初始化一个 ModelInference 实例，它负责将一个查询点在当前视频中进行追踪，产生其全视频轨迹及遮挡预测。
    anchor_cosine_similarity_threshold = 0.7
    cosine_similarity_threshold = 0.6
    model_inference = ModelInference(
        model=dino_tracker,
        range_normalizer=dino_tracker.range_normalizer,
        anchor_cosine_similarity_threshold=anchor_cosine_similarity_threshold,
        cosine_similarity_threshold=cosine_similarity_threshold,
    )
    # 获取模型内部存储的视频图像尺寸（注意：宽度先于高度）；实际值是 [width, height]
    rescale_sizes=[dino_tracker.video.shape[-1], dino_tracker.video.shape[-2]]
    # 加载追踪任务的基准配置文件（包含 ground truth、点位、视频尺寸等）
    benchmark_config = pickle.load(open('/home/liyh/3DCorrEnhance/data/tapvid_davis_data_strided.pkl', "rb"))
    # 找到当前视频的 ground truth
    for video_config in benchmark_config["videos"]:
        # print(video_config["video_idx"])
        if video_config["video_idx"] == video_id:
            break
    rescale_factor_x = rescale_sizes[0] / video_config['w']
    rescale_factor_y = rescale_sizes[1] / video_config['h']
    # 记录每一帧中所有查询点的重新缩放后的坐标和时间戳
    query_points_dict = {}
    for frame_idx, q_pts_at_frame in video_config['query_points'].items():
        query_points_at_frame = []
        for q_point in q_pts_at_frame:
            query_points_at_frame.append([rescale_factor_x * q_point[0], rescale_factor_y * q_point[1], frame_idx])
        query_points_dict[frame_idx] = query_points_at_frame
    # 每个点在每一帧的轨迹
    trajectories_dict = {}
    # 每个点在每一帧是否被遮挡
    occlusions_dict = {}
    for frame_idx in tqdm(sorted(query_points_dict.keys()), desc="Predicting trajectories"):
        # N x 3, (x, y, t)
        qpts_st_frame = torch.tensor(query_points_dict[frame_idx], dtype=torch.float32, device=device) 
        # N x T x 3, N x T
        trajectories_at_st_frame, occlusion_at_st_frame = model_inference.infer(query_points=qpts_st_frame, batch_size=None) 
        # 去除时间通道
        trajectories = trajectories_at_st_frame[..., :2].cpu().detach().numpy()
        occlusions = occlusion_at_st_frame.cpu().detach().numpy()
        trajectories_dict[frame_idx] = trajectories
        occlusions_dict[frame_idx] = occlusions
    # only test video id 0 for now    
    metrics = compute_tapvid_metrics_for_video(trajectories_dict=trajectories_dict, occlusions_dict=occlusions_dict,
                                                video_idx=video_id, benchmark_data=benchmark_config,
                                                pred_video_sizes=[w, h])
    metrics["video_idx"] = video_id
    return metrics

def tracking(model, num_videos, device):
    metrics_list = []
    for id in range(num_videos):
        metrics = tracking_single(id, model=model, pattern=pattern, device=device)
        metrics_list.append(metrics)
        print(metrics)
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.set_index(['video_idx'], inplace=True)
    return metrics_df