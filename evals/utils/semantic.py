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

# semantic transfer 辅助函数
def resize(img, target_res, resize=True, to_pil=True, edge=False):
    original_width, original_height = img.size
    original_channels = len(img.getbands())
    if not edge:
        canvas = np.zeros([target_res, target_res, 3], dtype=np.uint8)
        if original_channels == 1:
            canvas = np.zeros([target_res, target_res], dtype=np.uint8)
        if original_height <= original_width:
            if resize:
                img = img.resize((target_res, int(np.around(target_res * original_height / original_width))), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            canvas[(width - height) // 2: (width + height) // 2] = img
        else:
            if resize:
                img = img.resize((int(np.around(target_res * original_width / original_height)), target_res), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            canvas[:, (height - width) // 2: (height + width) // 2] = img
    else:
        if original_height <= original_width:
            if resize:
                img = img.resize((target_res, int(np.around(target_res * original_height / original_width))), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            top_pad = (target_res - height) // 2
            bottom_pad = target_res - height - top_pad
            img = np.pad(img, pad_width=[(top_pad, bottom_pad), (0, 0), (0, 0)], mode='edge')
        else:
            if resize:
                img = img.resize((int(np.around(target_res * original_width / original_height)), target_res), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            left_pad = (target_res - width) // 2
            right_pad = target_res - width - left_pad
            img = np.pad(img, pad_width=[(0, 0), (left_pad, right_pad), (0, 0)], mode='edge')
        canvas = img
    if to_pil:
        canvas = Image.fromarray(canvas)
    return canvas

# semantic transfer 辅助函数
def load_pascal_data(path, size=256, category='cat', split='test', same_view=False, cfg=''):
    def get_points(point_coords_list, idx):
        # x 坐标
        X = np.fromstring(point_coords_list.iloc[idx, 0], sep=";")
        # y 坐标
        Y = np.fromstring(point_coords_list.iloc[idx, 1], sep=";")
        # 最大支持 20 个关键点
        Xpad = -np.ones(20)
        Xpad[: len(X)] = X
        Ypad = -np.ones(20)
        Ypad[: len(X)] = Y
        # mask：指示哪些是有效关键点
        Zmask = np.zeros(20)
        Zmask[: len(X)] = 1
        point_coords = np.concatenate(
            (Xpad.reshape(1, 20), Ypad.reshape(1, 20), Zmask.reshape(1,20)), axis=0
        )
        # make arrays float tensor for subsequent processing
        point_coords = torch.Tensor(point_coords.astype(np.float32))
        # 3x20 数组 (X,Y,mask)
        return point_coords
    
    np.random.seed(cfg.random_seed)
    files = []
    kps = []
    # 读取图像配对 csv
    test_data = pd.read_csv('{}/{}_pairs_pf_{}_views.csv'.format(path, split, 'same' if same_view else 'different'))
    cls = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 
           'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    cls_ids = test_data.iloc[:,2].values.astype("int") - 1
    cat_id = cls.index(category)
    subset_id = np.where(cls_ids == cat_id)[0]
    print(f'Number of SPairs for {category} = {len(subset_id)}')
    subset_pairs = test_data.iloc[subset_id,:]
    src_img_names = np.array(subset_pairs.iloc[:,0])
    trg_img_names = np.array(subset_pairs.iloc[:,1])
    point_A_coords = subset_pairs.iloc[:,3:5]
    point_B_coords = subset_pairs.iloc[:,5:]
    for i in range(len(src_img_names)):
        point_coords_src = get_points(point_A_coords, i).transpose(1,0)
        point_coords_trg = get_points(point_B_coords, i).transpose(1,0)
        # 经过 transpose 转化为 (20, 3)
        src_fn= f'{path}/../{src_img_names[i]}'
        trg_fn= f'{path}/../{trg_img_names[i]}'
        src_size=Image.open(src_fn).size
        trg_size=Image.open(trg_fn).size
        source_kps = preprocess_kps_pad(point_coords_src, src_size[0], src_size[1], size)
        target_kps = preprocess_kps_pad(point_coords_trg, trg_size[0], trg_size[1], size)
        kps.append(source_kps)
        kps.append(target_kps)
        files.append(src_fn)
        files.append(trg_fn)
    kps = torch.stack(kps)
    # 筛掉未标注关键点
    used_kps, = torch.where(kps[:, :, 2].any(dim=0))
    kps = kps[:, used_kps, :]
    print(f'Final number of used key points: {kps.size(1)}')
    return files, kps, None

# semantic transfer
def semantic_transfer(model, num_cats=None, pattern="baseline", device='cuda', cfg=''):
    # 用于 DINOv2 特征提取层设定
    img_size = 840
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    torch.cuda.manual_seed(cfg.random_seed)
    torch.backends.cudnn.benchmark = True
    patch_size = 14
    stride = 14
    ph = 1 + (img_size - patch_size) // stride
    pw = 1 + (img_size - patch_size) // stride
    # choose from x_prenorm, x_norm_patchtokens
    # 从 DINOv2 中提取的特征层名称
    layer_name = 'x_norm_patchtokens'  
    # 用于存储 PCK@0.10, 0.05, 0.15 结果，分别衡量不同容差下的关键点匹配准确率
    pcks = []
    pcks_05 = []
    pcks_01 = []
    categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 
                  'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'] 
    if num_cats is not None:
        categories = categories[:num_cats]
    for cat in categories:
        files, kps, _ = load_pascal_data(
            '/home/liyh/3DCorrEnhance/data/PF-dataset-PASCAL/PF-dataset-PASCAL', 
            size=img_size, category=cat, same_view=False, cfg=cfg
        )
        gt_correspondences = []
        pred_correspondences = []
        # 每两张图像进行配对（load的时候是img1,img2,img1,img2...这样的）
        for pair_idx in tqdm(range(len(files) // 2)):
            # Load image 1
            img1 = Image.open(files[2*pair_idx]).convert('RGB')
            img1 = resize(img1, img_size, resize=True, to_pil=True, edge=False)
            img1_kps = kps[2*pair_idx]
            # Load image 2
            img2 = Image.open(files[2*pair_idx+1]).convert('RGB')
            img2 = resize(img2, img_size, resize=True, to_pil=True, edge=False)
            img2_kps = kps[2*pair_idx+1]
            # 归一化并调整为 (C, H, W) 格式
            img1 = torch.from_numpy(np.array(img1) / 255.).to(device).float().permute(2, 0, 1)
            img2 = torch.from_numpy(np.array(img2) / 255.).to(device).float().permute(2, 0, 1)
            # 使用 dinov2 提取特征
            if pattern == 'concat':
                feat_vanilla_1 = model.vit.forward_features(imagenet_norm(img1[None]))[layer_name][:, (1 if 'prenorm' in layer_name else 0):]
                feat_vanilla_1 = feat_vanilla_1.reshape(-1, ph, pw, feat_vanilla_1.shape[-1]).permute(0, 3, 1, 2)
                feat_finetuned_1 = model.finetuned_vit.forward_features(imagenet_norm(img1[None]))[layer_name][:, (1 if 'prenorm' in layer_name else 0):]
                feat_finetuned_1 = feat_finetuned_1.reshape(-1, ph, pw, feat_finetuned_1.shape[-1]).permute(0, 3, 1, 2)
                img1_desc = model.refine_conv(torch.cat([feat_vanilla_1, feat_finetuned_1], dim=1))
            elif pattern == 'finetune':
                img1_desc = model.finetuned_vit.forward_features(imagenet_norm(img1[None]))[layer_name][:, (1 if 'prenorm' in layer_name else 0):]
                # (B, C, ph, pw)
                img1_desc = img1_desc.reshape(-1, ph, pw, img1_desc.shape[-1]).permute(0, 3, 1, 2)
                # 进一步特征细化（一层卷积层）
                img1_desc = model.refine_conv(img1_desc)
            else:
                img1_desc = model.vit.forward_features(imagenet_norm(img1[None]))[layer_name][:, (1 if 'prenorm' in layer_name else 0):]
                # (B, C, ph, pw)
                img1_desc = img1_desc.reshape(-1, ph, pw, img1_desc.shape[-1]).permute(0, 3, 1, 2)
                # 进一步特征细化（一层卷积层）
                img1_desc = model.refine_conv(img1_desc)
            if pattern == 'concat':
                feat_vanilla_2 = model.vit.forward_features(imagenet_norm(img2[None]))[layer_name][:, (1 if 'prenorm' in layer_name else 0):]
                feat_vanilla_2 = feat_vanilla_2.reshape(-1, ph, pw, feat_vanilla_2.shape[-1]).permute(0, 3, 1, 2)
                feat_finetuned_2 = model.finetuned_vit.forward_features(imagenet_norm(img2[None]))[layer_name][:, (1 if 'prenorm' in layer_name else 0):]
                feat_finetuned_2 = feat_finetuned_2.reshape(-1, ph, pw, feat_finetuned_2.shape[-1]).permute(0, 3, 1, 2)
                img2_desc = model.refine_conv(torch.cat([feat_vanilla_2, feat_finetuned_2], dim=1))
            if pattern == 'finetune':
                img2_desc = model.finetuned_vit.forward_features(imagenet_norm(img2[None]))[layer_name][:, (1 if 'prenorm' in layer_name else 0):]
                img2_desc = img2_desc.reshape(-1, ph, pw, img2_desc.shape[-1]).permute(0, 3, 1, 2)
                img2_desc = model.refine_conv(img2_desc)
            else:
                img2_desc = model.vit.forward_features(imagenet_norm(img2[None]))[layer_name][:, (1 if 'prenorm' in layer_name else 0):]
                img2_desc = img2_desc.reshape(-1, ph, pw, img2_desc.shape[-1]).permute(0, 3, 1, 2)
                img2_desc = model.refine_conv(img2_desc)
            # img2 特征插值到像素级（原特征为 patch 级），为后面搜索匹配做准备
            ds_size = ( (img_size - patch_size) // stride ) * stride + 1
            img2_desc = F.interpolate(img2_desc, size=(ds_size, ds_size), mode='bilinear', align_corners=True)
            # 边缘补全，匹配原图尺寸
            img2_desc = VF.pad(img2_desc, (patch_size // 2, patch_size // 2, 
                                            img_size - img2_desc.shape[2] - (patch_size // 2), 
                                            img_size - img2_desc.shape[3] - (patch_size // 2)), 
                                padding_mode='edge')
            # 两个图中都可见的关键点
            vis = img1_kps[:, 2] * img2_kps[:, 2] > 0
            # 输出维度为 (N, F, K)，表示 K 个关键点的特征向量
            img1_kp_desc = interpolate_features(img1_desc, img1_kps[None, :, :2].to(device), 
                                                h=img_size, w=img_size, normalize=True)
            # 通过点积计算第一张图每个关键点与第二张图每个像素的特征相似度
            sim = torch.einsum('nfk,nif->nki', img1_kp_desc, 
                               img2_desc.permute(0, 2, 3, 1).reshape(1, img_size * img_size, -1))[0]
            # 得到最大相似度位置（最近邻），作为该关键点在第二张图的预测位置
            nn_idx = torch.argmax(sim, dim=1)
            nn_x = nn_idx % img_size
            nn_y = nn_idx // img_size
            kps_1_to_2 = torch.stack([nn_x, nn_y]).permute(1, 0)
            # ground truth
            gt_correspondences.append(img2_kps[vis][:, [1,0]])
            # prediction
            pred_correspondences.append(kps_1_to_2[vis][:, [1,0]])
        gt_correspondences = torch.cat(gt_correspondences, dim=0).cpu()
        pred_correspondences = torch.cat(pred_correspondences, dim=0).cpu()
        alpha = torch.tensor([0.1, 0.05, 0.15])
        correct = torch.zeros(len(alpha))
        # 欧氏距离
        err = (pred_correspondences - gt_correspondences).norm(dim=-1)
        err = err.unsqueeze(0).repeat(len(alpha), 1)
        threshold = alpha * img_size
        correct = err < threshold.unsqueeze(-1)
        # Percentage of Correct Keypoints
        correct = correct.sum(dim=-1) / len(gt_correspondences)
        alpha2pck = zip(alpha.tolist(), correct.tolist())
        print(' | '.join([f'PCK-Transfer@{alpha:.2f}: {pck_alpha * 100:.2f}%'
                        for alpha, pck_alpha in alpha2pck]))
        pck = correct
        pcks.append(pck[0])
        pcks_05.append(pck[1])
        pcks_01.append(pck[2])
    result = {}
    result['PCK0.05'] = [tensor.item() for tensor in pcks_05]
    result['PCK0.10'] = [tensor.item() for tensor in pcks]
    result['PCK0.15'] = [tensor.item() for tensor in pcks_01]
    metrics_df = pd.DataFrame(result)
    metrics_df['categories'] = categories[:num_cats]
    metrics_df.set_index(['categories'], inplace=True)
    # 用每类图像数量（近似）对不同类的 PCK 加权平均，得出整体迁移性能
    weights=[15,30,10,6,8,32,19,27,13,3,8,24,9,27,12,7,1,13,20,15][:num_cats]
    metrics_df['Weighted PCK0.05'] = np.average(metrics_df['PCK0.05'], weights=weights)
    metrics_df['Weighted PCK0.10'] = np.average(metrics_df['PCK0.10'], weights=weights)
    metrics_df['Weighted PCK0.15'] = np.average(metrics_df['PCK0.15'], weights=weights)
    return metrics_df