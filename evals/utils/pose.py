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
import os

from .functions import query_pose_error, interpolate_features, preprocess_kps_pad, _fix_pos_enc
from .tracking_metrics import compute_tapvid_metrics_for_video
from .tracking_model import ModelInference, Tracker

imagenet_norm = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

# pose
def oneposepp(module, num_objs, pattern, device):
    stride = 14
    patch_size = 14
    model = module
    root = '/home/liyh/OnePose_Plus_Plus/data/datasets/lowtexture_test_data'
    sfm_dir = '/home/liyh/3DCorrEnhance/data/sfm_output/outputs_softmax_loftr_loftr'
    all_obj = [name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))]
    if num_objs is not None:
        all_obj = all_obj[:num_objs]
    # 不同阈值
    threshold_1 = []
    threshold_3 = []
    threshold_5 = []
    for obj_name in all_obj:
        print(obj_name)
        anno_3d = np.load(f'{sfm_dir}/{obj_name}/anno/anno_3d_average.npz')
        # 3D 关键点
        keypoints3d = anno_3d['keypoints3d']
        templates = []
        # 读取该物体对应的多个模板图像关键点标注 JSON（LoFTR 风格的2D关键点+对应3D点的匹配矩阵）。
        all_json_fns = list((Path(root) / obj_name / '{}-1'.format(obj_name.split('-')[1]) / 'anno_loftr').glob('*.json'))
        for json_fn in tqdm(all_json_fns):
            idx = json_fn.stem
            anno = json.load(open(json_fn))
            keypoints2d = np.array(anno['keypoints2d'])
            assign_matrix = np.array(anno['assign_matrix'])
            # 读取模板 RGB 图片和相机内参
            rgb = cv2.imread(str(Path(root) / obj_name / '{}-1'.format(obj_name.split('-')[1]) / 'color' / f'{idx}.png'))[..., ::-1].copy()
            intrinsic = np.loadtxt(str(Path(root) / obj_name / '{}-1'.format(obj_name.split('-')[1]) / 'intrin_ba' / f'{idx}.txt'))
            # 通过 assign_matrix 将2D关键点和3D关键点对应起来
            keypoints2d = keypoints2d[assign_matrix[0]]
            kp3ds_canon = keypoints3d[assign_matrix[1]]
            # 将 RGB 图缩放到 DINOv2 输入大小（8倍下采样，再乘 patch_size 恢复）以匹配 patch 特征图
            rgb_resized = cv2.resize(rgb / 255., (rgb.shape[1] // 8 * patch_size, rgb.shape[0] // 8 * patch_size))
            if pattern == 'concat':
                feat_vanilla = model.vit.forward_features(imagenet_norm(torch.from_numpy(rgb_resized).to(device).float().permute(2, 0, 1)[None]))['x_norm_patchtokens']
                feat_vanilla = feat_vanilla.reshape(1, rgb.shape[0] // 8, rgb.shape[1] // 8, -1).permute(0, 3, 1, 2)
                feat_finetune = model.finetuned_vit.forward_features(imagenet_norm(torch.from_numpy(rgb_resized).to(device).float().permute(2, 0, 1)[None]))['x_norm_patchtokens']
                feat_finetune = feat_finetune.reshape(1, rgb.shape[0] // 8, rgb.shape[1] // 8, -1).permute(0, 3, 1, 2)
                feature = torch.cat([feat_vanilla, feat_finetune], dim=1)
                desc = model.refine_conv(feature)
            elif pattern == 'finetune':
                desc = model.finetuned_vit.forward_features(imagenet_norm(torch.from_numpy(rgb_resized).to(device).float().permute(2, 0, 1)[None]))['x_norm_patchtokens']
                desc = desc.reshape(1, rgb.shape[0] // 8, rgb.shape[1] // 8, -1).permute(0, 3, 1, 2)
                desc = module.refine_conv(desc)
            else:
                # dinov2 提取特征
                desc = model.vit.forward_features(imagenet_norm(torch.from_numpy(rgb_resized).to(device).float().permute(2, 0, 1)[None]))['x_norm_patchtokens']
                desc = desc.reshape(1, rgb.shape[0] // 8, rgb.shape[1] // 8, -1).permute(0, 3, 1, 2)
                desc = module.refine_conv(desc)
            # 关键点特征插值
            desc_temp = interpolate_features(desc, torch.from_numpy(keypoints2d).float().to(device)[None] / 8 * patch_size, 
                                            h=rgb_resized.shape[0], w=rgb_resized.shape[1], normalize=False, 
                                            patch_size=patch_size, stride=stride).permute(0, 2, 1)[0]
            # L2 归一化
            desc_temp /= (desc_temp.norm(dim=-1, keepdim=True) + 1e-9)
            kp_temp, kp3d_temp = keypoints2d, kp3ds_canon
            templates.append((kp_temp, desc_temp, kp3d_temp))
        all_descs_temp = torch.cat([t[1] for t in templates], 0).to(device)[::1]
        all_pts3d_temp = np.concatenate([t[2] for t in templates], 0)[::1]
        # subsample if too many
        if len(all_descs_temp) > 120000:
            idx = np.random.choice(len(all_descs_temp), 120000, replace=False)
            all_descs_temp = all_descs_temp[idx]
            all_pts3d_temp = all_pts3d_temp[idx]
        # 旋转误差
        R_errs = []
        # 平移误差
        t_errs = []
        # 三维点坐标的缩放因子（单位换算），通常模型中点是以米为单位，乘以1000转换成毫米，方便PnP算法计算。
        pts3d_scale = 1000
        # 后续生成均匀采样点的步长，用于在图像上稀疏采样像素点以提取特征。
        grid_stride = 4
        # 定义测试序列编号，这里是第2组图像序列。
        test_seq = '2'
        # 测试图像
        all_img_fns = list(sorted((Path(root) / obj_name / '{}-{}'.format(obj_name.split('-')[1], test_seq) / 'color').glob('*.png')))[::10]
        for i, img_fn in enumerate(tqdm(all_img_fns)):
            idx = img_fn.stem
            rgb = cv2.imread(str(Path(root) / obj_name / '{}-{}'.format(obj_name.split('-')[1], test_seq) / 'color' / f'{idx}.png'))[..., ::-1].copy()
            # 读取相机内参矩阵（intrinsic）和对应的真实相机位姿矩阵（pose_gt），用于后续PnP解算和误差计算。
            intrinsic = np.loadtxt(str(Path(root) / obj_name / '{}-{}'.format(obj_name.split('-')[1], test_seq) / 'intrin_ba' / f'{idx}.txt'))
            pose_gt = np.loadtxt(str(Path(root) / obj_name / '{}-{}'.format(obj_name.split('-')[1], test_seq) / 'poses_ba' / f'{idx}.txt'))
            with torch.no_grad():
                # 只在第一张图像时生成二维网格采样点 kp，坐标间隔为 grid_stride，网格覆盖整个图像
                if i == 0:
                    x_coords = np.arange(0, rgb.shape[1], grid_stride)
                    y_coords = np.arange(0, rgb.shape[0], grid_stride)
                    x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)
                    kp = np.column_stack((x_mesh.ravel(), y_mesh.ravel())).astype(float)
                rgb_resized = cv2.resize(rgb / 255., (rgb.shape[1] // 8 * patch_size, rgb.shape[0] // 8 * patch_size))
                if pattern == 'concat':
                    feat_vanilla = model.vit.forward_features(imagenet_norm(torch.from_numpy(rgb_resized).to(device).float().permute(2, 0, 1)[None]))['x_norm_patchtokens']
                    feat_vanilla = feat_vanilla.reshape(1, rgb.shape[0] // 8, rgb.shape[1] // 8, -1).permute(0, 3, 1, 2)
                    feat_finetune = model.finetuned_vit.forward_features(imagenet_norm(torch.from_numpy(rgb_resized).to(device).float().permute(2, 0, 1)[None]))['x_norm_patchtokens']
                    feat_finetune = feat_finetune.reshape(1, rgb.shape[0] // 8, rgb.shape[1] // 8, -1).permute(0, 3, 1, 2)
                    feature = torch.cat([feat_vanilla, feat_finetune], dim=1)
                    desc = model.refine_conv(feature)
                elif pattern == 'finetune':
                    desc = model.finetuned_vit.forward_features(imagenet_norm(torch.from_numpy(rgb_resized).to(device).float().permute(2, 0, 1)[None]))['x_norm_patchtokens']
                    desc = desc.reshape(1, rgb.shape[0] // 8, rgb.shape[1] // 8, -1).permute(0, 3, 1, 2)
                    desc = module.refine_conv(desc)
                else:
                    # dinov2 提取特征
                    desc = model.vit.forward_features(imagenet_norm(torch.from_numpy(rgb_resized).to(device).float().permute(2, 0, 1)[None]))['x_norm_patchtokens']
                    desc = desc.reshape(1, rgb.shape[0] // 8, rgb.shape[1] // 8, -1).permute(0, 3, 1, 2)
                    desc = module.refine_conv(desc)
                desc = interpolate_features(desc, torch.from_numpy(kp).float().to(device)[None] / 8 * patch_size, 
                                            h=rgb_resized.shape[0], w=rgb_resized.shape[1], normalize=False, 
                                            patch_size=patch_size, stride=stride).permute(0, 2, 1)[0]
                desc /= (desc.norm(dim=-1, keepdim=True) + 1e-9)
            with torch.no_grad():
                # 将测试图像的特征分块与模板库中的所有描述子做相似度矩阵乘法，计算余弦相似度。
                nbr1 = []
                for d in torch.split(desc, (25000 * 10000 - 1) // all_descs_temp.shape[0] + 1):
                    sim = d @ all_descs_temp.T
                    # 最近邻
                    nbr1.append(sim.argmax(-1))
                nbr1 = torch.cat(nbr1, 0)
                # 反向匹配：模板库描述子分块与测试图像描述子做相似度匹配
                nbr2 = []
                for d in torch.split(all_descs_temp, (25000 * 10000 - 1) // desc.shape[0] + 1):
                    sim = d @ desc.T
                    nbr2.append(sim.argmax(-1))
                nbr2 = torch.cat(nbr2, 0)
            # 双向一致性验证（mutual nearest neighbors）：只有测试图像的某特征匹配到模板的第nbr1[i]个特征，
            # 且模板的该特征也匹配回测试图像的第i个特征，才被视为有效匹配。
            m_mask = nbr2[nbr1] == torch.arange(len(nbr1)).to(nbr1.device)
            # 选出有效匹配点在测试图像上的二维坐标 src_pts。对应的三维点 dst_3dpts 来自模板三维点集合。
            src_pts = kp[m_mask.cpu().numpy()].reshape(-1,1,2)
            dst_3dpts =  all_pts3d_temp[nbr1[m_mask].cpu().numpy()]
            pose_pred = np.eye(4)
            if m_mask.sum() >= 4:
                # 如果有效匹配点≥4个，使用OpenCV的solvePnPRansac算法求解相机位姿（旋转向量 R_exp 和平移向量 trans）
                _, R_exp, trans, pnp_inlier = cv2.solvePnPRansac(dst_3dpts * pts3d_scale, src_pts[:, 0],
                                                        intrinsic, None, reprojectionError=8.0,
                                                        iterationsCount=10000, flags=cv2.SOLVEPNP_EPNP)
                trans /= pts3d_scale
                if pnp_inlier is not None:
                    if len(pnp_inlier) > 5:
                        # 如果内点数目大于5，转换旋转向量为旋转矩阵 R
                        R, _ = cv2.Rodrigues(R_exp)
                        r_t = np.concatenate([R, trans], axis=-1)
                        # 拼接旋转矩阵和平移向量构造4x4齐次变换矩阵 pose_pred
                        pose_pred = np.concatenate((r_t, [[0, 0, 0, 1]]), axis=0)
            R_err, t_err = query_pose_error(pose_pred, pose_gt)
            R_errs.append(R_err)
            t_errs.append(t_err)
        print(f'object: {obj_name}')
        # 对3个不同阈值（1度/单位，3度/单位，5度/单位）分别计算估计的准确率（旋转和平移误差同时小于阈值的比例）
        for pose_threshold in [1, 3, 5]:
            acc = np.mean((np.array(R_errs) < pose_threshold) & (np.array(t_errs) < pose_threshold))
            print(f'pose_threshold: {pose_threshold}, acc: {acc}')
            if pose_threshold == 1:
                threshold_1.append(acc)
            elif pose_threshold == 3:
                threshold_3.append(acc)
            else:
                threshold_5.append(acc)
    result = {}
    result['threshold_1'] = threshold_1
    result['threshold_3'] = threshold_3
    result['threshold_5'] = threshold_5
    metrics_df = pd.DataFrame(result)
    metrics_df['objs'] = all_obj
    metrics_df.set_index(['objs'], inplace=True)
    return metrics_df