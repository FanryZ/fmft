from datetime import datetime
import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from pathlib import Path
import time
import os
import cv2
import torch.nn.functional as F
import torchvision.transforms.functional as VF
import torch.nn as nn
import torch
import torch.nn.modules.utils as nn_utils
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
import cv2
import json
import torch.nn.functional as F
import torchvision.transforms.functional as VF
import os
import pandas as pd


def interpolate_features(descriptors, pts, h, w, normalize=True, patch_size=14, stride=14):
    # patch token 中心在像素坐标上的最大值
    last_coord_h = ( (h - patch_size) // stride ) * stride + (patch_size / 2)
    last_coord_w = ( (w - patch_size) // stride ) * stride + (patch_size / 2)
    ah = 2 / (last_coord_h - (patch_size / 2))
    aw = 2 / (last_coord_w - (patch_size / 2))
    bh = 1 - last_coord_h * 2 / ( last_coord_h - ( patch_size / 2 ))
    bw = 1 - last_coord_w * 2 / ( last_coord_w - ( patch_size / 2 ))
    # 把关键点从像素坐标空间转换到 [-1, 1] 的网格坐标空间，匹配 F.grid_sample 的要求。
    a = torch.tensor([[aw, ah]]).to(pts).float()
    b = torch.tensor([[bw, bh]]).to(pts).float()
    keypoints = a * pts + b
    # Expand dimensions for grid sampling
    # F.grid_sample 要求 grid 输入形状为 [B, H_out, W_out, 2]，所以这里把关键点作为一行多个点。
    keypoints = keypoints.unsqueeze(-3)  # Shape becomes [batch_size, 1, num_keypoints, 2]
    # Interpolate using bilinear sampling
    interpolated_features = F.grid_sample(descriptors, keypoints, align_corners=True, padding_mode='border')
    # interpolated_features will have shape [batch_size, channels, 1, num_keypoints]
    interpolated_features = interpolated_features.squeeze(-2)
    return F.normalize(interpolated_features, dim=1) if normalize else interpolated_features


# pose 的辅助函数
def query_pose_error(pose_pred, pose_gt, unit='m'):
    """
    Input:
    -----------
    pose_pred: np.array 3*4 or 4*4
    前三列为旋转矩阵，后一列为平移向量
    pose_gt: np.array 3*4 or 4*4
    """
    # Dim check:
    if pose_pred.shape[0] == 4:
        pose_pred = pose_pred[:3]
    if pose_gt.shape[0] == 4:
        pose_gt = pose_gt[:3]
    # Convert results' unit to cm
    # 平移向量欧氏距离
    if unit == 'm':
        translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_gt[:, 3]) * 100
    elif unit == 'cm':
        translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_gt[:, 3])
    elif unit == 'mm':
        translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_gt[:, 3]) / 10
    else:
        raise NotImplementedError
    # 从真实旋转到预测旋转的相对旋转
    rotation_diff = np.dot(pose_pred[:, :3], pose_gt[:, :3].T)
    # 计算旋转差矩阵的迹（矩阵对角元素之和），迹的值范围理论上是[-1, 3]。
    trace = np.trace(rotation_diff)
    # 由于数值误差可能导致迹大于3，这里限定最大值为3，避免数值问题
    trace = trace if trace <= 3 else 3
    # 旋转矩阵 R 的迹满足关系：trace(R) = 1 + 2 * cos(θ)，其中θ是旋转角度。所以 θ = arccos((trace(R) - 1) / 2)
    # np.arccos 返回弧度，np.rad2deg 转换成角度。这个角度即是预测旋转与真实旋转之间的旋转误差，单位是度。
    angular_distance = np.rad2deg(np.arccos((trace - 1.0) / 2.0))
    return angular_distance, translation_distance


@hydra.main("./configs", "onepose_pose", None)
def main(cfg: DictConfig):
    print(f"Config: \n {OmegaConf.to_yaml(cfg)}")

    device = "cuda"
    model = instantiate(cfg.backbone, output="dense", return_multilayer=cfg.multilayer)
    model = model.to(device).eval()
    patch_size = model.patch_size
    imagenet_norm = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    root = './data/lowtexture_test_data'
    sfm_dir = './data/outputs_softmax_loftr_loft'
    all_obj = [name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))]
    # if num_objs is not None:
    #     all_obj = all_obj[:num_objs]
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

            with torch.no_grad():
                desc = model(imagenet_norm(torch.from_numpy(rgb_resized).to(device).float().permute(2, 0, 1)[None]))
                desc_temp = interpolate_features(desc, torch.from_numpy(keypoints2d).float().to(device)[None] / 8 * patch_size, 
                                        h=rgb_resized.shape[0], w=rgb_resized.shape[1], normalize=False, 
                                        patch_size=patch_size, stride=patch_size).permute(0, 2, 1)[0]
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
                if i == 0:
                    x_coords = np.arange(0, rgb.shape[1], grid_stride)
                    y_coords = np.arange(0, rgb.shape[0], grid_stride)
                    x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)
                    kp = np.column_stack((x_mesh.ravel(), y_mesh.ravel())).astype(float)
                rgb_resized = cv2.resize(rgb / 255., (rgb.shape[1] // 8 * patch_size, rgb.shape[0] // 8 * patch_size))
                # rgb_resized = cv2.resize(rgb / 255., (rgb.shape[1] // 8 * patch_size, rgb.shape[0] // 8 * patch_size))
                
                desc = model(imagenet_norm(torch.from_numpy(rgb_resized).to(device).float().permute(2, 0, 1)[None]))
                desc = interpolate_features(desc, torch.from_numpy(kp).float().to(device)[None] / 8 * patch_size, 
                                            h=rgb_resized.shape[0], w=rgb_resized.shape[1], normalize=False, 
                                            patch_size=patch_size, stride=patch_size).permute(0, 2, 1)[0]
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
            dst_3dpts = all_pts3d_temp[nbr1[m_mask].cpu().numpy()]
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

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    output_path = Path(f'/onepose_exp/{cfg.backbone}/pose/{timestamp}.csv')
    
    exp_info = cfg.backbone
    dset = 'onepose'
    mean_result = {
        "threshold_1": np.mean(threshold_1),
        "threshold_3": np.mean(threshold_3),
        "threshold_5": np.mean(threshold_5)
    }
    log = f"{time}, {exp_info}, {dset}, {mean_result} \n"
    with open(cfg.log_file, "a") as f:
        f.write(log)
    # output_path.parent.mkdir(parents=True, exist_ok=True)
    # metrics_df.to_csv(output_path)
    # mean = metrics_df.mean()
    # txt_path = output_path.with_suffix('.txt')
    # with open(txt_path, 'w') as f:
    #     f.write(str(mean))
    # print(mean)

if __name__ == "__main__":
    main()
