import hydra
import torch
import pandas as pd
import torch.nn.modules.utils as nn_utils
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from pathlib import Path
import time
import torch.nn.modules.utils as nn_utils
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
import cv2
import json
import torch.nn.functional as F
import torchvision.transforms.functional as VF
import numpy as np

from evals.datasets.tapvid_video import TapVidDataset
from evals.utils.tracking_metrics import compute_tapvid_metrics_for_video
from evals.utils.tracking_model import ModelInference, Tracker
from datetime import datetime


device = 'cuda'
imagenet_norm = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


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


def preprocess_kps_pad(kps, img_width, img_height, size):
    # Once an image has been pre-processed via border (or zero) padding,
    # the location of key points needs to be updated. This function applies
    # that pre-processing to the key points so they are correctly located
    # in the border-padded (or zero-padded) image.
    kps = kps.clone()
    scale = size / max(img_width, img_height)
    kps[:, [0, 1]] *= scale
    if img_height < img_width:
        new_h = int(np.around(size * img_height / img_width))
        offset_y = int((size - new_h) / 2)
        offset_x = 0
        kps[:, 1] += offset_y
    elif img_width < img_height:
        new_w = int(np.around(size * img_width / img_height))
        offset_x = int((size - new_w) / 2)
        offset_y = 0
        kps[:, 0] += offset_x
    else:
        offset_x = 0
        offset_y = 0
    kps *= kps[:, 2:3].clone()  # zero-out any non-visible key points
    return kps, offset_x, offset_y, scale


def load_pascal_data(path, size=256, category='cat', split='test', same_view=False):
    
    def get_points(point_coords_list, idx):
        X = np.fromstring(point_coords_list.iloc[idx, 0], sep=";")
        Y = np.fromstring(point_coords_list.iloc[idx, 1], sep=";")
        Xpad = -np.ones(20)
        Xpad[: len(X)] = X
        Ypad = -np.ones(20)
        Ypad[: len(X)] = Y
        Zmask = np.zeros(20)
        Zmask[: len(X)] = 1
        point_coords = np.concatenate(
            (Xpad.reshape(1, 20), Ypad.reshape(1, 20), Zmask.reshape(1,20)), axis=0
        )
        # make arrays float tensor for subsequent processing
        point_coords = torch.Tensor(point_coords.astype(np.float32))
        return point_coords
    
    np.random.seed(42)
    files = []
    kps = []
    test_data = pd.read_csv('{}/{}_pairs_pf_{}_views.csv'.format(path, split, 'same' if same_view else 'different'))
    cls = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                    'bus', 'car', 'cat', 'chair', 'cow',
                    'diningtable', 'dog', 'horse', 'motorbike', 'person',
                    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    cls_ids = test_data.iloc[:,2].values.astype("int") - 1
    cat_id = cls.index(category)
    subset_id = np.where(cls_ids == cat_id)[0]
    print(f'Number of SPairs for {category} = {len(subset_id)}')
    subset_pairs = test_data.iloc[subset_id,:]
    src_img_names = np.array(subset_pairs.iloc[:,0])
    trg_img_names = np.array(subset_pairs.iloc[:,1])
    # print(src_img_names.shape, trg_img_names.shape)
    point_A_coords = subset_pairs.iloc[:,3:5]
    point_B_coords = subset_pairs.iloc[:,5:]
    # print(point_A_coords.shape, point_B_coords.shape)
    for i in range(len(src_img_names)):
        point_coords_src = get_points(point_A_coords, i).transpose(1,0)
        point_coords_trg = get_points(point_B_coords, i).transpose(1,0)
        src_fn= f'{path}/../{src_img_names[i]}'
        trg_fn= f'{path}/../{trg_img_names[i]}'
        src_size=Image.open(src_fn).size
        trg_size=Image.open(trg_fn).size
        # print(src_size)
        source_kps, src_x, src_y, src_scale = preprocess_kps_pad(point_coords_src, src_size[0], src_size[1], size)
        target_kps, trg_x, trg_y, trg_scale = preprocess_kps_pad(point_coords_trg, trg_size[0], trg_size[1], size)
        kps.append(source_kps)
        kps.append(target_kps)
        files.append(src_fn)
        files.append(trg_fn)
    
    kps = torch.stack(kps)
    used_kps, = torch.where(kps[:, :, 2].any(dim=0))
    kps = kps[:, used_kps, :]
    print(f'Final number of used key points: {kps.size(1)}')
    return files, kps, None


def semantic_transfer(model, num_cats=None):
    # img_size = 840
    img_size = 700
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.benchmark = True

    patch_size = model.patch_size
    stride = patch_size
    img_size = (img_size // patch_size) * patch_size

    ph = 1 + (img_size - patch_size) // stride
    pw = 1 + (img_size - patch_size) // stride

    layer_name = 'x_norm_patchtokens'  # choose from x_prenorm, x_norm_patchtokens

    pcks = []
    pcks_05 = []
    pcks_01 = []
    
    categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                    'bus', 'car', 'cat', 'chair', 'cow',
                    'diningtable', 'dog', 'horse', 'motorbike', 'person',
                    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'] # for pascal
    
    if num_cats is not None:
        categories = categories[:num_cats]

    for cat in categories:
        files, kps, _ = load_pascal_data('./data/PF-dataset-PASCAL', size=img_size, category=cat, same_view=True)
        
        gt_correspondences = []
        pred_correspondences = []
        for pair_idx in tqdm(range(len(files) // 2)):
            # Load image 1
            img1 = Image.open(files[2*pair_idx]).convert('RGB')
            img1 = resize(img1, img_size, resize=True, to_pil=True, edge=False)
            img1_kps = kps[2*pair_idx]

            # # Get patch index for the keypoints
            img1_y, img1_x = img1_kps[:, 1].numpy(), img1_kps[:, 0].numpy()

            # Load image 2
            img2 = Image.open(files[2*pair_idx+1]).convert('RGB')
            img2 = resize(img2, img_size, resize=True, to_pil=True, edge=False)
            img2_kps = kps[2*pair_idx+1]

            # Get patch index for the keypoints
            img2_y, img2_x = img2_kps[:, 1].numpy(), img2_kps[:, 0].numpy()
            
            img1 = torch.from_numpy(np.array(img1) / 255.).cuda().float().permute(2, 0, 1)
            img2 = torch.from_numpy(np.array(img2) / 255.).cuda().float().permute(2, 0, 1)

            # img1_desc = model.dinov2.forward_features(imagenet_norm(img1[None]))[layer_name][:, (1 if 'prenorm' in layer_name else 0):]
            # img1_desc = img1_desc.reshape(-1, ph, pw, img1_desc.shape[-1]).permute(0, 3, 1, 2)

            # img2_desc = model.dinov2.forward_features(imagenet_norm(img2[None]))[layer_name][:, (1 if 'prenorm' in layer_name else 0):]
            # img2_desc = img2_desc.reshape(-1, ph, pw, img2_desc.shape[-1]).permute(0, 3, 1, 2)

            # img1_desc = model.refine_conv(img1_desc)
            # img2_desc = model.refine_conv(img2_desc)
            img1_desc = model(imagenet_norm(img1[None]))
            img2_desc = model(imagenet_norm(img2[None]))
            
            ds_size = ( (img_size - patch_size) // stride ) * stride + 1
            img2_desc = F.interpolate(img2_desc, size=(ds_size, ds_size), mode='bilinear', align_corners=True)
            img2_desc = VF.pad(img2_desc, (patch_size // 2, patch_size // 2, 
                                                                        img_size - img2_desc.shape[2] - (patch_size // 2), 
                                                                        img_size - img2_desc.shape[3] - (patch_size // 2)), padding_mode='edge')
            
            
            vis = img1_kps[:, 2] * img2_kps[:, 2] > 0
            img1_kp_desc = interpolate_features(img1_desc, img1_kps[None, :, :2].cuda(), h=img_size, w=img_size, normalize=True) # N x F x K
            sim = torch.einsum('nfk,nif->nki', img1_kp_desc, img2_desc.permute(0, 2, 3, 1).reshape(1, img_size * img_size, -1))[0]
            nn_idx = torch.argmax(sim, dim=1)
            nn_x = nn_idx % img_size
            nn_y = nn_idx // img_size
            kps_1_to_2 = torch.stack([nn_x, nn_y]).permute(1, 0)
            
            gt_correspondences.append(img2_kps[vis][:, [1,0]])
            pred_correspondences.append(kps_1_to_2[vis][:, [1,0]])
        
        gt_correspondences = torch.cat(gt_correspondences, dim=0).cpu()
        pred_correspondences = torch.cat(pred_correspondences, dim=0).cpu()
        alpha = torch.tensor([0.1, 0.05, 0.15])
        correct = torch.zeros(len(alpha))

        err = (pred_correspondences - gt_correspondences).norm(dim=-1)
        err = err.unsqueeze(0).repeat(len(alpha), 1)
        threshold = alpha * img_size
        correct = err < threshold.unsqueeze(-1)
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
    
    weights=[15,30,10,6,8,32,19,27,13,3,8,24,9,27,12,7,1,13,20,15][:num_cats]

    metrics_df['Weighted PCK0.05'] = np.average(metrics_df['PCK0.05'], weights=weights)
    metrics_df['Weighted PCK0.10'] = np.average(metrics_df['PCK0.10'], weights=weights)
    metrics_df['Weighted PCK0.15'] = np.average(metrics_df['PCK0.15'], weights=weights)
    return metrics_df


@hydra.main("./configs", "pascal_pf", None)
def main(cfg: DictConfig):
    print(f"Config: \n {OmegaConf.to_yaml(cfg)}")
    # ===== Get model and dataset ====
    model = instantiate(cfg.backbone, output="dense", return_multilayer=cfg.multilayer)
    model = model.to(device).to(torch.float32).eval()
    patch_size = model.patch_size

    metrics_transfer = semantic_transfer(model)
    out_dir = Path("./logs")
    metrics_transfer.to_csv(out_dir / 'semantic_transfer.csv')
    
    exp_info = cfg.backbone
    results = ", ".join(metrics_transfer.mean().values.astype(str))
    log = f"{time}, {exp_info}, pascal_pf, {results} \n"
    with open(cfg.log_file, "a") as f:
        f.write(log)
    print(metrics_transfer.mean())


if __name__ == "__main__":
    main()
