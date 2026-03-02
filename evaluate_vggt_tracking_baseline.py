#!/usr/bin/env python3
"""Evaluate baseline tasks with pretrained VGGT tracking outputs.

Unlike feature-backbone baselines, this script uses VGGT's native tracking module to
predict point correspondences (and pose where applicable) directly from image pairs.
No training / finetuning is performed.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from evals.datasets.builder import build_loader
from evals.datasets.scannet_pairs import ScanNetPairsDataset
from evals.datasets.tapvid_video import TapVidDataset
from evals.utils.tracking_metrics import compute_tapvid_metrics_for_video
from evals.utils.transformations import so3_rotation_angle, transform_points_Rt
from evaluate_pascal_pf import load_pascal_data, resize

from vggt.models.vggt import VGGT


def project_3dto2d(xyz: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    uv = xyz @ K.T
    return uv[:, :2] / (uv[:, 2:3] + 1e-8)


def compute_binned_performance(metric: torch.Tensor, bins_val: torch.Tensor, bins=[0, 30, 60, 90, 120]):
    out = []
    for i in range(len(bins) - 1):
        m = (bins_val >= bins[i]) & (bins_val < bins[i + 1])
        out.append(metric[m].mean().item() if m.any() else float('nan'))
    return out



def tensor_to_bgr_uint8(img_chw: torch.Tensor) -> np.ndarray:
    arr = img_chw.detach().cpu().permute(1, 2, 0).numpy()
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def save_pair_correspondence_vis(
    img0_chw: torch.Tensor,
    img1_chw: torch.Tensor,
    src_uv: torch.Tensor,
    dst_uv: torch.Tensor,
    save_path: Path,
    max_draw: int = 300,
    seed: int = 0,
):
    img0 = tensor_to_bgr_uint8(img0_chw)
    img1 = tensor_to_bgr_uint8(img1_chw)
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]

    canvas = np.zeros((max(h0, h1), w0 + w1, 3), dtype=np.uint8)
    canvas[:h0, :w0] = img0
    canvas[:h1, w0:w0 + w1] = img1

    n = len(src_uv)
    if n == 0:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), canvas)
        return

    k = min(max_draw, n)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=k, replace=False)

    src = src_uv[idx].detach().cpu().numpy()
    dst = dst_uv[idx].detach().cpu().numpy()

    for p0, p1 in zip(src, dst):
        color = tuple(int(c) for c in rng.integers(0, 255, size=3))
        x0, y0 = int(round(p0[0])), int(round(p0[1]))
        x1, y1 = int(round(p1[0])) + w0, int(round(p1[1]))
        cv2.circle(canvas, (x0, y0), 2, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (x1, y1), 2, color, -1, lineType=cv2.LINE_AA)
        cv2.line(canvas, (x0, y0), (x1, y1), color, 1, lineType=cv2.LINE_AA)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), canvas)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VGGT tracking-module evaluation on baseline tasks.")
    parser.add_argument("--tasks", nargs="+", default=["all"],
                        choices=["all", "navi", "scannet_corr", "scannet_pose", "onepose_pose", "pascal_pf", "tapvid"],
                        help="Tasks to run. Default: all.")
    parser.add_argument("--log-file", default="./logs/vggt_tracking_baseline.log")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-items", type=int, default=-1, help="Debug cap per task; -1 means full dataset.")
    parser.add_argument("--num-corr", type=int, default=2048)
    parser.add_argument("--grid-step", type=int, default=14)
    parser.add_argument("--conf-thr", type=float, default=0.35)
    parser.add_argument("--vis-thr", type=float, default=0.35)
    parser.add_argument("--video-num", type=int, default=-1)
    parser.add_argument("--query-num", type=int, default=2000, help="Random query count from valid GT points for correspondence tasks.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for GT query sampling.")
    parser.add_argument("--save-correspondence-vis", action="store_true", help="Save correspondence visualization images for test pairs.")
    parser.add_argument("--vis-dir", default="./logs/vggt_tracking_vis", help="Output directory for correspondence visualization images.")
    parser.add_argument("--vis-max-pairs", type=int, default=20, help="Max number of pair visualizations per task.")
    return parser.parse_args()


class VGGTTracker:
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
        self.model = VGGT.from_pretrained("facebook/VGGT-1B").to(self.device).eval()
        if self.device.type == "cuda":
            major = torch.cuda.get_device_capability(self.device)[0]
            self.amp_dtype = torch.bfloat16 if major >= 8 else torch.float16
        else:
            self.amp_dtype = torch.float32

    def _run(self, images: torch.Tensor, query_points: torch.Tensor):
        with torch.no_grad():
            if self.device.type == "cuda":
                with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                    preds = self.model(images.to(self.device), query_points.to(self.device))
            else:
                preds = self.model(images.to(self.device), query_points.to(self.device))
        return preds

    @staticmethod
    def _ceil_to_multiple(x: int, multiple: int) -> int:
        return int(np.ceil(x / multiple) * multiple)

    def _prepare_pair(
        self,
        image0: torch.Tensor,
        image1: torch.Tensor,
        query_points_xy: torch.Tensor,
    ):
        """Pad a pair to a common spatial size divisible by patch size.

        This keeps original pixel coordinates valid while allowing any input H/W,
        including non-multiples of 14 and mismatched pair sizes.
        """
        patch = int(self.model.track_head.patch_size)
        h0, w0 = image0.shape[-2:]
        h1, w1 = image1.shape[-2:]

        out_h = self._ceil_to_multiple(max(h0, h1), patch)
        out_w = self._ceil_to_multiple(max(w0, w1), patch)

        pad0 = (0, out_w - w0, 0, out_h - h0)  # left, right, top, bottom
        pad1 = (0, out_w - w1, 0, out_h - h1)
        image0_pad = F.pad(image0, pad0, mode="replicate")
        image1_pad = F.pad(image1, pad1, mode="replicate")

        # Keep query points inside source image bounds.
        q = query_points_xy.clone().float()
        if q.numel() > 0:
            q[:, 0] = q[:, 0].clamp(0, w0 - 1)
            q[:, 1] = q[:, 1].clamp(0, h0 - 1)

        return image0_pad, image1_pad, q, (h0, w0, h1, w1)

    def track_pair(self, image0: torch.Tensor, image1: torch.Tensor, query_points_xy: torch.Tensor):
        image0, image1, query_points_xy, (h0, w0, h1, w1) = self._prepare_pair(image0, image1, query_points_xy)
        images = torch.stack([image0, image1], dim=0)[None]  # [1,2,3,H,W]
        preds = self._run(images, query_points_xy[None])
        tracks = preds["track"][0]  # [2,N,2]
        vis = preds["vis"][0, 1]
        conf = preds["conf"][0, 1]
        src = tracks[0]
        dst = tracks[1]

        # Clip tracks back to each original image extent.
        src[:, 0] = src[:, 0].clamp(0, w0 - 1)
        src[:, 1] = src[:, 1].clamp(0, h0 - 1)
        dst[:, 0] = dst[:, 0].clamp(0, w1 - 1)
        dst[:, 1] = dst[:, 1].clamp(0, h1 - 1)
        return src.detach().cpu(), dst.detach().cpu(), vis.detach().cpu(), conf.detach().cpu()


def make_grid_queries(h: int, w: int, step: int) -> torch.Tensor:
    ys = torch.arange(step / 2, h, step)
    xs = torch.arange(step / 2, w, step)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1).float()


def sample_random_queries_from_valid_mask(mask: torch.Tensor, num_points: int, seed: int = 0) -> torch.Tensor:
    """Randomly sample (x, y) query points from a valid-pixel mask."""
    if mask.ndim != 2:
        raise ValueError(f"Expected mask shape [H,W], got {tuple(mask.shape)}")
    ys, xs = torch.where(mask)
    if len(xs) == 0:
        h, w = mask.shape
        return make_grid_queries(h, w, step=14)

    total = len(xs)
    k = min(num_points, total)
    g = torch.Generator(device=xs.device)
    g.manual_seed(int(seed))
    perm = torch.randperm(total, generator=g, device=xs.device)[:k]
    return torch.stack([xs[perm].float(), ys[perm].float()], dim=1)


def valid_mask_from_xyz_grid(xyz_grid: torch.Tensor) -> torch.Tensor:
    if xyz_grid.ndim == 4:
        xyz_grid = xyz_grid[0]
    if xyz_grid.ndim != 3:
        raise ValueError(f"Expected xyz_grid shape [3,H,W], got {tuple(xyz_grid.shape)}")
    return torch.isfinite(xyz_grid).all(dim=0) & (xyz_grid.norm(dim=0) > 1e-8)


def valid_mask_from_depth(depth: torch.Tensor) -> torch.Tensor:
    if depth.ndim == 3:
        depth = depth[0]
    if depth.ndim != 2:
        raise ValueError(f"Expected depth shape [H,W] or [1,H,W], got {tuple(depth.shape)}")
    return torch.isfinite(depth) & (depth > 1e-8)
def sample_map_at_uv(map_hwc: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    # map_hwc: [C,H,W] or [1,H,W], uv in pixel coords [N,2]
    if map_hwc.ndim == 2:
        map_hwc = map_hwc.unsqueeze(0)
    c, h, w = map_hwc.shape
    x = (uv[:, 0] / (w - 1)) * 2 - 1
    y = (uv[:, 1] / (h - 1)) * 2 - 1
    grid = torch.stack([x, y], dim=-1).view(1, -1, 1, 2)
    sampled = F.grid_sample(map_hwc.unsqueeze(0).float(), grid.float(), align_corners=True, mode="bilinear")
    return sampled[0, :, :, 0].T  # [N,C]


def uv_depth_to_xyz(uv: torch.Tensor, depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    z = sample_map_at_uv(depth, uv).squeeze(-1)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (uv[:, 0] - cx) * z / fx
    y = (uv[:, 1] - cy) * z / fy
    return torch.stack([x, y, z], dim=1)


def filter_matches(src, dst, vis, conf, conf_thr, vis_thr, max_corr):
    mask = (vis > vis_thr) & (conf > conf_thr)
    if mask.sum() < 8:
        k = min(len(conf), max(8, len(conf) // 4))
        top = torch.topk(vis * conf, k=k).indices
        mask = torch.zeros_like(mask)
        mask[top] = True
    idx = torch.where(mask)[0]
    if len(idx) > max_corr:
        idx = idx[torch.topk((vis * conf)[idx], k=max_corr).indices]
    return src[idx], dst[idx]


def eval_navi(args, tracker: VGGTTracker):
    from omegaconf import OmegaConf

    # NOTE: `configs/navi_correspondence.yaml` contains Hydra defaults and does not
    # materialize `dataset` via plain OmegaConf.load. Load the dataset config directly.
    dataset_cfg = OmegaConf.load("configs/dataset/navi.yaml")
    loader = build_loader(dataset_cfg, "test", 1, 1, pair_dataset=True)
    n = len(loader.dataset) if args.max_items < 0 else min(len(loader.dataset), args.max_items)

    err_3d, err_2d, rel_rots = [], [], []
    vis_saved = 0
    for i in tqdm(range(n), desc="navi"):
        b = loader.dataset.__getitem__(i)
        img0, img1 = b["image_0"], b["image_1"]
        valid0 = valid_mask_from_xyz_grid(b["xyz_grid_0"])
        q = sample_random_queries_from_valid_mask(valid0, num_points=args.query_num, seed=args.seed + i)
        src, dst, vis, conf = tracker.track_pair(img0, img1, q)
        src, dst = filter_matches(src, dst, vis, conf, args.conf_thr, args.vis_thr, args.num_corr)

        valid1 = valid_mask_from_xyz_grid(b["xyz_grid_1"])
        v1 = sample_map_at_uv(valid1.float().unsqueeze(0), dst).squeeze(-1) > 0.5
        if v1.sum() < 8:
            continue
        src = src[v1]
        dst = dst[v1]

        xyz0 = sample_map_at_uv(b["xyz_grid_0"], src)
        xyz1 = sample_map_at_uv(b["xyz_grid_1"], dst)
        Rt = b["Rt_01"].float()[:3, :4]
        K = b["intrinsics_1"].float()

        xyz0in1 = transform_points_Rt(xyz0, Rt)
        err_3d.append((xyz0in1 - xyz1).norm(p=2, dim=1))
        uv0 = project_3dto2d(xyz0in1, K)
        uv1 = project_3dto2d(xyz1, K)
        err_2d.append((uv0 - uv1).norm(p=2, dim=1))
        rel_rots.append(Rt[:3, :3])

        if args.save_correspondence_vis and vis_saved < args.vis_max_pairs:
            save_pair_correspondence_vis(
                img0, img1, src, dst,
                Path(args.vis_dir) / "navi" / f"pair_{i:06d}.png",
                seed=args.seed + i,
            )
            vis_saved += 1

    if len(err_3d) == 0 or len(err_2d) == 0:
        bins = [float("nan")] * 4
        return {"task": "navi", "metrics": [float("nan")] * 6 + bins}

    err_3d = torch.cat(err_3d)
    err_2d = torch.cat(err_2d)
    results = [
        100 * (err_3d < 0.01).float().mean().item(),
        100 * (err_3d < 0.02).float().mean().item(),
        100 * (err_3d < 0.05).float().mean().item(),
        100 * (err_2d < 5).float().mean().item(),
        100 * (err_2d < 25).float().mean().item(),
        100 * (err_2d < 50).float().mean().item(),
    ]
    # Keep output shape compatible with previous logging by appending 4 bin placeholders.
    bins = [float("nan")] * 4
    return {"task": "navi", "metrics": results + bins}


def eval_scannet_corr(args, tracker: VGGTTracker):
    dataset = ScanNetPairsDataset()
    n = len(dataset) if args.max_items < 0 else min(len(dataset), args.max_items)
    errs = []
    vis_saved = 0
    for i in tqdm(range(n), desc="scannet_corr"):
        ins = dataset.__getitem__(i)
        img0, img1 = ins["rgb_0"], ins["rgb_1"]
        valid0 = valid_mask_from_depth(ins["depth_0"])
        q = sample_random_queries_from_valid_mask(valid0, num_points=args.query_num, seed=args.seed + i)
        src, dst, vis, conf = tracker.track_pair(img0, img1, q)
        src, dst = filter_matches(src, dst, vis, conf, args.conf_thr, args.vis_thr, args.num_corr)

        valid1 = valid_mask_from_depth(ins["depth_1"])
        v1 = sample_map_at_uv(valid1.float().unsqueeze(0), dst).squeeze(-1) > 0.5
        if v1.sum() < 8:
            continue
        src = src[v1]
        dst = dst[v1]

        K = ins["K"].float()
        Rt = ins["Rt_1"].float()[:3, :4]
        xyz0 = uv_depth_to_xyz(src, ins["depth_0"], K)
        xyz1 = uv_depth_to_xyz(dst, ins["depth_1"], K)
        xyz0in1 = transform_points_Rt(xyz0, Rt)
        uv0in1 = project_3dto2d(xyz0in1, K)
        uv1 = project_3dto2d(xyz1, K)
        errs.append((uv0in1 - uv1).norm(p=2, dim=1))

        if args.save_correspondence_vis and vis_saved < args.vis_max_pairs:
            save_pair_correspondence_vis(
                img0, img1, src, dst,
                Path(args.vis_dir) / "scannet_correspondence" / f"pair_{i:06d}.png",
                seed=args.seed + i,
            )
            vis_saved += 1
    if len(errs) == 0:
        return {"task": "scannet_correspondence", "metrics": [float("nan")] * 3}
    err = torch.cat(errs)
    return {"task": "scannet_correspondence", "metrics": [100 * (err < t).float().mean().item() for t in [5, 25, 50]]}


def pose_errors(uv0, uv1, K, Rt_gt):
    pts0 = uv0.numpy().astype(np.float32)
    pts1 = uv1.numpy().astype(np.float32)
    E, m = cv2.findEssentialMat(pts0, pts1, K.numpy(), cv2.RANSAC, 0.999, 1.0)
    if E is None:
        return None
    ok, R, t, _ = cv2.recoverPose(E, pts0, pts1, K.numpy(), mask=m)
    if ok <= 0:
        return None
    R_gt = Rt_gt[:3, :3].numpy()
    t_gt = Rt_gt[:3, 3].numpy()
    tr = np.clip((np.trace(R.T @ R_gt) - 1) / 2, -1, 1)
    r_err = np.rad2deg(np.arccos(tr))
    t = t.reshape(-1)
    t_err = np.rad2deg(np.arccos(np.clip(np.dot(t / np.linalg.norm(t), t_gt / np.linalg.norm(t_gt)), -1, 1)))
    return r_err, t_err


def eval_scannet_pose(args, tracker: VGGTTracker):
    dataset = ScanNetPairsDataset()
    n = len(dataset) if args.max_items < 0 else min(len(dataset), args.max_items)
    r_errs, t_errs = [], []
    for i in tqdm(range(n), desc="scannet_pose"):
        ins = dataset.__getitem__(i)
        img0, img1 = ins["rgb_0"], ins["rgb_1"]
        q = make_grid_queries(img0.shape[-2], img0.shape[-1], step=args.grid_step)
        src, dst, vis, conf = tracker.track_pair(img0, img1, q)
        src, dst = filter_matches(src, dst, vis, conf, args.conf_thr, args.vis_thr, args.num_corr)
        pe = pose_errors(src, dst, ins["K"].float(), ins["Rt_1"].float())
        if pe is not None:
            r_errs.append(pe[0])
            t_errs.append(pe[1])
    re = np.array(r_errs)
    te = np.array(t_errs)
    return {"task": "scannet_pose", "metrics": [re.mean(), (re < 5).mean() * 100, (re < 10).mean() * 100, (re < 15).mean() * 100, te.mean()]}


def eval_onepose_pose(args, tracker: VGGTTracker):
    root = Path("./data/lowtexture_test_data")
    sfm_dir = Path("./data/outputs_softmax_loftr_loft")
    obj_names = sorted([p.name for p in root.iterdir() if p.is_dir()])
    if args.max_items > 0:
        obj_names = obj_names[:args.max_items]
    thr1, thr3, thr5 = [], [], []

    for obj in tqdm(obj_names, desc="onepose_pose"):
        seq = obj.split("-")[1]
        anno_path = root / obj / f"{seq}-1" / "anno_loftr"
        json_files = sorted(anno_path.glob("*.json"))
        if not json_files:
            continue
        anno = json.load(open(json_files[0]))
        idx = json_files[0].stem
        keypoints2d = np.array(anno["keypoints2d"], dtype=np.float32)
        assign = np.array(anno["assign_matrix"])
        k2d = keypoints2d[assign[0]]
        kp3d = np.load(sfm_dir / obj / "anno" / "anno_3d_average.npz")["keypoints3d"][assign[1]]

        t_img = cv2.imread(str(root / obj / f"{seq}-1" / "color" / f"{idx}.png"))[..., ::-1].copy()
        t_img = torch.from_numpy((t_img / 255.0).astype(np.float32)).permute(2, 0, 1)
        query = torch.from_numpy(k2d)

        R_errs, T_errs = [], []
        test_seq = "2"
        all_img = sorted((root / obj / f"{seq}-{test_seq}" / "color").glob("*.png"))
        for img_fn in all_img:
            test = cv2.imread(str(img_fn))[..., ::-1].copy()
            test_t = torch.from_numpy((test / 255.0).astype(np.float32)).permute(2, 0, 1)
            src, dst, vis, conf = tracker.track_pair(t_img, test_t, query)
            mask = ((vis > args.vis_thr) & (conf > args.conf_thr)).numpy()
            if mask.sum() < 6:
                continue
            ok, rvec, tvec, inl = cv2.solvePnPRansac(kp3d[mask].astype(np.float32), dst[mask].numpy().astype(np.float32),
                                                     np.loadtxt(root / obj / f"{seq}-{test_seq}" / "intrin_ba" / f"{img_fn.stem}.txt"), None)
            if not ok:
                continue
            R, _ = cv2.Rodrigues(rvec)
            pose_pred = np.eye(4, dtype=np.float32)
            pose_pred[:3, :3] = R
            pose_pred[:3, 3] = tvec[:, 0]
            pose_gt = np.loadtxt(root / obj / f"{seq}-{test_seq}" / "poses_ba" / f"{img_fn.stem}.txt")
            t_dist = np.linalg.norm(pose_pred[:3, 3] - pose_gt[:3, 3]) * 100
            r_tr = np.clip((np.trace(pose_pred[:3, :3] @ pose_gt[:3, :3].T) - 1) / 2, -1, 1)
            r_dist = np.rad2deg(np.arccos(r_tr))
            R_errs.append(r_dist)
            T_errs.append(t_dist)

        if len(R_errs) == 0:
            continue
        R_errs = np.array(R_errs); T_errs = np.array(T_errs)
        thr1.append(np.mean((R_errs < 1) & (T_errs < 1)))
        thr3.append(np.mean((R_errs < 3) & (T_errs < 3)))
        thr5.append(np.mean((R_errs < 5) & (T_errs < 5)))

    return {"task": "onepose_pose", "metrics": [float(np.mean(thr1)), float(np.mean(thr3)), float(np.mean(thr5))]}


def eval_pascal(args, tracker: VGGTTracker):
    categories = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
    if args.max_items > 0:
        categories = categories[:args.max_items]
    p05, p10, p15 = [], [], []
    img_size = 700
    for cat in tqdm(categories, desc="pascal_pf"):
        files, kps, _ = load_pascal_data('./data/PF-dataset-PASCAL', size=img_size, category=cat, same_view=True)
        gts, preds = [], []
        for pair_idx in range(len(files) // 2):
            img1 = torch.from_numpy(np.array(resize(Image.open(files[2*pair_idx]).convert('RGB'), img_size, resize=True, to_pil=True, edge=False)) / 255.).float().permute(2,0,1)
            img2 = torch.from_numpy(np.array(resize(Image.open(files[2*pair_idx+1]).convert('RGB'), img_size, resize=True, to_pil=True, edge=False)) / 255.).float().permute(2,0,1)
            q = kps[2*pair_idx][:, :2].float()
            src, dst, vis, conf = tracker.track_pair(img1, img2, q)
            vismask = (kps[2*pair_idx][:,2] * kps[2*pair_idx+1][:,2] > 0)
            gts.append(kps[2*pair_idx+1][vismask][:, [1,0]])
            preds.append(dst[vismask][:, [1,0]])
        gt = torch.cat(gts, 0)
        pd = torch.cat(preds, 0)
        err = (pd - gt).norm(dim=-1)
        p05.append((err < 0.05 * img_size).float().mean().item())
        p10.append((err < 0.10 * img_size).float().mean().item())
        p15.append((err < 0.15 * img_size).float().mean().item())
    return {"task": "pascal_pf", "metrics": [float(np.mean(p05)), float(np.mean(p10)), float(np.mean(p15))]}


def eval_tapvid(args, tracker: VGGTTracker):
    ds = TapVidDataset(patch_size=14, stride=7, resize_hw=(400, 712))
    n = len(ds) if args.video_num <= 0 else min(args.video_num, len(ds))
    metrics = []
    for i in tqdm(range(n), desc="tapvid"):
        ins = ds.__getitem__(i)
        rgbs = ins["rgbs"]  # [T,3,H,W]
        vcfg = ins["video_config"]
        traj_dict, occ_dict = {}, {}
        rx = rgbs.shape[-1] / vcfg["w"]
        ry = rgbs.shape[-2] / vcfg["h"]
        for frame_idx, q_at_frame in vcfg["query_points"].items():
            if len(q_at_frame) == 0:
                continue
            q = torch.tensor([[p[0] * rx, p[1] * ry] for p in q_at_frame], dtype=torch.float32)
            # run tracking from query frame to all frames by rolling sequence
            seq = rgbs
            src, dst_all, vis, conf = tracker.track_pair(seq[int(frame_idx)], seq[-1], q)  # fallback 2-frame mode
            traj = np.repeat(dst_all.numpy()[None], repeats=rgbs.shape[0], axis=0)
            occ = np.repeat((1 - (vis.numpy() > args.vis_thr)).astype(np.float32)[None], repeats=rgbs.shape[0], axis=0)
            traj_dict[frame_idx] = traj
            occ_dict[frame_idx] = occ
        m = compute_tapvid_metrics_for_video(traj_dict, occ_dict, vcfg["video_idx"], {"videos": [vcfg]}, [rgbs.shape[-1], rgbs.shape[-2]])
        m["video_idx"] = vcfg["video_idx"]
        metrics.append(m)
    df = pd.DataFrame(metrics).set_index("video_idx")
    return {"task": "tapvid", "metrics": list(df.mean().values)}


def format_metrics(metrics: Iterable[float]) -> str:
    return ", ".join([f"{m:.4f}" for m in metrics])


def main():
    args = parse_args()
    tasks = ["navi", "scannet_corr", "scannet_pose", "onepose_pose", "pascal_pf", "tapvid"] if "all" in args.tasks else args.tasks

    tracker = VGGTTracker(device=args.device)
    runners = {
        "navi": eval_navi,
        "scannet_corr": eval_scannet_corr,
        "scannet_pose": eval_scannet_pose,
        "onepose_pose": eval_onepose_pose,
        "pascal_pf": eval_pascal,
        "tapvid": eval_tapvid,
    }

    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for t in tasks:
        out = runners[t](args, tracker)
        line = f"{stamp}, VGGT-1B-track-module, {out['task']}, {format_metrics(out['metrics'])}\n"
        with open(log_path, "a") as f:
            f.write(line)
        print(line.strip())


if __name__ == "__main__":
    main()
