from datetime import datetime
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nn_F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel, DataParallel
from torch.utils.data import DataLoader, DistributedSampler
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import random
import warnings
import os
import os.path as osp
import time
import math

from evals.datasets.builder import build_dataloader
from evals.models.modules.linear_head import BNHead
from evals.models.modules.depther import DepthEncoderDecoder

from evals.datasets.mmcv import get_root_logger
from evals.datasets.nyuv2 import NYUDataset
from evals.datasets.nyuv2_utils import LoadImageFromFile, DepthLoadAnnotations, NYUCrop, RandomCrop, RandomFlip, Resize, ImageToTensor
from evals.datasets.nyuv2_utils import RandomRotate, ColorAug, Normalize, Compose, resize, DefaultFormatBundle, Collect, MultiScaleFlipAug
from evals.utils.depth import metrics, pre_eval_to_metrics

device = 'cuda:0'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
dataset_type = "NYUDataset"
data_root = "/data/fanry/Desktop/fmft/probe3d/data/NYUv2/sync"
crop_size = (480, 640)
train_pipeline = [
    LoadImageFromFile(),
    DepthLoadAnnotations(),
    NYUCrop(depth=True),
    RandomRotate(prob=0.5, degree=2.5),
    RandomFlip(prob=0.5),
    RandomCrop(crop_size=crop_size),
    ColorAug(
        prob=0.5,
        gamma_range=[0.9, 1.1],
        brightness_range=[0.75, 1.25],
        color_range=[0.9, 1.1],
    ),
    Normalize(**img_norm_cfg),
    DefaultFormatBundle(),
    Collect(
        keys=["img", "depth_gt"],
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
            "cam_intrinsic",
        ),
    ),
]
test_pipeline = [
    LoadImageFromFile(),
    MultiScaleFlipAug(
        img_scale=(480, 640),
        img_ratios=1.0,
        flip=True,
        flip_direction="horizontal",
        transforms=[
            Resize(keep_ratio=True),
            RandomFlip(direction="horizontal"),
            Normalize(**img_norm_cfg),
            ImageToTensor(keys=["img"]),
            Collect(
                keys=["img"],
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
                    "cam_intrinsic",
                ),
            ),
        ],
    ),
]

train=dict(
    data_root=data_root,
    depth_scale=1000,
    split="/data/fanry/Desktop/fmft/probe3d/data/NYUv2/nyu_train.txt",
    pipeline=train_pipeline,
    garg_crop=False,
    eigen_crop=True,
    min_depth=0.001,
    max_depth=10,
)

test=dict(
    data_root=data_root,
    depth_scale=1000,
    split="/data/fanry/Desktop/fmft/probe3d/data/NYUv2/nyu_test.txt",
    pipeline=test_pipeline,
    garg_crop=False,
    eigen_crop=True,
    min_depth=0.001,
    max_depth=10,
)

def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_warmup_cosine_scheduler(optimizer, base_lr, total_iters, warmup_iters, min_lr_ratio):
    min_lr = base_lr * min_lr_ratio

    def lr_lambda(current_step):
        if current_step < warmup_iters:
            return float(current_step) / float(max(1, warmup_iters))
        else:
            cosine_iters = total_iters - warmup_iters
            progress = (current_step - warmup_iters) / float(max(1, cosine_iters))
            return max(min_lr / base_lr, 0.5 * (1 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

def single_gpu_test(model, data_loader, device, pre_eval=False, format_only=False):
    assert [pre_eval, format_only].count(True) <= 1, \
        "pre_eval and format_only are mutually exclusive"
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = tqdm(total=len(data_loader.dataset), desc="Evaluating")
    loader_indices = data_loader.batch_sampler
    for batch_indices, data in zip(loader_indices, data_loader):
        # indices = getattr(batch, "batch_indices", None)
        img = data["img"]
        img_metas = data["img_metas"]
        with torch.no_grad():
            result_depth = model(img, img_metas, return_loss=False, rescale=True, size=(480, 640))
        if format_only:
            result = data_loader.dataset.format_results(output, indices=indices)
        elif pre_eval:
            result, result_depth = dataset.pre_eval(result_depth, indices=batch_indices)
        else:
            result = list(output.squeeze(1).cpu().numpy())
        results.extend(result)
        prog_bar.update(len(img))
    prog_bar.close()
    return results

def train_depther(model, dataset, cfg, distributed=False, validate=False):
    # setup_distributed_training()
    logger = get_root_logger(cfg.log_level)
    model = model.to(device)
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        model = model.to(local_rank)
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    else:
        model = DataParallel(model, device_ids=cfg.gpu_ids)
    # prepare dataloaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(ds, 1, 2, len(cfg.gpu_ids), dist=distributed, seed=cfg.random_seed, drop_last=True)
        for ds in dataset
    ]
    # validation dataloader
    if validate:
        val_dataset = NYUDataset(**test)
        val_loader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=2,
            dist=distributed,
            shuffle=False,
            seed=cfg.random_seed
        )
    else:
        val_loader = None
    optimizer = torch.optim.AdamW(model.module.decode_head.parameters(), lr=0.0001, weight_decay=0.01, betas=(0.9, 0.999))
    scheduler = get_warmup_cosine_scheduler(optimizer, 0.0001, cfg.total_iters, 12800, 1e-08)

    max_iters = cfg.total_iters
    log_interval = cfg.log_interval
    eval_interval = cfg.interval
    # start training
    model.train()
    iter_count = 0
    best_abs_rel = 1000
    while iter_count < max_iters:
        for data in data_loaders[0]:
            inputs = data['img'].to(device)
            targets = data['depth_gt'].to(device)
            img_metas = data['img_metas']
            optimizer.zero_grad()
            loss = model(inputs, img_metas, return_loss=True, depth_gt=targets)
            loss_tensor = sum(v for k, v in loss.items() if isinstance(v, torch.Tensor) and v.requires_grad)
            loss_tensor.backward()
            params = getattr(model, "module", model).decode_head.parameters()
            torch.nn.utils.clip_grad_norm_(params, max_norm=35, norm_type=2)
            # torch.nn.utils.clip_grad_norm_(model.module.decode_head.parameters(), max_norm=35, norm_type=2)
            optimizer.step()
            scheduler.step()
            if iter_count % log_interval == 0:
                logger.info(f"Iter [{iter_count}/{max_iters}] Loss: {loss_tensor.item():.4f}")

            if validate and (iter_count > 0 and iter_count % cfg.interval == 0):
                pre_results = single_gpu_test(model, val_loader, device, pre_eval=True)
                metrics_dict = pre_eval_to_metrics(pre_results)
                logger.info(f"Validation metrics: {metrics_dict}")
                if metrics_dict["abs_rel"] < best_abs_rel:
                    best_abs_rel = metrics_dict["abs_rel"]
                    torch.save(model.state_dict(), '/home/fanry/Desktop/fmft/probe3d_liyh/probeRes/workdir/depth/nyu/best_model.pth')
                    logger.info(f"Saved best model at iter {iter_count}, abs_rel={best_abs_rel:.4f}")
                model.train()
            iter_count += 1
            if iter_count >= max_iters:
                break
    logger.info("Training finished. Start Testing. ")
    pre_results = single_gpu_test(model, val_loader, device, pre_eval=True)
    metrics_dict = pre_eval_to_metrics(pre_results)
    logger.info(f"Test metrics: {metrics_dict}")

@hydra.main("./configs", "nyu_depth", None)
def main(cfg: DictConfig):
    print(f"Config: \n {OmegaConf.to_yaml(cfg)}")
    # local_rank = int(os.environ.get("LOCAL_RANK", 0))
    # torch.cuda.set_device(local_rank)
    # device = torch.device(f"cuda:{local_rank}")
    # if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
    #     distributed = True
    #     dist.init_process_group(backend="nccl", init_method="env://")
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_dir = "/home/fanry/Desktop/fmft/probe3d_liyh/probeRes/workdir/depth/nyu"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    set_random_seed(cfg.random_seed)
    # ===== Get model and dataset ====
    model = instantiate(cfg.backbone, return_multilayer=cfg.multilayer, output="dense")
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    type_to_channels = {
        "clip_b16": 768,              
        "clip_convnext": 768,         
        "clip_convnext_augreg": 768,   
        "clip_114": 768,               
        "clip_b16_laion": 768,        
        "siglip_b16": 768,             
        "siglip_116": 768,             

        "dinov2_vits14": 384,          
        "dinov2_vitb14": 768,          
        "dinov2_vitl14": 1024,         
        "dinov2_vitg14": 1536,        
        "dinov2_b14": 768,             
        "dinov2_b14_reg": 768,         
        "dinov2_114": 1024,            
        "dinov2_s14": 384,             

        "ibot_b16": 768,               
        "ibot_b16_in22k": 768,         
        "ibot_116": 768,               
        "ibot_116_in22k": 768,         

        "mae_b16": 768,                
        "mae_116": 768,                

        "deit3_b16": 768,              
        "deit3_116": 768,              

        "convnext_in22k": 768,         
        "convnext_fcmae": 768,         

        "diff": 768,                   
        "dino_b16": 768,               
        "radio": 768,                  
        "midas_116": 256,              
        "sam_base": 768,               
        "sam_large": 1024,            
    }
    channels = type_to_channels[cfg.model_type]
    head = BNHead(
        min_depth=0.001,
        max_depth=10,
        loss_decode=[
            dict(type="SigLoss", valid_mask=True, loss_weight=1.0, warm_up=True, loss_name="loss_depth"),
            dict(type="GradientLoss", valid_mask=True, loss_weight=0.5, loss_name="loss_grad"),
        ],
        classify=True,
        n_bins=256,
        bins_strategy="UD",
        norm_strategy="linear",
        upsample=4,
        in_channels=[channels],
        in_index=[0],
        input_transform="resize_concat",
        # channels=channels + type_to_channels[cfg.model_type],
        channels=channels,
        align_corners=False,
        true_channels=type_to_channels[cfg.model_type]
    ).to(device)
    datasets = [NYUDataset(**train)]
    depther = DepthEncoderDecoder(
        backbone=model,
        decode_head=head,
        train_cfg=dict(),
        test_cfg=dict(mode="whole"),
    ).to(device)
    train_depther(depther, datasets, cfg, distributed=False, validate=True)

if __name__ == "__main__":
    main()