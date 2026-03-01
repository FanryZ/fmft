#!/usr/bin/env python3
"""Run baseline evaluations with features from the pretrained VGGT tracking module.

This script mirrors ``scripts/eval_all_baseline.sh`` but forces every task to use
``VGGTTrackingBackbone`` defined in this file. The backbone extracts dense feature
maps from VGGT's pretrained tracking module (TrackHead feature extractor), without
training or finetuning.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence


class VGGTTrackingBackbone:
    """Hydra-instantiable backbone wrapper around pretrained VGGT tracking features."""

    def __init__(
        self,
        output: str = "dense",
        return_multilayer: bool = False,
        model_name: str = "facebook/VGGT-1B",
        device: str = "cuda",
        autocast_dtype: str = "auto",
    ):
        import torch
        from vggt.models.vggt import VGGT

        if output not in {"dense", "dense-cls"}:
            raise ValueError(f"Unsupported output={output!r}; expected a dense output mode")
        if return_multilayer:
            raise ValueError("VGGTTrackingBackbone does not support return_multilayer=True")

        self.output = output
        self.return_multilayer = return_multilayer
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.model = VGGT.from_pretrained(model_name).to(self.device).eval()
        self.patch_size = self.model.track_head.patch_size

        if autocast_dtype == "auto":
            if self.device.type == "cuda":
                major, _ = torch.cuda.get_device_capability(self.device)
                self.autocast_dtype = torch.bfloat16 if major >= 8 else torch.float16
            else:
                self.autocast_dtype = torch.float32
        elif autocast_dtype == "bfloat16":
            self.autocast_dtype = torch.bfloat16
        elif autocast_dtype == "float16":
            self.autocast_dtype = torch.float16
        else:
            self.autocast_dtype = torch.float32

    def to(self, device: str):
        import torch

        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        return self

    def eval(self):
        self.model.eval()
        return self

    def __call__(self, images):
        return self.forward(images)

    def forward(self, images):
        import torch

        if images.ndim != 4:
            raise ValueError(f"Expected images with shape [B, 3, H, W], got {tuple(images.shape)}")

        images = images.to(self.device)
        seq_images = images.unsqueeze(1)  # [B, S=1, 3, H, W]

        with torch.no_grad():
            use_amp = self.device.type == "cuda"
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=self.autocast_dtype if use_amp else None):
                aggregated_tokens_list, patch_start_idx = self.model.aggregator(seq_images)
            # tracking module feature extractor output: [B, S, C, H', W']
            track_feats = self.model.track_head.feature_extractor(
                aggregated_tokens_list, seq_images, patch_start_idx
            )

        return track_feats[:, 0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline tasks with pretrained VGGT tracking-module features.",
    )
    parser.add_argument(
        "--log-file",
        default="./logs/vggt_tracking_baseline.log",
        help="Shared log file for all tasks.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch evaluation scripts.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running remaining tasks when one task fails.",
    )
    return parser.parse_args()


def run_command(cmd: Sequence[str], continue_on_error: bool) -> bool:
    print("\n>>>", " ".join(cmd), flush=True)
    result = subprocess.run(cmd)
    if result.returncode == 0:
        return True

    print(f"[ERROR] Command failed with exit code {result.returncode}: {' '.join(cmd)}")
    return continue_on_error


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    log_file = Path(args.log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    eval_scripts = [
        "evaluate_navi_correspondence.py",
        "evaluate_scannet_correspondence.py",
        "evaluate_scannet_pose.py",
        "evaluate_onepose_pose.py",
        "evaluate_pascal_pf.py",
        "evaluate_tapvid_video.py",
    ]

    # Hydra overrides that force all eval scripts to use this file's VGGT tracker backbone.
    common_overrides = [
        f"log_file={log_file}",
        "multilayer=False",
        "backbone._target_=evaluate_vggt_tracking_baseline.VGGTTrackingBackbone",
        "backbone.output=dense",
        "backbone.return_multilayer=False",
        "backbone.model_name=facebook/VGGT-1B",
        "backbone.device=cuda",
        "backbone.autocast_dtype=auto",
    ]

    print("[INFO] Running baseline tasks with pretrained VGGT tracking features")
    print(f"[INFO] Log file: {log_file}")

    for script in eval_scripts:
        cmd = [args.python, str(repo_root / script), *common_overrides]
        ok = run_command(cmd, args.continue_on_error)
        if not ok:
            return 1

    print("\n[INFO] All requested evaluations finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
