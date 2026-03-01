#!/usr/bin/env python3
"""Run all baseline evaluation tasks for the original VGGT setting.

This script mirrors `scripts/eval_all_baseline.sh` but provides a Python entrypoint,
so you can run all tasks in one command and keep a shared log file.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate original VGGT on all baseline tasks."
    )
    parser.add_argument(
        "--backbone",
        default="fit3d",
        help=(
            "Hydra backbone config name passed to each evaluate script "
            "(default: fit3d)."
        ),
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help=(
            "Shared log file path. If omitted, uses ./logs/{backbone}_vggt_original.log"
        ),
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
    if not continue_on_error:
        return False
    return True


def main() -> int:
    args = parse_args()

    repo_root = Path(__file__).resolve().parent
    log_file = (
        Path(args.log_file)
        if args.log_file is not None
        else repo_root / "logs" / f"{args.backbone}_vggt_original.log"
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)

    eval_scripts = [
        "evaluate_navi_correspondence.py",
        "evaluate_scannet_correspondence.py",
        "evaluate_scannet_pose.py",
        "evaluate_onepose_pose.py",
        "evaluate_pascal_pf.py",
        "evaluate_tapvid_video.py",
    ]

    print("[INFO] Running VGGT baseline evaluations")
    print(f"[INFO] Backbone: {args.backbone}")
    print(f"[INFO] Log file: {log_file}")

    for script in eval_scripts:
        cmd = [
            args.python,
            str(repo_root / script),
            f"backbone={args.backbone}",
            f"log_file={log_file}",
        ]
        ok = run_command(cmd, args.continue_on_error)
        if not ok:
            return 1

    print("\n[INFO] All requested evaluations finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
