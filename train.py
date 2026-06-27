#!/usr/bin/env python3
"""Two-stage YOLOv8 training for the panorama face dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "pano_face_detection" / "dataset.yaml"
DEFAULT_WEIGHTS = ROOT / "checkpoints" / "yolov8s.pt"
DEFAULT_PROJECT = ROOT / "runs" / "pano_face"


def parse_cache(value: str) -> bool | str:
    normalized = str(value).lower()
    if normalized in {"false", "0", "none", "no"}:
        return False
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"ram", "disk"}:
        return normalized
    raise argparse.ArgumentTypeError("cache must be one of: false, true, ram, disk")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train YOLOv8s on pano_face_detection with a frozen warmup stage "
            "followed by full fine-tuning."
        )
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Dataset YAML path.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Initial YOLOv8 weights.")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT, help="Output project directory.")
    parser.add_argument("--name-prefix", default="", help="Optional prefix for experiment names.")
    parser.add_argument(
        "--stage",
        choices=("all", "stage1", "stage2"),
        default="all",
        help="Which training stage to run.",
    )
    parser.add_argument("--imgsz", type=int, default=960, help="Training image size.")
    parser.add_argument("--batch", type=int, default=-1, help="Batch size. -1 enables Ultralytics AutoBatch.")
    parser.add_argument("--device", default=None, help="CUDA device, e.g. 0 or 0,1. Defaults to auto.")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers.")
    parser.add_argument("--seed", type=int, default=20260624, help="Random seed.")
    parser.add_argument("--cache", type=parse_cache, default=False, help="Dataset cache setting: false, true, ram, or disk.")
    parser.add_argument("--stage1-epochs", type=int, default=20, help="Frozen warmup epochs.")
    parser.add_argument("--stage2-epochs", type=int, default=150, help="Full fine-tuning epochs.")
    parser.add_argument("--stage1-freeze", type=int, default=10, help="Number of early layers to freeze.")
    parser.add_argument("--stage1-lr0", type=float, default=0.003, help="Initial LR for frozen warmup.")
    parser.add_argument("--stage2-lr0", type=float, default=0.001, help="Initial LR for full fine-tuning.")
    parser.add_argument("--stage1-patience", type=int, default=20, help="Early stopping patience for stage 1.")
    parser.add_argument("--stage2-patience", type=int, default=40, help="Early stopping patience for stage 2.")
    parser.add_argument("--close-mosaic", type=int, default=20, help="Disable mosaic for final N epochs in stage 2.")
    parser.add_argument(
        "--stage1-weights",
        type=Path,
        default=None,
        help="Stage 1 weights to use when running --stage stage2. Defaults to stage1 best.pt.",
    )
    parser.add_argument("--exist-ok", action="store_true", default=True, help="Allow overwriting run directories.")
    parser.add_argument("--no-exist-ok", action="store_false", dest="exist_ok", help="Create unique run dirs.")
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def run_train(model_path: Path, train_args: dict[str, Any]) -> Path:
    require_file(model_path, "Model weights")
    model = YOLO(str(model_path))
    model.train(**train_args)

    run_dir = Path(train_args["project"]) / train_args["name"]
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    if best.is_file():
        return best
    if last.is_file():
        return last
    raise FileNotFoundError(f"No checkpoint was produced under {run_dir / 'weights'}")


def common_args(args: argparse.Namespace) -> dict[str, Any]:
    train_args: dict[str, Any] = {
        "data": str(args.data),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": str(args.project),
        "exist_ok": args.exist_ok,
        "seed": args.seed,
        "deterministic": True,
        "task": "detect",
        "single_cls": False,
        "amp": True,
        "plots": True,
        "cache": args.cache,
    }
    if args.device is not None:
        train_args["device"] = args.device
    return train_args


def prefixed(args: argparse.Namespace, name: str) -> str:
    return f"{args.name_prefix}{name}" if args.name_prefix else name


def main() -> None:
    args = parse_args()
    require_file(args.data, "Dataset YAML")

    base = common_args(args)
    stage1_name = prefixed(args, "stage1_freeze")
    stage2_name = prefixed(args, "stage2_full")

    stage1_best = args.project / stage1_name / "weights" / "best.pt"

    if args.stage in {"all", "stage1"}:
        stage1_args = {
            **base,
            "name": stage1_name,
            "epochs": args.stage1_epochs,
            "freeze": args.stage1_freeze,
            "lr0": args.stage1_lr0,
            "patience": args.stage1_patience,
            "close_mosaic": 0,
        }
        print(f"Stage 1: frozen warmup from {args.weights}")
        stage1_best = run_train(args.weights, stage1_args)
        print(f"Stage 1 checkpoint: {stage1_best}")

    if args.stage in {"all", "stage2"}:
        start_weights = args.stage1_weights or stage1_best
        stage2_args = {
            **base,
            "name": stage2_name,
            "epochs": args.stage2_epochs,
            "freeze": 0,
            "lr0": args.stage2_lr0,
            "patience": args.stage2_patience,
            "close_mosaic": args.close_mosaic,
        }
        print(f"Stage 2: full fine-tuning from {start_weights}")
        stage2_best = run_train(start_weights, stage2_args)
        print(f"Stage 2 checkpoint: {stage2_best}")


if __name__ == "__main__":
    main()
