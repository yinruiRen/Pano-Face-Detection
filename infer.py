#!/usr/bin/env python3
"""Example: run face detection on one image."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "runs" / "infer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLOv8 face detection on a single image.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to the trained model, e.g. best.pt.")
    parser.add_argument("--image", type=Path, required=True, help="Path to the input image.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for the annotated result.")
    parser.add_argument("--imgsz", type=int, default=960, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--device", default=None, help="CUDA device, e.g. 0. Defaults to auto.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(f"Model weights not found: {args.weights}")
    if not args.image.is_file():
        raise FileNotFoundError(f"Input image not found: {args.image}")

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    predict_args = {
        "source": str(args.image),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "save": True,
        "project": str(args.output_dir.parent),
        "name": args.output_dir.name,
        "exist_ok": True,
    }
    if args.device is not None:
        predict_args["device"] = args.device

    result = model.predict(**predict_args)[0]
    boxes = result.boxes
    face_count = 0 if boxes is None else len(boxes)

    print(f"Image: {args.image}")
    print(f"Faces detected: {face_count}")
    if boxes is not None:
        for index, box in enumerate(boxes, start=1):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            print(f"{index}: xyxy=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}), conf={confidence:.3f}")
    print(f"Annotated result saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
