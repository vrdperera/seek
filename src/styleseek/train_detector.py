from __future__ import annotations

import argparse
import os

from .paths import DEFAULT_DETECTION_DATA, DETECTOR_RUNS_DIR, ULTRALYTICS_CONFIG_DIR

ULTRALYTICS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_DIR))

from ultralytics import YOLO

from .utils import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a YOLO garment detector")
    parser.add_argument("--data", default=str(DEFAULT_DETECTION_DATA))
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--project", default=str(DETECTOR_RUNS_DIR))
    parser.add_argument("--name", default="deepfashion2_yolo")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    print(f"Detector training device: {device}")
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch_size,
        device=str(device),
        project=args.project,
        name=args.name,
        patience=5,
        plots=True,
    )


if __name__ == "__main__":
    main()
