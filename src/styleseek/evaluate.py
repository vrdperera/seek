from __future__ import annotations

import argparse
import json

from .evaluation import evaluate_model
from .model import FashionEncoder, load_checkpoint
from .paths import DEFAULT_MANIFEST, METRICS_DIR
from .utils import choose_device, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate StyleSeek retrieval")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--checkpoint", help="Omit for the pretrained ResNet18 baseline")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output", default=str(METRICS_DIR / "retrieval.json"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    if args.checkpoint:
        model, payload = load_checkpoint(args.checkpoint, device)
        image_size = int(payload.get("image_size", args.image_size))
        objective = payload.get("objective", "metric-learning")
        model_name = f"{objective}-{payload.get('training_mode', 'unknown')}"
    else:
        model = FashionEncoder(embedding_dim=512, pretrained=True).to(device).eval()
        image_size = args.image_size
        model_name = "imagenet-resnet18-baseline"

    metrics, _ = evaluate_model(
        model,
        args.manifest,
        args.split,
        device,
        image_size,
        args.batch_size,
        args.workers,
    )
    result = {"model": model_name, "split": args.split, "device": str(device), **metrics}
    save_json(result, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
