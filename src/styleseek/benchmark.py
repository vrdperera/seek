from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from PIL import Image

from .data import build_transform, load_manifest
from .model import load_checkpoint
from .paths import CATALOGUE_INDEX, DEFAULT_MANIFEST, METRICS_DIR, RETRIEVAL_CHECKPOINT
from .utils import choose_device, resolve_image_path, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark end-to-end retrieval latency")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--checkpoint", default=str(RETRIEVAL_CHECKPOINT))
    parser.add_argument("--catalogue", default=str(CATALOGUE_INDEX))
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--output", default=str(METRICS_DIR / "latency.json"))
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    model, payload = load_checkpoint(args.checkpoint, device)
    catalogue = torch.load(args.catalogue, map_location="cpu", weights_only=False)
    gallery = catalogue["embeddings"].float()
    transform = build_transform(False, int(payload.get("image_size", 224)))
    frame = load_manifest(args.manifest)
    queries = frame[(frame["split"] == "test") & (frame["domain"] == "consumer")]
    if queries.empty:
        raise ValueError("No consumer test images found")

    timings = []
    for run in range(args.runs + 2):
        row = queries.iloc[run % len(queries)]
        image_path = resolve_image_path(args.manifest, row.image_path)
        with Image.open(image_path) as image:
            tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
        started = time.perf_counter()
        with torch.inference_mode():
            query = model(tensor).cpu().float()
        torch.topk((query @ gallery.T).squeeze(0), k=min(5, len(gallery)))
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - started) * 1000
        if run >= 2:
            timings.append(elapsed)

    result = {
        "device": str(device),
        "runs": len(timings),
        "catalogue_size": len(gallery),
        "mean_ms": float(np.mean(timings)),
        "median_ms": float(np.median(timings)),
        "p95_ms": float(np.percentile(timings, 95)),
    }
    save_json(result, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
