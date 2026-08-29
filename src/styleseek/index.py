from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .catalogue import make_portable_paths
from .data import ImageManifestDataset
from .model import load_checkpoint
from .paths import CATALOGUE_INDEX, DEFAULT_MANIFEST, PROJECT_ROOT, RETRIEVAL_CHECKPOINT
from .retrieval import embed_loader
from .utils import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a catalogue embedding index")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--checkpoint", default=str(RETRIEVAL_CHECKPOINT))
    parser.add_argument("--output", default=str(CATALOGUE_INDEX))
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def build_catalogue_index(
    manifest: str | Path,
    checkpoint: str | Path,
    output: str | Path,
    *,
    split: str = "test",
    batch_size: int = 32,
    workers: int = 0,
    device_name: str = "auto",
    portable_root: str | Path = PROJECT_ROOT,
) -> Path:
    device = choose_device(device_name)
    model, payload = load_checkpoint(checkpoint, device)
    image_size = int(payload.get("image_size", 224))
    dataset = ImageManifestDataset(manifest, split, "shop", image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    embeddings, product_ids, paths = embed_loader(
        model, loader, device, "Indexing catalogue"
    )
    categories = (
        dataset.frame["category"].astype(str).tolist()
        if "category" in dataset.frame.columns
        else []
    )
    stored_paths, path_base = make_portable_paths(paths, portable_root)
    stored_checkpoint, checkpoint_path_base = make_portable_paths(
        [checkpoint], portable_root
    )
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embeddings": embeddings,
            "product_ids": product_ids,
            "paths": stored_paths,
            "path_base": path_base,
            "categories": categories,
            "checkpoint": stored_checkpoint[0],
            "checkpoint_path_base": checkpoint_path_base,
            "image_size": image_size,
        },
        destination,
    )
    print(f"Saved {len(stored_paths)} catalogue embeddings to {destination}")
    return destination


def main() -> None:
    args = parse_args()
    build_catalogue_index(
        args.manifest,
        args.checkpoint,
        args.output,
        split=args.split,
        batch_size=args.batch_size,
        workers=args.workers,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
