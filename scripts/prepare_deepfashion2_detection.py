from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from PIL import Image

from styleseek.categories import GARMENT_CATEGORIES
from styleseek.paths import INTERIM_DATA_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert DeepFashion2 boxes to YOLO detection data")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", default=str(INTERIM_DATA_DIR / "detection"))
    parser.add_argument("--max-train-images", type=int, default=20000)
    parser.add_argument("--max-val-images", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy source images instead of creating space-saving symbolic links",
    )
    return parser.parse_args()


def annotation_records(dataset_root: Path, split: str):
    annotation_root = dataset_root / split / "annos"
    image_root = dataset_root / split / "image"
    records = []
    for annotation_path in sorted(annotation_root.glob("*.json")):
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        if str(data.get("source", "")).lower() not in {"user", "consumer"}:
            continue
        image_path = image_root / f"{annotation_path.stem}.jpg"
        if not image_path.exists():
            continue
        items = [
            item
            for key, item in data.items()
            if key.startswith("item")
            and isinstance(item, dict)
            and isinstance(item.get("bounding_box"), list)
            and len(item["bounding_box"]) == 4
            and 1 <= int(item.get("category_id", 0)) <= len(GARMENT_CATEGORIES)
        ]
        if items:
            records.append((image_path, items))
    return records


def prepare_split(
    dataset_root: Path,
    output_root: Path,
    source_split: str,
    target_split: str,
    limit: int,
    rng: random.Random,
    copy_images: bool,
) -> int:
    records = annotation_records(dataset_root, source_split)
    rng.shuffle(records)
    records = records[:limit]
    image_output = output_root / "images" / target_split
    label_output = output_root / "labels" / target_split
    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)

    for image_path, items in records:
        destination = image_output / image_path.name
        if not destination.exists():
            if copy_images:
                shutil.copy2(image_path, destination)
            else:
                destination.symlink_to(image_path.resolve())
        with Image.open(image_path) as image:
            image_width, image_height = image.size
        labels = []
        for item in items:
            x1, y1, x2, y2 = (float(value) for value in item["bounding_box"])
            x1 = max(0.0, min(x1, image_width))
            x2 = max(0.0, min(x2, image_width))
            y1 = max(0.0, min(y1, image_height))
            y2 = max(0.0, min(y2, image_height))
            if x2 <= x1 or y2 <= y1:
                continue
            class_id = int(item["category_id"]) - 1
            center_x = ((x1 + x2) / 2.0) / image_width
            center_y = ((y1 + y2) / 2.0) / image_height
            box_width = (x2 - x1) / image_width
            box_height = (y2 - y1) / image_height
            labels.append(
                f"{class_id} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}"
            )
        (label_output / f"{image_path.stem}.txt").write_text(
            "\n".join(labels) + "\n", encoding="utf-8"
        )
    return len(records)


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output).resolve()
    rng = random.Random(args.seed)
    train_count = prepare_split(
        dataset_root,
        output_root,
        "train",
        "train",
        args.max_train_images,
        rng,
        args.copy_images,
    )
    val_count = prepare_split(
        dataset_root,
        output_root,
        "validation",
        "val",
        args.max_val_images,
        rng,
        args.copy_images,
    )
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(GARMENT_CATEGORIES))
    yaml_text = (
        f"path: {output_root}\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "names:\n"
        f"{names}\n"
    )
    yaml_path = output_root / "deepfashion2.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")
    print(f"Prepared {train_count} train and {val_count} validation consumer images")
    print(f"YOLO data configuration: {yaml_path}")


if __name__ == "__main__":
    main()
