from __future__ import annotations

import argparse
import csv
import json
import os
import re
from functools import lru_cache
from pathlib import Path

from styleseek.categories import GARMENT_CATEGORIES


CROP_NAME = re.compile(r"^(train|validation)_(.+)_(item\d+)\.jpg$")


@lru_cache(maxsize=4096)
def load_annotation(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add DeepFashion2 categories to an existing retrieval manifest"
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output")
    return parser.parse_args()


def category_for_row(dataset_root: Path, image_path: str) -> str:
    match = CROP_NAME.match(Path(image_path).name)
    if match is None:
        raise ValueError(f"Cannot map crop name to a DeepFashion2 annotation: {image_path}")
    source_split, image_stem, item_key = match.groups()
    annotation_path = dataset_root / source_split / "annos" / f"{image_stem}.json"
    data = load_annotation(annotation_path)
    category_id = int(data[item_key]["category_id"])
    if not 1 <= category_id <= len(GARMENT_CATEGORIES):
        raise ValueError(f"Invalid category_id {category_id} in {annotation_path}")
    return GARMENT_CATEGORIES[category_id - 1]


def main() -> None:
    args = parse_args()
    manifest = Path(args.manifest)
    destination = Path(args.output) if args.output else manifest
    dataset_root = Path(args.dataset_root)

    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if "category" not in fieldnames:
        fieldnames.append("category")
    for row in rows:
        row["category"] = category_for_row(dataset_root, row["image_path"])

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, destination)
    print(f"Added categories to {len(rows)} rows in {destination}")


if __name__ == "__main__":
    main()
