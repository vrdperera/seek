from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image

from styleseek.categories import GARMENT_CATEGORIES
from styleseek.paths import DEFAULT_MANIFEST
from styleseek.preparation import load_reference_splits, select_products


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a DeepFashion2 subset into a StyleSeek manifest")
    parser.add_argument("--dataset-root", required=True, help="Folder containing train/validation image and annos folders")
    parser.add_argument("--output", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--max-products", type=int, default=5000)
    parser.add_argument(
        "--reference-manifest",
        help=(
            "Preserve product splits from an existing manifest and assign newly added "
            "products to training"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def collect(root: Path):
    products: dict[str, list[dict[str, str]]] = {}
    for source_split in ("train", "validation"):
        annotations = root / source_split / "annos"
        images = root / source_split / "image"
        if not annotations.exists():
            continue
        for annotation_path in sorted(annotations.glob("*.json")):
            data = json.loads(annotation_path.read_text(encoding="utf-8"))
            for key, item in data.items():
                if not key.startswith("item") or not isinstance(item, dict):
                    continue
                style = int(item.get("style", 0))
                pair_id = item.get("pair_id", data.get("pair_id"))
                if pair_id is None or style <= 0:
                    continue
                source = str(item.get("source", data.get("source", ""))).lower()
                domain = "shop" if source in {"shop", "commercial"} else "consumer"
                category_id = int(item.get("category_id", 0))
                if not 1 <= category_id <= len(GARMENT_CATEGORIES):
                    continue
                product_id = f"{pair_id}_{style}"
                image_path = images / f"{annotation_path.stem}.jpg"
                if image_path.exists():
                    box = item.get("bounding_box")
                    crop_name = f"{source_split}_{annotation_path.stem}_{key}.jpg"
                    products.setdefault(product_id, []).append(
                        {
                            "source_path": str(image_path.resolve()),
                            "crop_name": crop_name,
                            "box": box,
                            "product_id": product_id,
                            "domain": domain,
                            "category": GARMENT_CATEGORIES[category_id - 1],
                        }
                    )
    return {
        key: value
        for key, value in products.items()
        if {row["domain"] for row in value} == {"consumer", "shop"}
    }


def main() -> None:
    args = parse_args()
    destination = Path(args.output)
    crop_root = destination.parent / "images" / "deepfashion2"
    products = collect(Path(args.dataset_root))
    reference_splits = load_reference_splits(args.reference_manifest)
    product_ids, split_assignments = select_products(
        list(products), args.max_products, args.seed, reference_splits
    )
    rows = []
    crop_root.mkdir(parents=True, exist_ok=True)
    for product_id in product_ids:
        split = split_assignments[product_id]
        for row in products[product_id]:
            crop_path = crop_root / row["crop_name"]
            if not crop_path.exists():
                with Image.open(row["source_path"]) as image:
                    image = image.convert("RGB")
                    box = row["box"]
                    if isinstance(box, list) and len(box) == 4:
                        left, top, right, bottom = (int(value) for value in box)
                        image = image.crop((left, top, right, bottom))
                    image.save(crop_path, quality=92)
            rows.append(
                {
                    "image_path": str(crop_path.resolve()),
                    "product_id": product_id,
                    "domain": row["domain"],
                    "split": split,
                    "category": row["category"],
                }
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "product_id", "domain", "split", "category"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} images from {len(product_ids)} product identities to {destination}")


if __name__ == "__main__":
    main()
