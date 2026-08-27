from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from PIL import Image, ImageDraw

from styleseek.paths import SAMPLE_DATA_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a tiny synthetic retrieval dataset")
    parser.add_argument("--output", default=str(SAMPLE_DATA_DIR))
    parser.add_argument("--products", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def garment_image(product: int, domain: str, variant: int, seed: int) -> Image.Image:
    rng = random.Random(seed + product * 101 + variant * 13 + (0 if domain == "shop" else 7))
    size = 256
    background = (245, 245, 245) if domain == "shop" else tuple(rng.randint(90, 210) for _ in range(3))
    image = Image.new("RGB", (size, size), background)
    draw = ImageDraw.Draw(image)

    base = (
        45 + (product * 67) % 180,
        45 + (product * 97) % 180,
        45 + (product * 37) % 180,
    )
    offset_x = rng.randint(-10, 10) if domain == "consumer" else 0
    offset_y = rng.randint(-7, 7) if domain == "consumer" else 0
    body = [(73 + offset_x, 66 + offset_y), (183 + offset_x, 222 + offset_y)]
    draw.rounded_rectangle(body, radius=15, fill=base, outline=(30, 30, 30), width=3)
    draw.polygon(
        [(73 + offset_x, 75 + offset_y), (35 + offset_x, 129 + offset_y), (74 + offset_x, 145 + offset_y)],
        fill=base,
    )
    draw.polygon(
        [(183 + offset_x, 75 + offset_y), (221 + offset_x, 129 + offset_y), (182 + offset_x, 145 + offset_y)],
        fill=base,
    )

    pattern = product % 4
    accent = tuple(255 - channel for channel in base)
    if pattern == 0:
        for y in range(90, 205, 24):
            draw.line((78 + offset_x, y + offset_y, 178 + offset_x, y + offset_y), fill=accent, width=7)
    elif pattern == 1:
        for x in range(90, 176, 22):
            draw.line((x + offset_x, 82 + offset_y, x + offset_x, 215 + offset_y), fill=accent, width=6)
    elif pattern == 2:
        draw.ellipse((105 + offset_x, 125 + offset_y, 151 + offset_x, 171 + offset_y), fill=accent)
    else:
        draw.polygon(
            [(128 + offset_x, 105 + offset_y), (158 + offset_x, 175 + offset_y), (98 + offset_x, 175 + offset_y)],
            fill=accent,
        )
    return image


def main() -> None:
    args = parse_args()
    if args.products < 10:
        raise ValueError("Use at least 10 products so every split has multiple identities")
    root = Path(args.output)
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    rows = []
    train_end = int(args.products * 0.7)
    val_end = train_end + int(args.products * 0.15)
    for product in range(args.products):
        split = "train" if product < train_end else "val" if product < val_end else "test"
        for domain, variants in (("consumer", 2), ("shop", 2)):
            for variant in range(variants):
                filename = f"{product:04d}_{domain}_{variant}.jpg"
                garment_image(product, domain, variant, args.seed).save(images / filename, quality=92)
                rows.append(
                    {
                        "image_path": f"images/{filename}",
                        "product_id": f"P{product:04d}",
                        "domain": domain,
                        "split": split,
                    }
                )
    manifest = root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "product_id", "domain", "split"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Created {len(rows)} images and {manifest}")


if __name__ == "__main__":
    main()
