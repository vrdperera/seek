from __future__ import annotations

import csv
import random
from pathlib import Path


def load_reference_splits(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    reference_path = Path(path)
    splits: dict[str, str] = {}
    with reference_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            product_id = str(row["product_id"])
            split = str(row["split"]).lower()
            existing = splits.setdefault(product_id, split)
            if existing != split:
                raise ValueError(
                    f"Reference product '{product_id}' occurs in multiple splits"
                )
    return splits


def select_products(
    available_product_ids: list[str],
    max_products: int,
    seed: int,
    reference_splits: dict[str, str],
) -> tuple[list[str], dict[str, str]]:
    if max_products <= 0:
        raise ValueError("max_products must be greater than zero")
    available = set(available_product_ids)
    missing = sorted(set(reference_splits) - available)
    if missing:
        raise ValueError(f"Reference products are unavailable: {missing[:5]}")
    if len(reference_splits) > max_products:
        raise ValueError(
            "max_products cannot be smaller than the number of reference identities"
        )

    shuffled = sorted(available_product_ids)
    random.Random(seed).shuffle(shuffled)
    if reference_splits:
        selected = list(reference_splits)
        selected.extend(
            product_id
            for product_id in shuffled
            if product_id not in reference_splits
        )
        selected = selected[:max_products]
        assignments = {
            product_id: reference_splits.get(product_id, "train")
            for product_id in selected
        }
        return selected, assignments

    selected = shuffled[:max_products]
    train_end = int(len(selected) * 0.7)
    val_end = train_end + int(len(selected) * 0.15)
    assignments = {
        product_id: (
            "train" if index < train_end else "val" if index < val_end else "test"
        )
        for index, product_id in enumerate(selected)
    }
    return selected, assignments
