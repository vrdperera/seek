from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights

from .utils import resolve_image_path

REQUIRED_COLUMNS = {"image_path", "product_id", "domain", "split"}


def load_manifest(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"product_id": str})
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    frame["domain"] = frame["domain"].str.lower()
    frame["split"] = frame["split"].str.lower()
    invalid_domains = set(frame["domain"]) - {"consumer", "shop"}
    if invalid_domains:
        raise ValueError(f"Unsupported domains: {sorted(invalid_domains)}")
    split_counts = frame.groupby("product_id")["split"].nunique()
    leaking_ids = split_counts[split_counts > 1].index.tolist()
    if leaking_ids:
        preview = leaking_ids[:5]
        raise ValueError(
            "Product IDs occur in more than one split, causing evaluation leakage: "
            f"{preview}"
        )
    return frame


def build_transform(train: bool = False, image_size: int = 224):
    normalize = transforms.Normalize(
        mean=ResNet18_Weights.DEFAULT.transforms().mean,
        std=ResNet18_Weights.DEFAULT.transforms().std,
    )
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )


def _read_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


class TripletFashionDataset(Dataset):
    """Samples consumer anchors, matching shop positives, and shop negatives."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str = "train",
        image_size: int = 224,
        train_augmentation: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        frame = load_manifest(manifest_path)
        frame = frame[frame["split"] == split].copy()

        consumers: dict[str, list[str]] = defaultdict(list)
        shops: dict[str, list[str]] = defaultdict(list)
        for row in frame.itertuples(index=False):
            target = consumers if row.domain == "consumer" else shops
            target[str(row.product_id)].append(row.image_path)

        self.product_ids = sorted(set(consumers) & set(shops))
        if len(self.product_ids) < 2:
            raise ValueError(
                f"Split '{split}' needs at least two product IDs with both consumer and shop images"
            )
        self.consumers = consumers
        self.shops = shops
        self.samples = [
            (product_id, path)
            for product_id in self.product_ids
            for path in consumers[product_id]
        ]
        self.transform = build_transform(train_augmentation, image_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        product_id, anchor_rel = self.samples[index]
        positive_rel = random.choice(self.shops[product_id])
        negative_id = random.choice(self.product_ids)
        while negative_id == product_id:
            negative_id = random.choice(self.product_ids)
        negative_rel = random.choice(self.shops[negative_id])

        anchor = self.transform(_read_rgb(resolve_image_path(self.manifest_path, anchor_rel)))
        positive = self.transform(_read_rgb(resolve_image_path(self.manifest_path, positive_rel)))
        negative = self.transform(_read_rgb(resolve_image_path(self.manifest_path, negative_rel)))
        return anchor, positive, negative, product_id


class PairedFashionDataset(Dataset):
    """Samples matched consumer and shop images for in-batch contrastive learning."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str = "train",
        image_size: int = 224,
        train_augmentation: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        frame = load_manifest(manifest_path)
        frame = frame[frame["split"] == split].copy()

        consumers: dict[str, list[str]] = defaultdict(list)
        shops: dict[str, list[str]] = defaultdict(list)
        for row in frame.itertuples(index=False):
            target = consumers if row.domain == "consumer" else shops
            target[str(row.product_id)].append(row.image_path)

        self.product_ids = sorted(set(consumers) & set(shops))
        if len(self.product_ids) < 2:
            raise ValueError(
                f"Split '{split}' needs at least two product IDs with both consumer and shop images"
            )
        self.shops = shops
        self.samples = [
            (product_id, path)
            for product_id in self.product_ids
            for path in consumers[product_id]
        ]
        self.transform = build_transform(train_augmentation, image_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        product_id, query_rel = self.samples[index]
        product_rel = random.choice(self.shops[product_id])
        query = self.transform(_read_rgb(resolve_image_path(self.manifest_path, query_rel)))
        product = self.transform(_read_rgb(resolve_image_path(self.manifest_path, product_rel)))
        return query, product, product_id


class ImageManifestDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        domain: str,
        image_size: int = 224,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        frame = load_manifest(manifest_path)
        self.frame = frame[
            (frame["split"] == split) & (frame["domain"] == domain)
        ].reset_index(drop=True)
        if self.frame.empty:
            raise ValueError(f"No {domain} images found in split '{split}'")
        self.transform = build_transform(False, image_size)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        absolute_path = resolve_image_path(self.manifest_path, row.image_path)
        return self.transform(_read_rgb(absolute_path)), str(row.product_id), str(absolute_path)
