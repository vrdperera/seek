from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from .data import ImageManifestDataset
from .retrieval import embed_loader, retrieval_metrics


def evaluate_model(
    model,
    manifest: str,
    split: str,
    device: torch.device,
    image_size: int,
    batch_size: int,
    workers: int,
):
    query_dataset = ImageManifestDataset(manifest, split, "consumer", image_size)
    gallery_dataset = ImageManifestDataset(manifest, split, "shop", image_size)
    loader_options = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
    }
    query_loader = DataLoader(query_dataset, shuffle=False, **loader_options)
    gallery_loader = DataLoader(gallery_dataset, shuffle=False, **loader_options)
    query_embeddings, query_ids, query_paths = embed_loader(
        model, query_loader, device, "Embedding queries"
    )
    gallery_embeddings, gallery_ids, gallery_paths = embed_loader(
        model, gallery_loader, device, "Embedding gallery"
    )
    metrics = retrieval_metrics(
        query_embeddings, query_ids, gallery_embeddings, gallery_ids
    )
    return metrics, {
        "query_embeddings": query_embeddings,
        "query_ids": query_ids,
        "query_paths": query_paths,
        "gallery_embeddings": gallery_embeddings,
        "gallery_ids": gallery_ids,
        "gallery_paths": gallery_paths,
    }
