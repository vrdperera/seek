from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.inference_mode()
def embed_loader(model, loader: DataLoader, device: torch.device, description: str):
    embeddings: list[torch.Tensor] = []
    product_ids: list[str] = []
    paths: list[str] = []
    model.eval()
    for images, batch_ids, batch_paths in tqdm(loader, desc=description, leave=False):
        embeddings.append(model(images.to(device)).cpu())
        product_ids.extend(str(value) for value in batch_ids)
        paths.extend(str(value) for value in batch_paths)
    return torch.cat(embeddings), product_ids, paths


def retrieval_metrics(
    query_embeddings: torch.Tensor,
    query_ids: Iterable[str],
    gallery_embeddings: torch.Tensor,
    gallery_ids: Iterable[str],
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    query_ids = np.asarray(list(query_ids), dtype=str)
    gallery_ids = np.asarray(list(gallery_ids), dtype=str)
    scores = query_embeddings.float() @ gallery_embeddings.float().T
    ranking = torch.argsort(scores, dim=1, descending=True).cpu().numpy()

    recalls = {k: [] for k in ks}
    average_precisions: list[float] = []
    for query_index, product_id in enumerate(query_ids):
        relevant = gallery_ids == product_id
        relevant_count = int(relevant.sum())
        if relevant_count == 0:
            continue
        ranked_relevance = relevant[ranking[query_index]]
        for k in ks:
            recalls[k].append(float(ranked_relevance[:k].any()))
        hit_positions = np.flatnonzero(ranked_relevance)
        precisions = [
            ranked_relevance[: position + 1].sum() / (position + 1)
            for position in hit_positions
        ]
        average_precisions.append(float(np.mean(precisions)))

    if not average_precisions:
        raise ValueError("No query product IDs have a matching gallery product ID")
    result = {f"recall@{k}": float(np.mean(values)) for k, values in recalls.items()}
    result["map"] = float(np.mean(average_precisions))
    result["queries"] = float(len(average_precisions))
    return result
