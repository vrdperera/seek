from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.nn import functional as F


def contrastive_loss(
    query_embeddings: torch.Tensor,
    product_embeddings: torch.Tensor,
    product_ids: Sequence[str] | None = None,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Symmetric InfoNCE loss with support for duplicate identities in a batch."""
    if query_embeddings.shape != product_embeddings.shape:
        raise ValueError("Query and product embedding tensors must have the same shape")
    if query_embeddings.ndim != 2:
        raise ValueError("Embeddings must have shape [batch, embedding_dim]")
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")

    queries = F.normalize(query_embeddings, dim=1)
    products = F.normalize(product_embeddings, dim=1)
    logits = queries @ products.T / temperature

    batch_size = logits.size(0)
    if product_ids is None:
        positive_mask = torch.eye(batch_size, dtype=torch.bool, device=logits.device)
    else:
        if len(product_ids) != batch_size:
            raise ValueError("product_ids length must equal the batch size")
        positive_mask = torch.tensor(
            [[left == right for right in product_ids] for left in product_ids],
            dtype=torch.bool,
            device=logits.device,
        )

    def direction_loss(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        log_probabilities = F.log_softmax(scores, dim=1)
        positive_counts = mask.sum(dim=1).clamp_min(1)
        return -(
            (log_probabilities * mask).sum(dim=1) / positive_counts
        ).mean()

    return 0.5 * (
        direction_loss(logits, positive_mask)
        + direction_loss(logits.T, positive_mask.T)
    )
