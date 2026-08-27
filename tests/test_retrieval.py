import pytest
import torch

from styleseek.retrieval import retrieval_metrics
from styleseek.losses import contrastive_loss


def test_retrieval_metrics_for_perfect_ranking():
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    gallery = queries.clone()

    metrics = retrieval_metrics(queries, ["a", "b"], gallery, ["a", "b"])

    assert metrics["recall@1"] == pytest.approx(1.0)
    assert metrics["recall@5"] == pytest.approx(1.0)
    assert metrics["map"] == pytest.approx(1.0)


def test_contrastive_loss_prefers_matching_pairs():
    matching = torch.eye(3)
    mismatched = matching[[1, 2, 0]]

    good_loss = contrastive_loss(matching, matching, temperature=0.07)
    bad_loss = contrastive_loss(matching, mismatched, temperature=0.07)

    assert good_loss < bad_loss


def test_contrastive_loss_accepts_duplicate_product_ids():
    embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    loss = contrastive_loss(embeddings, embeddings, ["a", "a", "b"])

    assert torch.isfinite(loss)
