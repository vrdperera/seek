import pytest
import torch

from styleseek.retrieval import retrieval_metrics


def test_retrieval_metrics_for_perfect_ranking():
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    gallery = queries.clone()

    metrics = retrieval_metrics(queries, ["a", "b"], gallery, ["a", "b"])

    assert metrics["recall@1"] == pytest.approx(1.0)
    assert metrics["recall@5"] == pytest.approx(1.0)
    assert metrics["map"] == pytest.approx(1.0)
