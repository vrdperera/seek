import pytest
import torch

from styleseek.retrieval import rank_catalogue, retrieval_metrics
from styleseek.losses import contrastive_loss
from styleseek.preparation import select_products


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


def test_larger_selection_preserves_reference_splits_and_adds_only_training_ids():
    reference = {"a": "train", "b": "val", "c": "test"}
    selected, assignments = select_products(
        ["a", "b", "c", "d", "e"], 5, 42, reference
    )

    assert set(selected) == {"a", "b", "c", "d", "e"}
    assert assignments["a"] == "train"
    assert assignments["b"] == "val"
    assert assignments["c"] == "test"
    assert assignments["d"] == "train"
    assert assignments["e"] == "train"


def test_rank_catalogue_filters_candidates_by_category():
    query = torch.tensor([[1.0, 0.0]])
    catalogue = torch.tensor([[0.8, 0.0], [1.0, 0.0], [0.7, 0.0]])

    values, indices, filtered = rank_catalogue(
        query,
        catalogue,
        top_k=5,
        catalogue_categories=["outerwear", "trousers", "outerwear"],
        query_category="outerwear",
    )

    assert filtered is True
    assert indices.tolist() == [0, 2]
    assert values.tolist() == pytest.approx([0.8, 0.7])


def test_rank_catalogue_falls_back_when_category_is_absent():
    query = torch.tensor([[1.0, 0.0]])
    catalogue = torch.tensor([[0.8, 0.0], [1.0, 0.0]])

    _, indices, filtered = rank_catalogue(
        query,
        catalogue,
        top_k=2,
        catalogue_categories=["outerwear", "trousers"],
        query_category="skirt",
    )

    assert filtered is False
    assert indices.tolist() == [1, 0]
