from styleseek.paths import (
    CATALOGUE_INDEX,
    DEFAULT_MANIFEST,
    DETECTOR_CHECKPOINT,
    RETRIEVAL_CHECKPOINT,
)


def test_artifacts_have_explicit_roles():
    assert "checkpoints/retrieval" in RETRIEVAL_CHECKPOINT.as_posix()
    assert "checkpoints/detector" in DETECTOR_CHECKPOINT.as_posix()
    assert "indexes" in CATALOGUE_INDEX.as_posix()
    assert "processed/retrieval" in DEFAULT_MANIFEST.as_posix()
