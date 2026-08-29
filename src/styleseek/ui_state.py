from __future__ import annotations


def retrieval_block_reason(
    image_present: bool,
    detection_values: list[dict] | None,
) -> str | None:
    """Return a user-facing reason when retrieval must not run."""
    if not image_present:
        return "Upload an image to start."
    if not detection_values:
        return (
            "Retrieval blocked: no garment was detected. "
            "Upload a photograph containing visible clothing."
        )
    return None
