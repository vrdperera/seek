from styleseek.ui_state import retrieval_block_reason


def test_retrieval_is_blocked_without_an_uploaded_image():
    assert retrieval_block_reason(False, []) == "Upload an image to start."


def test_retrieval_is_blocked_when_no_garment_was_detected():
    message = retrieval_block_reason(True, [])

    assert message is not None
    assert "no garment was detected" in message.lower()


def test_retrieval_is_allowed_after_a_real_detection():
    detection = {
        "box": [10, 20, 100, 200],
        "class_id": 1,
        "category": "long sleeve top",
        "confidence": 0.91,
    }

    assert retrieval_block_reason(True, [detection]) is None
