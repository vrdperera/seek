import pytest
import torch
from PIL import Image

from styleseek.detector import (
    GarmentDetection,
    PersonDetector,
    crop_detection,
    draw_detections,
)


def test_detection_crop_adds_bounded_padding():
    image = Image.new("RGB", (400, 600), "white")
    detection = GarmentDetection((100, 150, 300, 450), 1, "long sleeve top", 0.9)

    crop = crop_detection(image, detection, padding=0.1)

    assert crop.size == (240, 360)


def test_draw_detections_preserves_image_size():
    image = Image.new("RGB", (400, 600), "white")
    detection = GarmentDetection((100, 150, 300, 450), 1, "long sleeve top", 0.9)

    assert draw_detections(image, [detection]).size == image.size


def test_person_detector_requests_only_the_coco_person_class():
    class FakeBox:
        xyxy = torch.tensor([[10.0, 20.0, 110.0, 220.0]])
        conf = torch.tensor([0.87])

    class FakeResult:
        boxes = [FakeBox()]

    class FakeModel:
        def predict(self, **kwargs):
            assert kwargs["classes"] == [0]
            return [FakeResult()]

    detector = PersonDetector.__new__(PersonDetector)
    detector.model = FakeModel()
    detector.device = "cpu"

    detections = detector.detect(Image.new("RGB", (320, 320), "white"))

    assert len(detections) == 1
    assert detections[0].box == (10, 20, 110, 220)
    assert detections[0].confidence == pytest.approx(0.87)
