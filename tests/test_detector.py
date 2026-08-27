from PIL import Image

from styleseek.detector import GarmentDetection, crop_detection, draw_detections


def test_detection_crop_adds_bounded_padding():
    image = Image.new("RGB", (400, 600), "white")
    detection = GarmentDetection((100, 150, 300, 450), 1, "long sleeve top", 0.9)

    crop = crop_detection(image, detection, padding=0.1)

    assert crop.size == (240, 360)


def test_draw_detections_preserves_image_size():
    image = Image.new("RGB", (400, 600), "white")
    detection = GarmentDetection((100, 150, 300, 450), 1, "long sleeve top", 0.9)

    assert draw_detections(image, [detection]).size == image.size
