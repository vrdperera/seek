from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .paths import ULTRALYTICS_CONFIG_DIR

ULTRALYTICS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_DIR))

from ultralytics import YOLO

from .categories import GARMENT_CATEGORIES


@dataclass(frozen=True)
class GarmentDetection:
    box: tuple[int, int, int, int]
    class_id: int
    category: str
    confidence: float

    def to_dict(self) -> dict:
        result = asdict(self)
        result["box"] = list(self.box)
        return result

    @classmethod
    def from_dict(cls, value: dict) -> "GarmentDetection":
        return cls(
            box=tuple(int(number) for number in value["box"]),
            class_id=int(value["class_id"]),
            category=str(value["category"]),
            confidence=float(value["confidence"]),
        )


@dataclass(frozen=True)
class PersonDetection:
    box: tuple[int, int, int, int]
    confidence: float


class PersonDetector:
    """COCO-pretrained person gate used before the fashion detector."""

    def __init__(self, weights: str | Path, device: str = "cpu") -> None:
        self.weights = Path(weights)
        if not self.weights.exists():
            raise FileNotFoundError(f"Person detector weights not found: {self.weights}")
        self.model = YOLO(str(self.weights))
        self.device = device

    def detect(
        self,
        image: Image.Image,
        confidence: float = 0.35,
        max_detections: int = 5,
    ) -> list[PersonDetection]:
        result = self.model.predict(
            source=image.convert("RGB"),
            conf=confidence,
            classes=[0],
            max_det=max_detections,
            device=self.device,
            verbose=False,
        )[0]
        detections: list[PersonDetection] = []
        if result.boxes is None:
            return detections
        for box in result.boxes:
            x1, y1, x2, y2 = (int(round(value)) for value in box.xyxy[0].tolist())
            detections.append(
                PersonDetection(
                    box=(x1, y1, x2, y2),
                    confidence=float(box.conf[0].item()),
                )
            )
        return sorted(detections, key=lambda item: item.confidence, reverse=True)


class GarmentDetector:
    def __init__(self, weights: str | Path, device: str = "cpu") -> None:
        self.weights = Path(weights)
        if not self.weights.exists():
            raise FileNotFoundError(f"Garment detector weights not found: {self.weights}")
        self.model = YOLO(str(self.weights))
        self.device = device

    def detect(
        self,
        image: Image.Image,
        confidence: float = 0.30,
        max_detections: int = 10,
    ) -> list[GarmentDetection]:
        result = self.model.predict(
            source=image.convert("RGB"),
            conf=confidence,
            max_det=max_detections,
            device=self.device,
            verbose=False,
        )[0]
        detections: list[GarmentDetection] = []
        if result.boxes is None:
            return detections
        for box in result.boxes:
            x1, y1, x2, y2 = (int(round(value)) for value in box.xyxy[0].tolist())
            class_id = int(box.cls[0].item())
            category = (
                GARMENT_CATEGORIES[class_id]
                if 0 <= class_id < len(GARMENT_CATEGORIES)
                else f"garment {class_id}"
            )
            detections.append(
                GarmentDetection(
                    box=(x1, y1, x2, y2),
                    class_id=class_id,
                    category=category,
                    confidence=float(box.conf[0].item()),
                )
            )
        return sorted(detections, key=lambda item: item.confidence, reverse=True)


def crop_detection(
    image: Image.Image,
    detection: GarmentDetection,
    padding: float = 0.08,
) -> Image.Image:
    image = image.convert("RGB")
    x1, y1, x2, y2 = detection.box
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    pad_x = int(width * padding)
    pad_y = int(height * padding)
    bounds = (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(image.width, x2 + pad_x),
        min(image.height, y2 + pad_y),
    )
    return image.crop(bounds)


def draw_detections(
    image: Image.Image,
    detections: list[GarmentDetection],
) -> Image.Image:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default(size=16)
    colors = ["#00E5FF", "#FFB000", "#FF4D8D", "#7CFF6B", "#C084FC"]
    for index, detection in enumerate(detections):
        color = colors[index % len(colors)]
        x1, y1, x2, y2 = detection.box
        draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
        label = f"{index + 1}. {detection.category} {detection.confidence:.0%}"
        text_box = draw.textbbox((x1, y1), label, font=font)
        label_height = text_box[3] - text_box[1] + 8
        label_top = max(0, y1 - label_height)
        draw.rectangle((x1, label_top, max(x2, x1 + text_box[2] - text_box[0] + 8), y1), fill=color)
        draw.text((x1 + 4, label_top + 4), label, fill="black", font=font)
    return annotated
