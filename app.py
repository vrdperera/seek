from __future__ import annotations

import argparse
import time
from pathlib import Path

import gradio as gr
import torch
from PIL import Image

from styleseek.data import build_transform
from styleseek.detector import (
    GarmentDetection,
    GarmentDetector,
    crop_detection,
    draw_detections,
)
from styleseek.model import load_checkpoint
from styleseek.paths import (
    CATALOGUE_INDEX,
    DETECTOR_CHECKPOINT,
    RETRIEVAL_CHECKPOINT,
    SAMPLE_DATA_DIR,
)
from styleseek.utils import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the StyleSeek Gradio demo")
    parser.add_argument("--checkpoint", default=str(RETRIEVAL_CHECKPOINT))
    parser.add_argument("--catalogue", default=str(CATALOGUE_INDEX))
    parser.add_argument("--detector", default=str(DETECTOR_CHECKPOINT))
    parser.add_argument("--detection-confidence", type=float, default=0.30)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def create_demo(
    checkpoint_path: str,
    catalogue_path: str,
    top_k: int,
    device_name: str,
    detector_path: str | None = None,
    detection_confidence: float = 0.30,
):
    device = choose_device(device_name)
    model, payload = load_checkpoint(checkpoint_path, device)
    catalogue = torch.load(catalogue_path, map_location="cpu", weights_only=False)
    embeddings = catalogue["embeddings"].float()
    paths = catalogue["paths"]
    product_ids = catalogue["product_ids"]
    transform = build_transform(False, int(payload.get("image_size", 224)))
    detector = None
    if detector_path and Path(detector_path).exists():
        detector = GarmentDetector(detector_path, device=str(device))

    def choice_label(index: int, detection: GarmentDetection) -> str:
        return f"{index + 1} · {detection.category} · {detection.confidence:.0%}"

    def full_image_detection(image: Image.Image) -> GarmentDetection:
        return GarmentDetection(
            box=(0, 0, image.width, image.height),
            class_id=-1,
            category="full image (detector unavailable)",
            confidence=1.0,
        )

    def detect_garments(image: Image.Image):
        if image is None:
            return None, gr.Dropdown(choices=[], value=None), [], "Upload a photograph first."
        started = time.perf_counter()
        if detector is None:
            detections = [full_image_detection(image)]
            message = (
                "Detector weights are not installed yet. Retrieval will use the full image; "
                "train the detector and install best.pt under artifacts/checkpoints/detector/."
            )
        else:
            detections = detector.detect(image, confidence=detection_confidence)
            if not detections:
                detections = [full_image_detection(image)]
                message = "No garment exceeded the confidence threshold; using the full image."
            else:
                elapsed = (time.perf_counter() - started) * 1000
                message = f"Detected {len(detections)} garment(s) in {elapsed:.1f} ms. Select one below."
        choices = [choice_label(index, item) for index, item in enumerate(detections)]
        annotated = draw_detections(image, detections)
        return (
            annotated,
            gr.Dropdown(choices=choices, value=choices[0]),
            [item.to_dict() for item in detections],
            message,
        )

    def selected_detection(selection: str | None, detection_values: list[dict]):
        if not detection_values:
            return None
        try:
            index = int(str(selection).split("·", maxsplit=1)[0].strip()) - 1
        except (TypeError, ValueError):
            index = 0
        index = max(0, min(index, len(detection_values) - 1))
        return GarmentDetection.from_dict(detection_values[index])

    def preview_crop(image: Image.Image, selection: str, detection_values: list[dict]):
        if image is None:
            return None
        detection = selected_detection(selection, detection_values)
        return crop_detection(image, detection) if detection else image

    def retrieve(image: Image.Image, selection: str, detection_values: list[dict]):
        if image is None:
            return None, [], "Upload an image to start."
        started = time.perf_counter()
        detection = selected_detection(selection, detection_values)
        garment = crop_detection(image, detection) if detection else image.convert("RGB")
        tensor = transform(garment).unsqueeze(0).to(device)
        with torch.inference_mode():
            query = model(tensor).cpu().float()
        scores = (query @ embeddings.T).squeeze(0)
        count = min(top_k, len(paths))
        values, indices = torch.topk(scores, k=count)
        results = []
        for score, index in zip(values.tolist(), indices.tolist()):
            caption = f"Product {product_ids[index]} · similarity {score:.3f}"
            results.append((paths[index], caption))
        latency_ms = (time.perf_counter() - started) * 1000
        category = detection.category if detection else "full image"
        return (
            garment,
            results,
            f"Searched using '{category}'. Retrieved {count} products in {latency_ms:.1f} ms on {device}.",
        )

    example_paths = [
        str(path)
        for path in (SAMPLE_DATA_DIR / "images").glob("*_consumer_*.jpg")
    ][:4]
    with gr.Blocks(title="StyleSeek AI") as demo:
        gr.Markdown(
            "# StyleSeek AI\n"
            "Upload a full-person photograph, detect its garments, select one, and retrieve "
            "matching catalogue products."
        )
        detection_state = gr.State([])
        with gr.Row():
            with gr.Column():
                upload = gr.Image(type="pil", label="Full consumer photograph")
                detect_button = gr.Button("1. Detect garments", variant="primary")
            with gr.Column():
                annotated = gr.Image(type="pil", label="Detected garments", interactive=False)
                garment_choice = gr.Dropdown(label="2. Select a garment", choices=[])
                detection_status = gr.Textbox(label="Detection details", interactive=False)
                retrieve_button = gr.Button("3. Find matching products", variant="primary")
        with gr.Row():
            selected_crop = gr.Image(type="pil", label="Garment used for search", interactive=False)
            gallery = gr.Gallery(label="Matching catalogue garments", columns=5, height="auto")
        retrieval_status = gr.Textbox(label="Retrieval details", interactive=False)
        if example_paths:
            gr.Examples(examples=example_paths, inputs=upload)

        detect_button.click(
            fn=detect_garments,
            inputs=upload,
            outputs=[annotated, garment_choice, detection_state, detection_status],
        )
        garment_choice.change(
            fn=preview_crop,
            inputs=[upload, garment_choice, detection_state],
            outputs=selected_crop,
        )
        retrieve_button.click(
            fn=retrieve,
            inputs=[upload, garment_choice, detection_state],
            outputs=[selected_crop, gallery, retrieval_status],
        )
    return demo


def main() -> None:
    args = parse_args()
    demo = create_demo(
        args.checkpoint,
        args.catalogue,
        args.top_k,
        args.device,
        args.detector,
        args.detection_confidence,
    )
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
