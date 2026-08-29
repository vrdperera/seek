from __future__ import annotations

import argparse
import time
from pathlib import Path

import gradio as gr
import torch
from PIL import Image

from styleseek.catalogue import require_catalogue_images, resolve_catalogue_paths
from styleseek.data import build_transform
from styleseek.detector import (
    GarmentDetection,
    GarmentDetector,
    PersonDetector,
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
from styleseek.retrieval import rank_catalogue
from styleseek.ui_state import retrieval_block_reason


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the StyleSeek Gradio demo")
    parser.add_argument("--checkpoint", default=str(RETRIEVAL_CHECKPOINT))
    parser.add_argument("--catalogue", default=str(CATALOGUE_INDEX))
    parser.add_argument("--detector", default=str(DETECTOR_CHECKPOINT))
    parser.add_argument("--detection-confidence", type=float, default=0.30)
    parser.add_argument("--person-detector", default="yolo11n.pt")
    parser.add_argument("--person-confidence", type=float, default=0.35)
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
    person_detector_path: str | None = "yolo11n.pt",
    person_confidence: float = 0.35,
):
    device = choose_device(device_name)
    model, payload = load_checkpoint(checkpoint_path, device)
    catalogue = torch.load(catalogue_path, map_location="cpu", weights_only=False)
    embeddings = catalogue["embeddings"].float()
    paths = resolve_catalogue_paths(
        catalogue["paths"], catalogue.get("path_base")
    )
    require_catalogue_images(paths)
    product_ids = catalogue["product_ids"]
    categories = catalogue.get("categories") or None
    transform = build_transform(False, int(payload.get("image_size", 224)))
    detector = None
    if detector_path and Path(detector_path).exists():
        detector = GarmentDetector(detector_path, device=str(device))
    person_detector = None
    if person_detector_path and Path(person_detector_path).exists():
        person_detector = PersonDetector(person_detector_path, device=str(device))

    def choice_label(index: int, detection: GarmentDetection) -> str:
        return f"{index + 1} · {detection.category} · {detection.confidence:.0%}"

    def detect_garments(image: Image.Image):
        if image is None:
            return (
                None,
                gr.Dropdown(choices=[], value=None, interactive=False),
                [],
                "Upload a photograph first.",
                None,
                [],
                "Retrieval is unavailable until a garment is detected.",
                gr.Button(interactive=False),
            )
        started = time.perf_counter()
        if person_detector is None:
            detections = []
            message = (
                "Person validation unavailable: install the COCO-pretrained yolo11n.pt "
                "weights before searching."
            )
        elif not person_detector.detect(image, confidence=person_confidence):
            detections = []
            message = (
                "No person detected. Upload a full-person photograph containing clearly "
                f"visible clothing (person confidence threshold: {person_confidence:.0%})."
            )
        elif detector is None:
            detections = []
            message = (
                "Garment detector unavailable: install best.pt under "
                "artifacts/checkpoints/detector/ before searching."
            )
        else:
            detections = detector.detect(image, confidence=detection_confidence)
            if not detections:
                message = (
                    "No garment detected. Upload a photograph containing clearly visible clothing "
                    f"(confidence threshold: {detection_confidence:.0%})."
                )
            else:
                elapsed = (time.perf_counter() - started) * 1000
                message = f"Detected {len(detections)} garment(s) in {elapsed:.1f} ms. Select one below."
        choices = [choice_label(index, item) for index, item in enumerate(detections)]
        annotated = draw_detections(image, detections)
        can_retrieve = bool(detections)
        return (
            annotated,
            gr.Dropdown(
                choices=choices,
                value=choices[0] if choices else None,
                interactive=can_retrieve,
            ),
            [item.to_dict() for item in detections],
            message,
            None,
            [],
            "" if can_retrieve else "Retrieval blocked because no garment was detected.",
            gr.Button(interactive=can_retrieve),
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
        return crop_detection(image, detection) if detection else None

    def retrieve(image: Image.Image, selection: str, detection_values: list[dict]):
        block_reason = retrieval_block_reason(image is not None, detection_values)
        if block_reason:
            return None, [], block_reason
        started = time.perf_counter()
        detection = selected_detection(selection, detection_values)
        if detection is None:
            return None, [], "Retrieval blocked: select a detected garment first."
        garment = crop_detection(image, detection)
        tensor = transform(garment).unsqueeze(0).to(device)
        with torch.inference_mode():
            query = model(tensor).cpu().float()
        category = detection.category
        values, indices, category_filtered = rank_catalogue(
            query,
            embeddings,
            top_k,
            categories,
            category,
        )
        count = len(indices)
        results = []
        for rank, (score, index) in enumerate(
            zip(values.tolist(), indices.tolist()), start=1
        ):
            item_category = categories[index] if categories else "category unavailable"
            caption = (
                f"Rank {rank} · Product {product_ids[index]} · {item_category} · "
                f"similarity {score:.3f}"
            )
            results.append((paths[index], caption))
        latency_ms = (time.perf_counter() - started) * 1000
        filter_status = (
            f"Filtered catalogue to '{category}'."
            if category_filtered
            else "Category filter unavailable; searched the full catalogue."
        )
        return (
            garment,
            results,
            f"{filter_status} Retrieved {count} products in {latency_ms:.1f} ms on {device}.",
        )

    example_paths = [
        str(path)
        for path in (SAMPLE_DATA_DIR / "full_person").glob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
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
                garment_choice = gr.Dropdown(
                    label="2. Select a garment", choices=[], interactive=False
                )
                detection_status = gr.Textbox(label="Detection details", interactive=False)
                retrieve_button = gr.Button(
                    "3. Find matching products", variant="primary", interactive=False
                )
        with gr.Row():
            selected_crop = gr.Image(type="pil", label="Garment used for search", interactive=False)
            gallery = gr.Gallery(label="Matching catalogue garments", columns=5, height="auto")
        retrieval_status = gr.Textbox(label="Retrieval details", interactive=False)
        if example_paths:
            gr.Examples(examples=example_paths, inputs=upload)

        detect_button.click(
            fn=detect_garments,
            inputs=upload,
            outputs=[
                annotated,
                garment_choice,
                detection_state,
                detection_status,
                selected_crop,
                gallery,
                retrieval_status,
                retrieve_button,
            ],
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
        checkpoint_path=args.checkpoint,
        catalogue_path=args.catalogue,
        top_k=args.top_k,
        device_name=args.device,
        detector_path=args.detector,
        detection_confidence=args.detection_confidence,
        person_detector_path=args.person_detector,
        person_confidence=args.person_confidence,
    )
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
