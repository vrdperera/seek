# StyleSeek AI architecture

## Offline workflows

1. DeepFashion2 raw images and annotations are placed under `data/raw/`.
2. `prepare_deepfashion2_detection.py` creates YOLO labels under `data/interim/detection/`.
3. `train_detector` fine-tunes YOLO and records the run under `artifacts/runs/detector/`.
4. The best detector checkpoint is promoted to `artifacts/checkpoints/detector/best.pt`.
5. `prepare_deepfashion2.py` creates paired crops and a manifest under `data/processed/retrieval/`.
6. `train` fine-tunes the shared ResNet18 encoder and saves `artifacts/checkpoints/retrieval/best.pt`.
7. `index` embeds shop images and writes `artifacts/indexes/catalogue.pt`.

## Online workflow

```text
Full photograph
  -> YOLO garment detection
  -> user garment selection
  -> padded garment crop
  -> ResNet18 normalized embedding
  -> cosine similarity against the catalogue index
  -> top-K products
```

## Evaluation

- Detector: precision, recall, and detection mAP.
- Retrieval: Recall@1, Recall@5, Recall@10, and mAP.
- System: end-to-end latency and user acceptance.
