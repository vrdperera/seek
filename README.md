# StyleSeek AI

StyleSeek AI is a consumer-to-shop fashion retrieval system. A COCO-pretrained person gate first rejects inputs without a visible person, a fine-tuned YOLO detector then locates garments, and a contrastively trained ResNet18 encoder retrieves matching shop products from a precomputed catalogue index.

## Repository layout

```text
seek/
├── app.py                         # Gradio application entry point
├── configs/                       # Versioned experiment configuration
├── docs/                          # Architecture and project documentation
├── scripts/                       # Reproducible data-preparation jobs
├── src/styleseek/                 # Installable application and ML package
├── tests/                         # Automated unit and structural tests
├── data/
│   ├── raw/                       # Immutable downloaded datasets
│   ├── interim/                   # YOLO labels and intermediate transforms
│   ├── processed/retrieval/       # Model-ready crops and manifest
│   └── samples/                   # Synthetic smoke-test data
├── artifacts/
│   ├── checkpoints/detector/      # Promoted YOLO weights
│   ├── checkpoints/retrieval/     # Promoted ResNet18 weights
│   ├── indexes/                   # Catalogue embedding indexes
│   ├── runs/                      # Complete training-run outputs
│   └── cache/                     # Downloaded weights/tool configuration
└── reports/
    ├── metrics/                   # Evaluation and latency JSON
    └── figures/                   # Plots and qualitative results
```

Raw and generated data, large checkpoints, catalogue indexes and generated figures are
excluded from Git. Lightweight metrics, training histories and detector result summaries are
tracked as experimental evidence; all generated assets remain reproducible from the source
and scripts.

## Supervisor quick start

The source repository keeps large runtime binaries outside Git. A versioned GitHub
Release supplies the trained checkpoints and a small generated catalogue whose paths are
portable across computers.

```bash
git clone https://github.com/vrdperera/seek.git
cd seek
uv sync --locked
uv run python scripts/download_demo.py
uv run pytest
uv run python app.py
```

The release catalogue is generated specifically for portable software demonstration. It
does not replace the identity-disjoint DeepFashion2 test catalogue used to calculate the
reported retrieval metrics. The small metric JSON files and training histories are kept
in Git so the reported evidence remains inspectable without downloading the dataset.

### Create the GitHub Release bundle

Run this command from the trained project workspace:

```bash
uv run python scripts/package_demo_release.py
```

It creates both files below:

```text
dist/styleseek-demo-v1.zip
dist/styleseek-demo-v1.zip.sha256
```

Create a GitHub Release tagged `v1.0-demo`, then upload both files. The downloader verifies
the SHA-256 checksum before extracting the artifacts. Each release bundle contains the
trained retrieval checkpoint, garment detector, COCO person-detector weights, portable
catalogue index and generated catalogue images. Do not add the raw DeepFashion2 archives
to the release.

## Environment

Python 3.12 is pinned in `.python-version` and dependencies are locked in `uv.lock`.

```bash
uv sync --locked
uv run pytest
```

No manual virtual-environment activation is required.

## Existing raw data

The current workspace contains the official extracted dataset and a preserved mirror:

```text
data/raw/deepfashion2/
data/raw/deepfashion2_mirror/
```

Use one source consistently. The commands below use the official extracted dataset at `data/raw/deepfashion2/`.

## 1. Prepare detector data

This converts full consumer photographs and DeepFashion2 bounding boxes into YOLO format. Symbolic links avoid duplicating the large image dataset.

```bash
uv run python scripts/prepare_deepfashion2_detection.py \
  --dataset-root data/raw/deepfashion2 \
  --output data/interim/detection \
  --max-train-images 5000 \
  --max-val-images 1000
```

## 2. Fine-tune the garment detector

```bash
uv run python -m styleseek.train_detector \
  --data data/interim/detection/deepfashion2.yaml \
  --model yolo11n.pt \
  --epochs 10 \
  --batch-size 8 \
  --device mps
```

The complete run is written under:

```text
artifacts/runs/detector/deepfashion2_yolo/
```

Promote the best checkpoint for application use:

```bash
cp artifacts/runs/detector/deepfashion2_yolo/weights/best.pt \
  artifacts/checkpoints/detector/best.pt
```

## 3. Prepare retrieval data

This matches consumer and shop items using `pair_id + style`, crops their annotated garment boxes, and creates identity-disjoint train, validation, and test splits.

```bash
uv run python scripts/prepare_deepfashion2.py \
  --dataset-root data/raw/deepfashion2 \
  --output data/processed/retrieval/manifest.csv \
  --max-products 1000
```

To grow an experiment without moving validation/test identities into training, use the
existing manifest as a split reference and write the larger dataset separately:

```bash
uv run python scripts/prepare_deepfashion2.py \
  --dataset-root data/raw/deepfashion2 \
  --reference-manifest data/processed/retrieval/manifest.csv \
  --output data/processed/retrieval_5k/manifest.csv \
  --max-products 5000
```

## 4. Evaluate the pretrained baseline

```bash
uv run python -m styleseek.evaluate \
  --manifest data/processed/retrieval/manifest.csv \
  --output reports/metrics/baseline.json
```

## 5. Train retrieval experiments

The default objective is symmetric InfoNCE: a batch contains matched consumer/catalogue
pairs, and all non-matching products in the batch act as negatives. Duplicate product IDs
are treated as additional positives instead of false negatives. Use larger batches when
memory permits because they provide more in-batch negatives. The previous triplet
objective remains available with `--loss triplet` for comparison.

Frozen-backbone experiment:

```bash
uv run python -m styleseek.train \
  --manifest data/processed/retrieval/manifest.csv \
  --mode frozen \
  --loss contrastive \
  --temperature 0.07 \
  --epochs 8 \
  --batch-size 16 \
  --output artifacts/checkpoints/retrieval/frozen.pt
```

Partial fine-tuning experiment:

```bash
uv run python -m styleseek.train \
  --manifest data/processed/retrieval/manifest.csv \
  --mode layer4 \
  --loss contrastive \
  --temperature 0.07 \
  --epochs 5 \
  --batch-size 8 \
  --output artifacts/checkpoints/retrieval/best.pt
```

Each command saves an adjacent `.history.json` containing its configuration, loss, and validation metrics.

## 6. Evaluate the trained model

```bash
uv run python -m styleseek.evaluate \
  --checkpoint artifacts/checkpoints/retrieval/best.pt \
  --split test \
  --output reports/metrics/retrieval_test.json
```

Metrics include Recall@1, Recall@5, Recall@10, and mAP.

## 7. Build the catalogue index

```bash
uv run python -m styleseek.index \
  --checkpoint artifacts/checkpoints/retrieval/best.pt \
  --output artifacts/indexes/catalogue.pt
```

`catalogue.pt` is an embedding index, not a model checkpoint. Rebuild it whenever the retrieval checkpoint or shop catalogue changes.

## 8. Run the application

```bash
uv run python app.py
```

The application defaults to:

```text
artifacts/checkpoints/detector/best.pt
artifacts/checkpoints/retrieval/best.pt
artifacts/indexes/catalogue.pt
```

Its online flow is: upload, detect garments, select a garment, crop, embed, rank catalogue products, and display the top matches.

Catalogue indexes built from manifests containing a `category` column automatically
restrict retrieval to the selected detector category. If category metadata or matching
candidates are unavailable, the application falls back to the complete catalogue.

To upgrade an older DeepFashion2 manifest that predates category metadata:

```bash
uv run python scripts/add_manifest_categories.py \
  --dataset-root data/raw/deepfashion2 \
  --manifest data/processed/retrieval_5k/manifest.csv
```

## 9. Benchmark latency

```bash
uv run python -m styleseek.benchmark \
  --runs 30 \
  --output reports/metrics/latency.json
```

## Smoke-test assets

The previous synthetic outputs were preserved and separated from real experiments:

```text
data/samples/manifest.csv
artifacts/checkpoints/retrieval/smoke.pt
artifacts/indexes/smoke_catalogue.pt
reports/metrics/*_smoke.json
```

They validate software behavior only. To launch them explicitly:

```bash
uv run python app.py \
  --checkpoint artifacts/checkpoints/retrieval/smoke.pt \
  --catalogue artifacts/indexes/smoke_catalogue.pt
```

If the detector checkpoint is absent, the interface reports the condition and falls back to the entire image.

## Evaluation table

| Model | Backbone training | Recall@1 | Recall@5 | Recall@10 | mAP |
|---|---|---:|---:|---:|---:|
| ImageNet ResNet18 | None | | | | |
| Triplet ResNet18 | Frozen | | | | |
| Triplet ResNet18 | Layer 4 | | | | |

Populate this table from files under `reports/metrics/`; do not invent values.

See [docs/architecture.md](docs/architecture.md) and the lifecycle notes inside `data/`, `artifacts/`, and `reports/`.
