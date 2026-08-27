# Data lifecycle

This directory follows an immutable-to-derived data lifecycle:

- `raw/`: source datasets exactly as downloaded. Never edit these files.
- `interim/`: transformed data used by another preparation step, including YOLO labels and image links.
- `processed/`: final model-ready retrieval manifests and garment crops.
- `samples/`: small synthetic or hand-picked examples used only for smoke tests.

Large data is excluded from Git. Every derived dataset must be reproducible from a script in `scripts/`.
