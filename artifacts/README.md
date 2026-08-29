# Runtime and training artifacts

- `checkpoints/retrieval/`: trained ResNet18 checkpoints.
- `checkpoints/detector/`: promoted YOLO checkpoints used by the application.
- `indexes/`: precomputed catalogue embedding indexes.
- `runs/`: complete framework-generated training runs, including `best.pt`, `last.pt`, plots, and logs.
- `cache/`: downloaded pretrained weights and tool configuration.

Artifacts are generated, excluded from Git, and should be versioned in object storage or an experiment registry in production.

For coursework handoff, `scripts/package_demo_release.py` packages the promoted detector,
retrieval checkpoint, person-detector weights and a generated portable catalogue into a
GitHub Release ZIP. `scripts/download_demo.py` verifies and installs that ZIP after a fresh
clone. Lightweight histories, detector result summaries and final evaluation JSON files
remain tracked in Git as inspectable experimental evidence.
