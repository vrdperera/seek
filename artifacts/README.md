# Runtime and training artifacts

- `checkpoints/retrieval/`: trained ResNet18 checkpoints.
- `checkpoints/detector/`: promoted YOLO checkpoints used by the application.
- `indexes/`: precomputed catalogue embedding indexes.
- `runs/`: complete framework-generated training runs, including `best.pt`, `last.pt`, plots, and logs.
- `cache/`: downloaded pretrained weights and tool configuration.

Artifacts are generated, excluded from Git, and should be versioned in object storage or an experiment registry in production.
