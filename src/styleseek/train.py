from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import TripletFashionDataset
from .evaluation import evaluate_model
from .model import FashionEncoder, save_checkpoint
from .paths import DEFAULT_MANIFEST, RETRIEVAL_CHECKPOINT
from .utils import choose_device, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the StyleSeek triplet model")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output", default=str(RETRIEVAL_CHECKPOINT))
    parser.add_argument("--mode", choices=["frozen", "layer4", "full"], default="frozen")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)
    print(f"Device: {device}")

    train_dataset = TripletFashionDataset(
        args.manifest, "train", args.image_size, train_augmentation=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model = FashionEncoder(args.embedding_dim, pretrained=True).to(device)
    model.set_trainable(args.mode)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    default_lr = 1e-3 if args.mode == "frozen" else 1e-4
    optimizer = AdamW(parameters, lr=args.learning_rate or default_lr, weight_decay=1e-4)
    criterion = nn.TripletMarginLoss(margin=args.margin, p=2)

    best_recall = -1.0
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.mode == "frozen":
            model.backbone.eval()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for anchor, positive, negative, _ in progress:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(anchor), model(positive), model(negative))
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * anchor.size(0)
            progress.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = running_loss / len(train_dataset)
        metrics, _ = evaluate_model(
            model,
            args.manifest,
            "val",
            device,
            args.image_size,
            args.batch_size,
            args.workers,
        )
        record = {"epoch": float(epoch), "loss": epoch_loss, **metrics}
        history.append(record)
        print(
            f"loss={epoch_loss:.4f} recall@1={metrics['recall@1']:.4f} "
            f"recall@5={metrics['recall@5']:.4f} mAP={metrics['map']:.4f}"
        )
        if metrics["recall@1"] > best_recall:
            best_recall = metrics["recall@1"]
            save_checkpoint(
                model,
                args.output,
                image_size=args.image_size,
                training_mode=args.mode,
                metrics=metrics,
            )

    history_path = Path(args.output).with_suffix(".history.json")
    save_json({"configuration": vars(args), "epochs": history}, history_path)
    print(f"Best checkpoint: {args.output}")
    print(f"Training history: {history_path}")


if __name__ == "__main__":
    main()
