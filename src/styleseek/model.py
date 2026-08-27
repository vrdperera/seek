from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18


class FashionEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 256, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.embedding_dim = embedding_dim
        self.projection = (
            nn.Identity()
            if embedding_dim == feature_dim
            else nn.Linear(feature_dim, embedding_dim)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(self.backbone(images)), p=2, dim=1)

    def set_trainable(self, mode: str) -> None:
        if mode not in {"frozen", "layer4", "full"}:
            raise ValueError("mode must be one of: frozen, layer4, full")
        for parameter in self.backbone.parameters():
            parameter.requires_grad = mode == "full"
        if mode == "layer4":
            for parameter in self.backbone.layer4.parameters():
                parameter.requires_grad = True
        for parameter in self.projection.parameters():
            parameter.requires_grad = True


def save_checkpoint(
    model: FashionEncoder,
    path: str | Path,
    *,
    image_size: int,
    training_mode: str,
    objective: str,
    metrics: dict[str, float],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "embedding_dim": model.embedding_dim,
            "image_size": image_size,
            "training_mode": training_mode,
            "objective": objective,
            "metrics": metrics,
        },
        destination,
    )


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[FashionEncoder, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = FashionEncoder(embedding_dim=int(payload["embedding_dim"]), pretrained=False)
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, payload
