"""Custom face-embedding CNN (Model A) and the heads used only during training.

The network must be importable from the backend with this same module path
(lib.models.custom_cnn) so that a full-model .pth can be unpickled by
FaceService._load_model. Exporting to ONNX (export_onnx) avoids that dependency.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class ConvBlock(nn.Module):
    """Two 3x3 conv + BatchNorm + ReLU layers, then 2x2 max pooling (halves the resolution)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class CustomFaceCNN(nn.Module):
    """Backbone of ConvBlocks, global average pooling and a linear embedding layer.

    Input: (N, 3, H, W) float32 RGB scaled to [-1, 1], as produced by
    FaceService.preprocess_face. Output: (N, embedding_dim) embedding without
    L2 normalization (FaceService normalizes it).
    """

    def __init__(
        self,
        channels: tuple[int, ...] = (32, 64, 128, 256),
        embedding_dim: int = 512,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        in_channels = 3
        for out_channels in channels:
            blocks.append(ConvBlock(in_channels, out_channels))
            in_channels = out_channels
        self.backbone = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_channels, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
        )
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(self.pool(self.backbone(x)))


class SoftmaxHead(nn.Module):
    """Linear classifier over the training identities, for cross-entropy training."""

    def __init__(self, embedding_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, embedding: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        return self.fc(embedding)


class ArcFaceHead(nn.Module):
    """Additive angular margin head (ArcFace, Deng et al. 2019).

    Logits are scale * cos(theta) for every class, with the margin added to the
    angle of the true class: scale * cos(theta + margin).
    """

    def __init__(self, embedding_dim: int, num_classes: int, scale: float = 64.0, margin: float = 0.5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.margin = margin
        # Beyond theta = pi - margin, cos(theta + margin) stops decreasing: use a linear penalty instead
        self.threshold = math.cos(math.pi - margin)
        self.fallback = math.sin(math.pi - margin) * margin

    def forward(self, embedding: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        cosine = F.linear(F.normalize(embedding), F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
        if labels is None:
            return self.scale * cosine
        target = torch.cos(torch.acos(cosine) + self.margin)
        target = torch.where(cosine > self.threshold, target, cosine - self.fallback)
        is_true_class = F.one_hot(labels, num_classes=cosine.size(1)).bool()
        return self.scale * torch.where(is_true_class, target, cosine)


def export_onnx(model: CustomFaceCNN, path: Path | str, face_size: int = 112) -> None:
    """Export the embedding network (without training head) to ONNX with a dynamic batch axis."""
    model.eval()
    dummy = torch.zeros(1, 3, face_size, face_size)
    # dynamo=False: the default exporter in torch 2.11 requires onnxscript, which is not a dependency
    torch.onnx.export(
        model,
        dummy,
        str(path),
        dynamo=False,
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
    )
