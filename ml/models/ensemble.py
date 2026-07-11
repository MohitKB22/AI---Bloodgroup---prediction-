"""Ensemble of EfficientNetV2, ViT, and ConvNeXt.

Two supported strategies (see ModelConfig.ensemble_strategy):

  weighted_softmax: per-model softmax probabilities are combined with
    learnable, softmax-normalized weights. Simple, robust with few
    validation samples, and the weights themselves are interpretable
    (see `current_weights()`).

  stacking: a small MLP meta-learner is trained on the concatenated
    per-model softmax outputs. Can capture cases where one model is
    reliable only for certain classes, at the cost of needing a held-out
    stacking set to avoid overfitting the meta-learner to the base models'
    training data.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.config.config import NUM_CLASSES
from ml.models.backbones import build_backbone


class EnsembleModel(nn.Module):
    def __init__(
        self,
        backbone_names: tuple[str, ...],
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        drop_rate: float = 0.2,
        strategy: str = "weighted_softmax",
    ):
        super().__init__()
        self.backbone_names = list(backbone_names)
        self.strategy = strategy
        self.num_classes = num_classes

        self.backbones = nn.ModuleDict(
            {name: build_backbone(name, num_classes, pretrained, drop_rate) for name in backbone_names}
        )

        if strategy == "weighted_softmax":
            self.raw_weights = nn.Parameter(torch.zeros(len(backbone_names)))
        elif strategy == "stacking":
            self.meta_learner = nn.Sequential(
                nn.Linear(num_classes * len(backbone_names), 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(64, num_classes),
            )
        else:
            raise ValueError(f"Unknown ensemble strategy: {strategy}")

    def forward_individual(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Per-backbone logits, e.g. for individual-model training/eval."""
        return {name: backbone(x) for name, backbone in self.backbones.items()}

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        per_model_logits = self.forward_individual(x)
        per_model_probs = {name: F.softmax(logits, dim=-1) for name, logits in per_model_logits.items()}

        if self.strategy == "weighted_softmax":
            weights = F.softmax(self.raw_weights, dim=0)
            stacked = torch.stack([per_model_probs[n] for n in self.backbone_names], dim=0)  # (M, B, C)
            combined_probs = (weights.view(-1, 1, 1) * stacked).sum(dim=0)
        else:
            concat = torch.cat([per_model_probs[n] for n in self.backbone_names], dim=-1)
            combined_probs = F.softmax(self.meta_learner(concat), dim=-1)

        return {
            "ensemble_probs": combined_probs,
            "per_model_logits": per_model_logits,
            "per_model_probs": per_model_probs,
        }

    def current_weights(self) -> dict[str, float]:
        if self.strategy != "weighted_softmax":
            return {}
        weights = F.softmax(self.raw_weights, dim=0).detach().cpu().tolist()
        return dict(zip(self.backbone_names, weights, strict=True))

    def freeze_backbones(self) -> None:
        for backbone in self.backbones.values():
            for p in backbone.parameters():
                p.requires_grad = False

    def unfreeze_backbones(self) -> None:
        for backbone in self.backbones.values():
            for p in backbone.parameters():
                p.requires_grad = True
