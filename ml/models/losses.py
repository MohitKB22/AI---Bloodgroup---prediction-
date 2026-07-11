"""Focal loss with label smoothing, plus optional per-class alpha weighting.

Blood-type distribution in most populations is naturally imbalanced (O+ and
A+ are far more common than AB-), so `alpha` should generally be set
inversely proportional to training-set class frequency -- see
`compute_class_alpha` in ml/data/splits.py.

Label smoothing softens the target distribution to reduce overconfidence;
the focal term additionally down-weights samples the model already
classifies confidently, which keeps rare, hard classes contributing
meaningful gradient throughout training instead of being drowned out by
easy majority-class examples.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLossWithLabelSmoothing(nn.Module):
    def __init__(
        self,
        num_classes: int,
        gamma: float = 2.0,
        label_smoothing: float = 0.1,
        alpha: torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        if not (0.0 <= label_smoothing < 1.0):
            raise ValueError("label_smoothing must be in [0, 1)")
        self.num_classes = num_classes
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.register_buffer("alpha", alpha if alpha is not None else None, persistent=False)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp(min=1e-7, max=1.0 - 1e-7)
        focal_weight = (1.0 - pt).pow(self.gamma)

        with torch.no_grad():
            smooth_dist = torch.full_like(log_probs, self.label_smoothing / (self.num_classes - 1))
            smooth_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        smoothed_ce = -(smooth_dist * log_probs).sum(dim=-1)
        loss = focal_weight * smoothed_ce

        if self.alpha is not None:
            loss = loss * self.alpha.to(logits.device).gather(0, targets)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
