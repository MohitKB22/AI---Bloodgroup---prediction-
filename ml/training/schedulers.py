"""Cosine-annealed LR schedule with a linear warmup phase.

Warmup avoids destabilizing the pretrained backbone weights with a large
LR before the newly-initialized classification head has adjusted; cosine
decay afterward gives smoother late-training convergence than step decay.
"""
from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_warmup_cosine_scheduler(
    optimizer: Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    steps_per_epoch: int,
    min_lr_ratio: float = 0.01,
) -> LambdaLR:
    warmup_steps = max(1, warmup_epochs * steps_per_epoch)
    total_steps = max(warmup_steps + 1, total_epochs * steps_per_epoch)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / warmup_steps
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(progress, 1.0)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)
