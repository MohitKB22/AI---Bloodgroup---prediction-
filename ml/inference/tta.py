"""Test-time augmentation (TTA): builds a small batch of mildly-perturbed
views of one input image so the model's prediction is averaged over them,
which tends to reduce variance from any single unlucky crop/rotation. The
actual model forward pass happens in the caller (ml/inference/predictor.py)
so TTA stays a plain batching utility, not a second place that knows about
model internals.
"""
from __future__ import annotations

import numpy as np
import torch

from ml.data.augmentation import apply_augmentation, build_tta_transforms
from ml.preprocessing.pipeline import FingerprintPreprocessor


def build_tta_batch(
    preprocessor: FingerprintPreprocessor,
    enhanced_gray_image: np.ndarray,
    num_augments: int,
) -> torch.Tensor:
    transforms = build_tta_transforms(num_augments)
    tensors = []
    for transform in transforms:
        augmented = apply_augmentation(transform, enhanced_gray_image)
        model_input = preprocessor.to_model_input(augmented)
        tensors.append(torch.from_numpy(model_input.transpose(2, 0, 1)).float())
    return torch.stack(tensors, dim=0)  # (T, 3, H, W)


def aggregate_tta_probs(probs_batch: torch.Tensor) -> torch.Tensor:
    """`probs_batch`: (T, C) -> mean probability vector (C,)."""
    return probs_batch.mean(dim=0)
