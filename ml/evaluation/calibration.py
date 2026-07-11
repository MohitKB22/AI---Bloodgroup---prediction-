"""Temperature scaling (Guo et al., 2017) for confidence calibration.

Fits a single scalar T > 0 that divides the logits before softmax, chosen
to minimize NLL on a held-out calibration split (never the test set, and
never training data -- fitting T on training data would just learn to undo
label smoothing rather than correct genuine miscalibration). T > 1 softens
overconfident predictions; T < 1 sharpens underconfident ones. This
reshapes confidence values so they're closer to true empirical accuracy;
it does not, and cannot, improve the model's actual discriminative
accuracy or validate that the underlying prediction task is meaningful.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class CalibrationResult:
    temperature: float
    pre_calibration_nll: float
    post_calibration_nll: float
    pre_calibration_ece: float
    post_calibration_ece: float


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 200, lr: float = 0.01) -> float:
    logits = logits.detach()
    labels = labels.detach()
    log_temperature = torch.zeros(1, requires_grad=True)  # optimize in log-space to keep T > 0
    optimizer = torch.optim.LBFGS([log_temperature], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp()
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.exp().item())


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return F.softmax(logits / temperature, dim=-1)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """ECE: bins predictions by confidence, compares mean confidence to
    empirical accuracy within each bin, weights by bin size."""
    confidences = probs.max(axis=-1)
    predictions = probs.argmax(axis=-1)
    correctness = (predictions == labels).astype(np.float64)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:], strict=True):
        mask = (confidences > lo) & (confidences <= hi)
        if not np.any(mask):
            continue
        bin_confidence = confidences[mask].mean()
        bin_accuracy = correctness[mask].mean()
        ece += (mask.sum() / len(confidences)) * abs(bin_confidence - bin_accuracy)
    return float(ece)


def calibrate_and_report(
    val_logits: torch.Tensor,
    val_labels: torch.Tensor,
) -> CalibrationResult:
    temperature = fit_temperature(val_logits, val_labels)

    pre_probs = F.softmax(val_logits, dim=-1)
    post_probs = apply_temperature(val_logits, temperature)

    pre_nll = float(F.nll_loss(torch.log(pre_probs.clamp_min(1e-12)), val_labels))
    post_nll = float(F.nll_loss(torch.log(post_probs.clamp_min(1e-12)), val_labels))

    labels_np = val_labels.numpy()
    pre_ece = expected_calibration_error(pre_probs.numpy(), labels_np)
    post_ece = expected_calibration_error(post_probs.numpy(), labels_np)

    return CalibrationResult(
        temperature=temperature,
        pre_calibration_nll=pre_nll,
        post_calibration_nll=post_nll,
        pre_calibration_ece=pre_ece,
        post_calibration_ece=post_ece,
    )
