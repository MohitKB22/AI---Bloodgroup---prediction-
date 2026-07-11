"""Fits the ensemble combination layer on out-of-fold (OOF) predictions.

Each backbone's k-fold training loop produces, for every trainval sample,
exactly one prediction from a model that did NOT see that sample during
training (the fold where it was held out as validation). Fitting the
combiner on these concatenated OOF predictions -- rather than on
in-sample predictions from the final trained models -- is what keeps
stacking honest: the combiner never sees a prediction that benefited from
that sample's label already being learned.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def fit_weighted_softmax(
    oof_probs: dict[str, np.ndarray],
    labels: np.ndarray,
    epochs: int = 300,
    lr: float = 0.05,
) -> dict[str, float]:
    names = list(oof_probs.keys())
    stacked = torch.tensor(np.stack([oof_probs[n] for n in names], axis=0), dtype=torch.float32)  # (M, N, C)
    labels_t = torch.tensor(labels, dtype=torch.long)

    raw_weights = torch.zeros(len(names), requires_grad=True)
    optimizer = torch.optim.Adam([raw_weights], lr=lr)

    for _ in range(epochs):
        optimizer.zero_grad()
        weights = F.softmax(raw_weights, dim=0)
        combined = (weights.view(-1, 1, 1) * stacked).sum(dim=0)
        loss = F.nll_loss(torch.log(combined.clamp_min(1e-12)), labels_t)
        loss.backward()
        optimizer.step()

    final_weights = F.softmax(raw_weights.detach(), dim=0)
    return dict(zip(names, (w.item() for w in final_weights), strict=True))


def fit_stacking_meta_learner(
    oof_probs: dict[str, np.ndarray],
    labels: np.ndarray,
    num_classes: int,
    epochs: int = 400,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> nn.Module:
    names = list(oof_probs.keys())
    concat = np.concatenate([oof_probs[n] for n in names], axis=-1)
    X = torch.tensor(concat, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    meta = nn.Sequential(
        nn.Linear(X.shape[1], 64),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(64, num_classes),
    )
    optimizer = torch.optim.Adam(meta.parameters(), lr=lr, weight_decay=weight_decay)

    meta.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = meta(X)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

    meta.eval()
    return meta
