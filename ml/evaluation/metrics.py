"""Standard multi-class classification metrics, computed only on held-out
data that never participated in training or model selection.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from ml.config.config import CLASS_NAMES


def macro_f1_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1).numpy()
    return float(f1_score(labels.numpy(), preds, average="macro", zero_division=0))


def compute_classification_report(
    probs: np.ndarray,
    labels: np.ndarray,
    class_names: list[str] = CLASS_NAMES,
) -> dict:
    """`probs` is (N, C) softmax/ensemble output; `labels` is (N,) int array.
    ROC-AUC is one-vs-rest, macro-averaged, and is skipped (reported as null)
    if a class is entirely absent from `labels`, since sklearn can't define
    an ROC curve for a class with zero positive examples.
    """
    preds = probs.argmax(axis=-1)

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=list(range(len(class_names))), zero_division=0
    )
    per_class = {
        class_names[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(class_names))
    }

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )

    present_classes = sorted(set(labels.tolist()))
    roc_auc_macro = None
    if len(present_classes) >= 2:
        try:
            roc_auc_macro = float(
                roc_auc_score(labels, probs, multi_class="ovr", average="macro", labels=present_classes)
            )
        except ValueError:
            roc_auc_macro = None

    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "roc_auc_macro_ovr": roc_auc_macro,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": class_names,
        "n_samples": int(len(labels)),
    }
