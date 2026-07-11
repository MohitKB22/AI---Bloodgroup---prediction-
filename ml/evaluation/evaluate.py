"""Produces the full evaluation report for a held-out split: classification
metrics, calibration metrics, a confusion-matrix plot, and a reliability
diagram. Intended to be run exactly once per model version against the
test set -- repeated peeking at test-set results to tune choices defeats
the purpose of holding it out in the first place.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ml.config.config import CLASS_NAMES
from ml.evaluation.calibration import expected_calibration_error
from ml.evaluation.metrics import compute_classification_report


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (held-out test set)")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_reliability_diagram(probs: np.ndarray, labels: np.ndarray, save_path: str, n_bins: int = 15) -> None:
    confidences = probs.max(axis=-1)
    predictions = probs.argmax(axis=-1)
    correctness = (predictions == labels).astype(np.float64)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers, bin_acc, bin_counts = [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:], strict=True):
        mask = (confidences > lo) & (confidences <= hi)
        bin_centers.append((lo + hi) / 2)
        bin_acc.append(correctness[mask].mean() if np.any(mask) else np.nan)
        bin_counts.append(mask.sum())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.bar(bin_centers, bin_acc, width=1.0 / n_bins, edgecolor="black", alpha=0.75, label="Observed accuracy")
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Reliability Diagram")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def full_evaluation_report(
    probs: np.ndarray,
    labels: np.ndarray,
    output_dir: str,
    split_name: str = "test",
    temperature: float | None = None,
    class_names: list[str] = CLASS_NAMES,
    extra_metadata: dict | None = None,
) -> dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = compute_classification_report(probs, labels, class_names)
    report["ece"] = expected_calibration_error(probs, labels)
    report["temperature_used"] = temperature
    report["split"] = split_name
    report["metadata"] = extra_metadata or {}
    report["disclaimer"] = (
        "These metrics describe how well the model reproduces labels in this "
        "dataset. They are not evidence that fingerprint patterns are a "
        "validated biological predictor of blood group in general -- see "
        "MODEL_CARD.md."
    )

    cm = np.array(report["confusion_matrix"])
    plot_confusion_matrix(cm, class_names, str(out_dir / f"{split_name}_confusion_matrix.png"))
    plot_reliability_diagram(probs, labels, str(out_dir / f"{split_name}_reliability_diagram.png"))

    with open(out_dir / f"{split_name}_report.json", "w") as f:
        json.dump(report, f, indent=2)

    return report
