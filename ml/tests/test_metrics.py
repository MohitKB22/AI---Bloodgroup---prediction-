import numpy as np
import torch

from ml.evaluation.metrics import compute_classification_report, macro_f1_from_logits


def test_perfect_predictions_score_1():
    n, c = 80, 8
    labels = np.random.default_rng(0).integers(0, c, n)
    probs = np.zeros((n, c))
    probs[np.arange(n), labels] = 1.0

    report = compute_classification_report(probs, labels)
    assert report["accuracy"] == 1.0
    assert report["macro_f1"] == 1.0


def test_missing_class_handled_without_crash():
    """A class absent from a small test split must not raise -- support=0,
    metrics=0 for that class, and the run still completes."""
    n, c = 20, 8
    labels = np.zeros(n, dtype=int)  # only class 0 present
    labels[:5] = 1
    probs = np.random.default_rng(1).dirichlet(np.ones(c), size=n)

    report = compute_classification_report(probs, labels)
    assert report["per_class"]["AB+"]["support"] == 0
    assert "confusion_matrix" in report


def test_macro_f1_from_logits_matches_report():
    torch.manual_seed(0)
    logits = torch.randn(40, 8)
    labels = torch.randint(0, 8, (40,))

    f1_direct = macro_f1_from_logits(logits, labels)
    probs = torch.softmax(logits, dim=-1).numpy()
    report = compute_classification_report(probs, labels.numpy())

    assert abs(f1_direct - report["macro_f1"]) < 1e-6
