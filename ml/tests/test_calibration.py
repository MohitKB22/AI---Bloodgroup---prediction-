import torch

from ml.evaluation.calibration import calibrate_and_report, expected_calibration_error


def test_temperature_scaling_softens_genuine_overconfidence():
    torch.manual_seed(1)
    n, c = 300, 8
    labels = torch.randint(0, c, (n,))

    base = torch.randn(n, c) * 0.1
    picked = torch.where(torch.rand(n) < 0.4, labels, torch.randint(0, c, (n,)))
    logits = base.clone()
    logits[torch.arange(n), picked] += 15.0  # near-certain on a mostly-wrong class -> classic overconfidence

    result = calibrate_and_report(logits, labels)

    assert result.temperature > 1.0
    assert result.post_calibration_ece < result.pre_calibration_ece
    assert result.post_calibration_nll <= result.pre_calibration_nll + 1e-4


def test_temperature_scaling_does_not_change_argmax():
    """Calibration should reshape confidence, not flip predictions."""
    torch.manual_seed(2)
    logits = torch.randn(50, 8) * 3
    labels = torch.randint(0, 8, (50,))

    from ml.evaluation.calibration import apply_temperature, fit_temperature
    t = fit_temperature(logits, labels)
    calibrated = apply_temperature(logits, t)

    assert torch.equal(logits.argmax(dim=-1), calibrated.argmax(dim=-1))


def test_ece_is_zero_for_perfectly_calibrated_predictions():
    import numpy as np
    rng = np.random.default_rng(0)
    n = 5000
    # Predictions where, by construction, "confidence c" is correct with
    # probability exactly c: P(true_label == 1 | confidence = c) = c.
    confidences = rng.uniform(0.5, 1.0, size=n)
    true_label = (rng.uniform(size=n) < confidences).astype(int)
    probs = np.stack([1 - confidences, confidences], axis=1)

    ece = expected_calibration_error(probs, true_label, n_bins=10)
    assert ece < 0.05  # small residual is expected sampling noise, not miscalibration
