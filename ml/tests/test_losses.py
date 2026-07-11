import torch

from ml.models.losses import FocalLossWithLabelSmoothing


def test_focal_loss_produces_finite_gradient():
    torch.manual_seed(0)
    logits = torch.randn(16, 8, requires_grad=True)
    targets = torch.randint(0, 8, (16,))

    loss_fn = FocalLossWithLabelSmoothing(num_classes=8, gamma=2.0, label_smoothing=0.1)
    loss = loss_fn(logits, targets)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_focal_loss_alpha_weighting_changes_value():
    torch.manual_seed(0)
    logits = torch.randn(16, 8)
    targets = torch.randint(0, 8, (16,))

    unweighted = FocalLossWithLabelSmoothing(num_classes=8)(logits, targets)
    alpha = torch.tensor([2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 1.0, 1.0])
    weighted = FocalLossWithLabelSmoothing(num_classes=8, alpha=alpha)(logits, targets)

    assert not torch.isclose(unweighted, weighted)


def test_label_smoothing_rejects_invalid_range():
    import pytest as _pytest
    with _pytest.raises(ValueError):
        FocalLossWithLabelSmoothing(num_classes=8, label_smoothing=1.0)
