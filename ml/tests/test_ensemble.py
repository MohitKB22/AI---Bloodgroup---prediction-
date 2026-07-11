import torch

from ml.models.ensemble import EnsembleModel

BACKBONES = ("efficientnetv2_rw_s", "vit_base_patch16_224", "convnext_tiny")


def _build(strategy):
    torch.manual_seed(0)
    return EnsembleModel(backbone_names=BACKBONES, num_classes=8, pretrained=False, strategy=strategy)


def test_weighted_softmax_output_is_a_valid_distribution():
    model = _build("weighted_softmax")
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)

    assert out["ensemble_probs"].shape == (2, 8)
    assert torch.allclose(out["ensemble_probs"].sum(dim=-1), torch.ones(2), atol=1e-4)
    assert (out["ensemble_probs"] >= 0).all()


def test_stacking_output_is_a_valid_distribution():
    model = _build("stacking")
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)

    assert out["ensemble_probs"].shape == (2, 8)
    assert torch.allclose(out["ensemble_probs"].sum(dim=-1), torch.ones(2), atol=1e-4)


def test_current_weights_sum_to_one():
    model = _build("weighted_softmax")
    weights = model.current_weights()
    assert set(weights.keys()) == set(BACKBONES)
    assert abs(sum(weights.values()) - 1.0) < 1e-5


def test_freeze_backbones_stops_gradient_flow():
    model = _build("weighted_softmax")
    model.freeze_backbones()
    assert all(not p.requires_grad for b in model.backbones.values() for p in b.parameters())

    model.unfreeze_backbones()
    assert all(p.requires_grad for b in model.backbones.values() for p in b.parameters())


def test_invalid_strategy_raises():
    import pytest
    with pytest.raises(ValueError):
        EnsembleModel(backbone_names=BACKBONES, num_classes=8, pretrained=False, strategy="not_a_real_strategy")
