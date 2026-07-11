import numpy as np
import torch

from ml.evaluation.attention_rollout import rollout_heatmap
from ml.evaluation.gradcam import GradCAM
from ml.models.backbones import CNNBackbone, ViTBackbone


def test_gradcam_heatmap_shape_and_range():
    torch.manual_seed(0)
    backbone = CNNBackbone("convnext_tiny", num_classes=8, pretrained=False)
    cam = GradCAM(backbone)
    try:
        x = torch.randn(2, 3, 224, 224)
        heatmap, logits = cam.generate(x)
        assert heatmap.shape[0] == 2
        assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0 + 1e-5
        assert logits.shape == (2, 8)
    finally:
        cam.close()


def test_gradcam_overlay_matches_display_image_size():
    backbone = CNNBackbone("efficientnetv2_rw_s", num_classes=8, pretrained=False)
    cam = GradCAM(backbone)
    try:
        x = torch.randn(1, 3, 224, 224)
        heatmap, _ = cam.generate(x)
        display = np.random.randint(0, 255, (300, 250), dtype=np.uint8)
        overlay = cam.overlay(heatmap[0], display)
        assert overlay.shape == (300, 250, 3)
    finally:
        cam.close()


def test_attention_rollout_heatmap_shape():
    torch.manual_seed(0)
    vit = ViTBackbone("vit_base_patch16_224", num_classes=8, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    vit(x)  # populate attention maps as a side effect, mirroring real usage

    heatmap = rollout_heatmap(vit, patch_grid_size=14)
    assert heatmap.shape == (2, 14, 14)
    assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0 + 1e-5


def test_attention_rollout_raises_without_prior_forward_pass():
    import pytest
    vit = ViTBackbone("vit_base_patch16_224", num_classes=8, pretrained=False)
    with pytest.raises(RuntimeError):
        rollout_heatmap(vit, patch_grid_size=14)
