"""Grad-CAM (Selvaraju et al., 2017) for the CNN backbones.

Hooks the backbone's last conv-stage output for both the forward
activations and the backward gradients, then weights each channel by its
gradient's global-average-pool -- channels the network relied on more for
this prediction get more weight in the resulting heatmap.

Does not apply to the ViT backbone (no spatial conv feature map); ViT
explainability instead uses attention rollout, see attention_rollout.py.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ml.evaluation.viz_utils import overlay_heatmap
from ml.models.backbones import CNNBackbone


class GradCAM:
    def __init__(self, backbone: CNNBackbone):
        self.backbone = backbone
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._forward_handle = backbone.target_layer.register_forward_hook(self._save_activations)
        self._backward_handle = backbone.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, _module, _inp, output):
        self._activations = output.detach()

    def _save_gradients(self, _module, _grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def close(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()

    def generate(self, x: torch.Tensor, class_idx: torch.Tensor | None = None) -> tuple[np.ndarray, torch.Tensor]:
        """Returns (heatmap [B,H,W] in 0..1, logits) for the given batch."""
        self.backbone.zero_grad(set_to_none=True)
        logits = self.backbone(x)

        if class_idx is None:
            class_idx = logits.argmax(dim=-1)

        selected = logits[torch.arange(logits.size(0), device=logits.device), class_idx]
        selected.sum().backward(retain_graph=False)

        if self._activations is None or self._gradients is None:
            raise RuntimeError("Grad-CAM hooks did not fire -- check that target_layer is on the forward path.")

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self._activations).sum(dim=1))  # (B, h, w)

        cam_max = cam.amax(dim=(1, 2), keepdim=True)
        cam = cam / (cam_max + 1e-8)
        return cam.cpu().numpy(), logits.detach()

    def overlay(self, heatmap: np.ndarray, display_image_gray: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        """Resizes a single heatmap to the display image's resolution and
        composites it as a color overlay for the frontend/report."""
        return overlay_heatmap(heatmap, display_image_gray, alpha)
