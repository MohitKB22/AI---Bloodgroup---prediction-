"""Backbone factory: wraps timm pretrained models with a uniform interface.

Each backbone exposes:
  - forward(x) -> logits, shape (B, num_classes)
  - forward_features(x) -> spatial feature map, for Grad-CAM (CNN backbones)
  - get_attention(x) -> last-block attention weights, for ViT explainability
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn


class CNNBackbone(nn.Module):
    """Wraps a conv-based timm model (EfficientNetV2, ConvNeXt) for
    classification + Grad-CAM. `target_layer` is the module whose output
    activations/gradients Grad-CAM will hook.
    """

    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True, drop_rate: float = 0.2):
        super().__init__()
        self.model_name = model_name
        self.net = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate
        )
        self.target_layer = self._locate_last_conv_block()

    def _locate_last_conv_block(self) -> nn.Module:
        # timm's feature_info gives us the last stage's module name without
        # hardcoding architecture-specific internals.
        feature_extractor = timm.create_model(self.model_name, pretrained=False, features_only=True)
        last_module_name = feature_extractor.feature_info.module_name(-1)
        module = dict(self.net.named_modules())[last_module_name]
        return module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ViTBackbone(nn.Module):
    """Wraps a timm Vision Transformer. Grad-CAM doesn't apply directly (no
    spatial conv feature map), so explainability instead uses attention
    rollout over the transformer's self-attention weights.
    """

    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True, drop_rate: float = 0.2):
        super().__init__()
        self.model_name = model_name
        self.net = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate
        )
        self._attn_weights: list[torch.Tensor] = []
        self._register_attention_hooks()

    def _register_attention_hooks(self) -> None:
        for block in self.net.blocks:
            block.attn.fused_attn = False  # forces eager attention so softmax weights are materialized

            def hook(_module, _inp, out, store=self._attn_weights):
                store.append(out)

            block.attn.register_forward_hook(self._capture_softmax)

    def _capture_softmax(self, module, inp, out):
        # Recompute the attention softmax explicitly since timm's fused attention
        # path doesn't expose it. Cheap relative to the rest of the forward pass.
        x = inp[0]
        B, N, C = x.shape
        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
        q, k, _ = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * module.scale
        attn = attn.softmax(dim=-1)
        self._attn_weights.append(attn.detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._attn_weights = []
        return self.net(x)

    def get_attention_maps(self) -> list[torch.Tensor]:
        """Returns per-block attention tensors from the most recent forward pass,
        each shaped (B, heads, tokens, tokens)."""
        return self._attn_weights


def build_backbone(model_name: str, num_classes: int, pretrained: bool = True, drop_rate: float = 0.2) -> nn.Module:
    if "vit" in model_name:
        return ViTBackbone(model_name, num_classes, pretrained, drop_rate)
    return CNNBackbone(model_name, num_classes, pretrained, drop_rate)
