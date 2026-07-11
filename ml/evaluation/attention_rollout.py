"""Attention rollout (Abnar & Zuidema, 2020) for the ViT backbone.

Grad-CAM needs a spatial conv feature map, which a Vision Transformer
doesn't have. Attention rollout instead traces how much each patch token's
information could have reached the CLS token by the final layer, by
composing per-layer attention matrices (with a residual term added, since
each transformer block also passes its input through unchanged) across all
blocks. The CLS row of the resulting matrix gives one relevance score per
patch, which reshapes directly onto the image's patch grid.
"""
from __future__ import annotations

import numpy as np
import torch

from ml.evaluation.viz_utils import overlay_heatmap
from ml.models.backbones import ViTBackbone


def compute_attention_rollout(attention_maps: list[torch.Tensor]) -> torch.Tensor:
    """`attention_maps`: list of (B, heads, N, N) tensors, one per transformer
    block, as captured by ViTBackbone.get_attention_maps(). Returns (B, N, N).
    """
    batch_size, _, num_tokens, _ = attention_maps[0].shape
    device = attention_maps[0].device
    identity = torch.eye(num_tokens, device=device).unsqueeze(0).expand(batch_size, -1, -1)

    rollout = identity.clone()
    for attn in attention_maps:
        avg_heads = attn.mean(dim=1)  # average over heads -> (B, N, N)
        with_residual = 0.5 * avg_heads + 0.5 * identity
        with_residual = with_residual / with_residual.sum(dim=-1, keepdim=True)
        rollout = with_residual @ rollout

    return rollout


def rollout_heatmap(vit_backbone: ViTBackbone, patch_grid_size: int) -> np.ndarray:
    """Returns a (B, grid, grid) array of CLS-to-patch relevance, normalized
    to 0..1 per sample. Call this right after a forward pass through the
    same ViTBackbone instance, since it reads the attention maps captured
    during that forward call.
    """
    attention_maps = vit_backbone.get_attention_maps()
    if not attention_maps:
        raise RuntimeError("No attention maps captured -- run a forward pass through this backbone first.")

    rollout = compute_attention_rollout(attention_maps)  # (B, N, N)
    cls_to_patches = rollout[:, 0, 1:]  # drop the CLS-to-CLS entry, keep CLS-to-patch relevance

    batch_size = cls_to_patches.size(0)
    heatmap = cls_to_patches.reshape(batch_size, patch_grid_size, patch_grid_size)
    heatmap = heatmap / (heatmap.amax(dim=(1, 2), keepdim=True) + 1e-8)
    return heatmap.cpu().numpy()


def overlay(heatmap: np.ndarray, display_image_gray: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    return overlay_heatmap(heatmap, display_image_gray, alpha)
