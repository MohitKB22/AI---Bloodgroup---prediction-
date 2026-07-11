"""Generates a synthetic ridge-pattern image for smoke-testing purposes only.

This is NOT a substitute for real fingerprint data and must never be used to
train or evaluate the classifier -- it exists purely so the preprocessing
and inference code paths can be exercised without a real dataset present.
"""
from __future__ import annotations

import numpy as np


def synthetic_fingerprint(size: int = 300, seed: int | None = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx, cy = size * rng.uniform(0.4, 0.6), size * rng.uniform(0.4, 0.6)
    dx, dy = xx - cx, yy - cy
    r = np.sqrt(dx ** 2 + dy ** 2)
    theta = np.arctan2(dy, dx)

    # Concentric, slightly perturbed ridges radiating from a core point,
    # loosely mimicking a whorl/loop pattern.
    wavelength = rng.uniform(6, 10)
    warp = 6.0 * np.sin(3 * theta + rng.uniform(0, 6.28))
    ridges = 0.5 + 0.5 * np.sin((r + warp) * (2 * np.pi / wavelength))

    img = (ridges * 255).astype(np.uint8)
    noise = rng.normal(0, 12, size=img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Vignette so segmentation has an actual foreground/background boundary
    # to find, like a real sensor capture would.
    vignette = np.clip(1.2 - r / (size * 0.65), 0, 1)
    background = rng.integers(150, 180)
    img = (img * vignette + background * (1 - vignette)).astype(np.uint8)
    return img
