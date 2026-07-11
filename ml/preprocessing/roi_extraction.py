"""Crop the fingerprint region of interest from a segmentation mask."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ROI:
    x: int
    y: int
    width: int
    height: int
    coverage_fraction: float  # fraction of the crop that is actual foreground


def extract_roi(
    image: np.ndarray,
    mask: np.ndarray,
    padding_fraction: float = 0.08,
) -> tuple[np.ndarray, ROI]:
    """Crop to the mask's bounding box with a small margin, falling back to
    the full image if segmentation found no usable foreground (e.g. a blank
    or fully saturated capture).
    """
    ys, xs = np.where(mask > 0)
    h, w = image.shape[:2]

    if len(xs) == 0 or len(ys) == 0:
        roi = ROI(x=0, y=0, width=w, height=h, coverage_fraction=0.0)
        return image.copy(), roi

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    pad_x = int((x1 - x0) * padding_fraction)
    pad_y = int((y1 - y0) * padding_fraction)

    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(w, x1 + pad_x)
    y1 = min(h, y1 + pad_y)

    crop = image[y0:y1, x0:x1]
    crop_mask = mask[y0:y1, x0:x1]
    coverage = float(np.mean(crop_mask > 0)) if crop_mask.size else 0.0

    roi = ROI(x=int(x0), y=int(y0), width=int(x1 - x0), height=int(y1 - y0), coverage_fraction=coverage)
    return crop, roi
