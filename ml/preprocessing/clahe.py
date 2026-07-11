"""Contrast Limited Adaptive Histogram Equalization for fingerprint images."""
from __future__ import annotations

import cv2
import numpy as np


def apply_clahe(
    gray_image: np.ndarray,
    clip_limit: float = 2.5,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE to a single-channel uint8 image.

    Fingerprint captures (especially from low-cost sensors or phone-camera
    photos of inked prints) often have uneven illumination. Local histogram
    equalization brings out ridge contrast in dim regions without blowing
    out already-bright regions, which a global equalization would do.
    """
    if gray_image.dtype != np.uint8:
        gray_image = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray_image)
