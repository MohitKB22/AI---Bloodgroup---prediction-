"""Shared heatmap-to-image overlay helper used by both Grad-CAM and
attention-rollout visualizations, so the two explainability methods produce
visually consistent output for the frontend and PDF reports."""
from __future__ import annotations

import cv2
import numpy as np


def overlay_heatmap(heatmap: np.ndarray, display_image_gray: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    h, w = display_image_gray.shape[:2]
    resized = cv2.resize(heatmap.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    resized = np.clip(resized, 0, 1)
    heat_color = cv2.applyColorMap((resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    base_bgr = cv2.cvtColor(display_image_gray, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(heat_color, alpha, base_bgr, 1 - alpha, 0)
