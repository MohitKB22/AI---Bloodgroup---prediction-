"""End-to-end fingerprint preprocessing pipeline.

Order: grayscale -> denoise -> CLAHE -> segment -> extract ROI ->
ridge-enhance -> quality-assess -> resize/normalize for model input.

Denoising and CLAHE run before segmentation because the block-variance
segmenter is sensitive to sensor noise; ridge enhancement runs on the ROI
crop (not the full frame) so the Gabor filter bank isn't wasted on
background.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ml.preprocessing.clahe import apply_clahe
from ml.preprocessing.denoise import denoise_fingerprint
from ml.preprocessing.quality import QualityReport, assess_quality
from ml.preprocessing.ridge_enhancement import RidgeFields, enhance_ridges
from ml.preprocessing.roi_extraction import ROI, extract_roi
from ml.preprocessing.segmentation import segment_foreground


@dataclass
class PreprocessResult:
    model_input: np.ndarray  # float32, normalized, shape (H, W, 3), ready to batch
    enhanced_display: np.ndarray  # uint8 grayscale, for UI / Grad-CAM overlay
    roi: ROI
    quality: QualityReport
    ridge_fields: RidgeFields


class FingerprintPreprocessor:
    def __init__(self, target_size: int = 224, block_size: int = 16):
        self.target_size = target_size
        self.block_size = block_size

    def __call__(self, bgr_or_gray_image: np.ndarray) -> PreprocessResult:
        gray = self._to_gray(bgr_or_gray_image)

        denoised = denoise_fingerprint(gray)
        contrast_enhanced = apply_clahe(denoised)

        mask = segment_foreground(contrast_enhanced, block_size=self.block_size)
        roi_crop, roi = extract_roi(contrast_enhanced, mask, padding_fraction=0.08)

        roi_crop = self._pad_to_min_size(roi_crop, self.block_size * 4)
        enhanced, ridge_fields = enhance_ridges(roi_crop, block_size=self.block_size)

        quality = assess_quality(roi_crop, ridge_fields, roi.coverage_fraction)

        model_input = self.to_model_input(enhanced)

        return PreprocessResult(
            model_input=model_input,
            enhanced_display=enhanced,
            roi=roi,
            quality=quality,
            ridge_fields=ridge_fields,
        )

    def _to_gray(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def _pad_to_min_size(self, img: np.ndarray, min_size: int) -> np.ndarray:
        h, w = img.shape[:2]
        pad_h = max(0, min_size - h)
        pad_w = max(0, min_size - w)
        if pad_h == 0 and pad_w == 0:
            return img
        return cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

    def to_model_input(self, enhanced_gray: np.ndarray) -> np.ndarray:
        """Resize + ImageNet-normalize an already-enhanced grayscale image.
        Public so callers (e.g. the training Dataset) can insert augmentation
        between ridge enhancement and normalization without repeating the
        expensive segmentation/ridge-enhancement steps per epoch."""
        resized = cv2.resize(enhanced_gray, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
        # ImageNet mean/std normalization since all three backbones use
        # ImageNet-pretrained weights.
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return (rgb - mean) / std
