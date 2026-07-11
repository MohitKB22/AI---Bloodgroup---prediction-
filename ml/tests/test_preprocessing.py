import numpy as np
import pytest

from ml.preprocessing.pipeline import FingerprintPreprocessor
from ml.tests.synthetic_data import synthetic_fingerprint


@pytest.fixture(scope="module")
def preprocessor():
    return FingerprintPreprocessor(target_size=224, block_size=16)


def test_pipeline_output_shape_and_range(preprocessor):
    img = synthetic_fingerprint(300, seed=1)
    result = preprocessor(img)

    assert result.model_input.shape == (224, 224, 3)
    assert result.model_input.dtype == np.float32
    assert not np.isnan(result.model_input).any()
    assert not np.isinf(result.model_input).any()


def test_quality_gate_ranks_degraded_image_lower(preprocessor):
    import cv2

    clean = synthetic_fingerprint(300, seed=2)
    degraded = clean.astype(np.float32)
    degraded = cv2.GaussianBlur(degraded, (9, 9), 4)
    degraded = degraded * 0.3 + 90
    degraded = np.clip(degraded + np.random.default_rng(3).normal(0, 25, degraded.shape), 0, 255).astype(np.uint8)

    clean_result = preprocessor(clean)
    degraded_result = preprocessor(degraded)

    assert degraded_result.quality.overall_score < clean_result.quality.overall_score
    assert not degraded_result.quality.is_acceptable


def test_quality_report_serializes(preprocessor):
    img = synthetic_fingerprint(280, seed=4)
    result = preprocessor(img)
    d = result.quality.to_dict()
    assert 0 <= d["overall_score"] <= 100
    assert set(d["sub_scores"].keys()) == {
        "ridge_orientation_coherence", "ridge_frequency_consistency", "foreground_coverage", "global_contrast",
    }


def test_roi_falls_back_to_full_image_on_blank_input(preprocessor):
    blank = np.full((200, 200), 128, dtype=np.uint8)  # no ridge structure at all
    result = preprocessor(blank)
    assert result.model_input.shape == (224, 224, 3)  # must not crash on a degenerate input
