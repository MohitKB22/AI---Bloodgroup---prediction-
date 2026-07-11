from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class QualitySubScores(BaseModel):
    ridge_orientation_coherence: float
    ridge_frequency_consistency: float
    foreground_coverage: float
    global_contrast: float


class QualityReportSchema(BaseModel):
    overall_score: float
    sub_scores: QualitySubScores
    is_acceptable: bool
    warnings: list[str]


class ExplainabilityAssetSchema(BaseModel):
    backbone: str
    method: str
    overlay_url: str


class PredictionResponse(BaseModel):
    id: str
    predicted_label: str
    raw_probabilities: dict[str, float]
    calibrated_probabilities: dict[str, float]
    confidence_calibrated: float
    confidence_raw: float
    ensemble_weights: dict[str, float]
    quality: QualityReportSchema
    explainability: list[ExplainabilityAssetSchema]
    tta_used: bool
    model_version: str
    disclaimer: str
    created_at: datetime


class PredictionHistoryItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    predicted_label: str
    confidence_calibrated: float
    model_version: str
    created_at: datetime


class PredictionHistoryPage(BaseModel):
    items: list[PredictionHistoryItem]
    total: int
    page: int
    page_size: int
