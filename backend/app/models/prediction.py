from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.user import User


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

    predicted_label: Mapped[str] = mapped_column(String(8), nullable=False)
    confidence_calibrated: Mapped[float] = mapped_column(Float, nullable=False)
    confidence_raw: Mapped[float] = mapped_column(Float, nullable=False)
    raw_probabilities: Mapped[dict] = mapped_column(JSON, nullable=False)
    calibrated_probabilities: Mapped[dict] = mapped_column(JSON, nullable=False)
    quality_report: Mapped[dict] = mapped_column(JSON, nullable=False)
    ensemble_weights: Mapped[dict] = mapped_column(JSON, nullable=False)
    explainability: Mapped[list] = mapped_column(JSON, default=list)
    tta_used: Mapped[bool] = mapped_column(Boolean, default=True)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    original_filename: Mapped[str | None] = mapped_column(String(512), nullable=True)
    image_path: Mapped[str | None] = mapped_column(String(512), nullable=True)

    user: Mapped[User] = relationship(back_populates="predictions")
