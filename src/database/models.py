"""SQLAlchemy ORM models for persisted prediction/decision records.

Per the hybrid schema decision (chốt đầu Phase 9, xem PHASE9_LOG.md):
frequently-filtered fields are stored as indexed columns for fast querying
(used by the History API, and later Phase 10 Dashboard analytics), while
the complete Decision Report (see
src.decision_engine.decision_report_builder.build_decision_report()) is
also stored as-is in a JSONB column - so no information is ever lost or
needs to be reconstructed from the individual columns.
"""

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from src.database.session import Base


class PredictionRecord(Base):
    """One row per prediction served by the Prediction / History API."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True, nullable=False
    )

    # --- Request metadata (không thuộc dict Decision Report, gắn thêm ở tầng API) ---
    model_key: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    image_reference: Mapped[str] = mapped_column(String(512), nullable=False)

    # --- Cột index, nhân bản từ Decision Report để History API / Phase 10
    #     Dashboard query nhanh (quyết định "hybrid schema" đầu Phase 9) ---
    predicted_class: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    plant_type: Mapped[str] = mapped_column(String(64), nullable=False)
    disease_name: Mapped[str] = mapped_column(String(128), nullable=False)
    is_healthy: Mapped[bool] = mapped_column(Boolean, index=True, nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, index=True, nullable=False)
    confidence_tier: Mapped[str] = mapped_column(String(16), nullable=False)
    severity_level: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    priority: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    verification_status: Mapped[str] = mapped_column(String(32), index=True, nullable=False)

    # --- Decision Report đầy đủ, nguyên trạng (nguồn sự thật cho trang chi tiết) ---
    full_report: Mapped[dict] = mapped_column(JSONB, nullable=False)
