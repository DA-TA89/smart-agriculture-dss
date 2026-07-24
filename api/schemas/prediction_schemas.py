"""Pydantic request/response schemas for the Prediction & History API.

Field names/nesting mirror
src.decision_engine.decision_report_builder.build_decision_report()
exactly, so the API response is a faithful, typed representation of the
real Decision Report - no renaming, no reshaping.
"""

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# Cứng 2 giá trị, khớp đúng model_factory.SUPPORTED_MODEL_KEYS - cùng cách
# argparse choices=["efficientnet", "vit"] đã dùng ở decision_pipeline.py CLI
# (Phase 8), giữ nhất quán thay vì đọc động từ config (đánh đổi lấy type
# checking/validation tĩnh của Pydantic Literal).
ModelKey = Literal["efficientnet", "vit"]


class AIPredictionSchema(BaseModel):
    predicted_class: str
    plant_type: str
    disease_name: str
    is_healthy: bool
    confidence_score: float
    confidence_tier: str


class SeverityAssessmentSchema(BaseModel):
    severity_level: str
    pathogen_type: str
    pathogen_name: str
    symptoms: str
    spread_risk: str


class TopKPredictionItemSchema(BaseModel):
    class_name: str
    confidence_score: float


class DecisionReportSchema(BaseModel):
    """Mirrors build_decision_report()'s output, plus top_k_predictions."""

    image_reference: str
    generated_at: str
    ai_prediction: AIPredictionSchema
    severity_assessment: SeverityAssessmentSchema
    priority: str
    verification_status: str
    treatment_recommendation: str
    prevention_recommendation: str
    top_k_predictions: List[TopKPredictionItemSchema]


class PredictionResponseSchema(DecisionReportSchema):
    """Full response for a single prediction (create, or get-by-id)."""

    model_config = ConfigDict(protected_namespaces=())

    id: int
    created_at: datetime
    model_key: ModelKey


class PredictionSummarySchema(BaseModel):
    """Lightweight row for the History API list endpoint (indexed columns only)."""

    model_config = ConfigDict(protected_namespaces=(), from_attributes=True)

    id: int
    created_at: datetime
    model_key: ModelKey
    image_reference: str
    predicted_class: str
    is_healthy: bool
    confidence_score: float
    severity_level: str
    priority: str
    verification_status: str


class PaginatedPredictionsResponseSchema(BaseModel):
    items: List[PredictionSummarySchema]
    total: int
    page: int
    page_size: int


class PredictByPathRequestSchema(BaseModel):
    """Request body for POST /api/v1/predictions/by-path (secondary/internal use)."""

    model_config = ConfigDict(protected_namespaces=())

    image_path: str = Field(
        ...,
        description=(
            "Absolute or server-relative path to an image file. JSON escapes "
            "backslash, so on Windows use forward slashes (e.g. "
            '"data/raw/plantvillage dataset/color/.../leaf.JPG") or double '
            'backslashes ("data\\\\raw\\\\...") - a single backslash causes a '
            "422 Unprocessable Content error."
        ),
    )
    model_key: Optional[ModelKey] = Field(
        None, description="Defaults to configs/inference.yaml -> default_model_key if omitted."
    )