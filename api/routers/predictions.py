"""Prediction & History API routes (Phase 9 checklist: Prediction API,
History API). Delegates all business logic to src.services
.prediction_service / src.database.prediction_repository - this module
only translates HTTP <-> service layer (SDS Section 3, Application layer;
never touches AI-layer or ORM internals directly).
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from sqlalchemy.orm import Session

from api.dependencies import get_default_model_key, get_model_context
from api.schemas.prediction_schemas import (
    ModelKey,
    PaginatedPredictionsResponseSchema,
    PredictByPathRequestSchema,
    PredictionResponseSchema,
    PredictionSummarySchema,
)
from src.config.config_loader import load_config
from src.database.models import PredictionRecord
from src.database.prediction_repository import get_prediction_by_id, list_predictions
from src.database.session import get_db
from src.services.prediction_service import run_prediction_from_path, run_prediction_from_upload
from src.utils.logger import setup_logger

logger: logging.Logger = setup_logger("api_predictions_router")

router = APIRouter(prefix="/api/v1/predictions", tags=["predictions"])


def _to_response(record: PredictionRecord) -> PredictionResponseSchema:
    """Builds the full API response from a persisted record's full_report JSON."""
    return PredictionResponseSchema(
        **record.full_report,
        id=record.id,
        created_at=record.created_at,
        model_key=record.model_key,
    )


def _validate_upload_extension(filename: str) -> None:
    """Rejects unsupported file extensions before any processing.

    Reuses configs/data.yaml -> dataset.allowed_extensions instead of
    redefining the list in configs/api.yaml (SDS: không Duplicate Code).
    """
    data_config = load_config("data")
    allowed_extensions = {ext.lower() for ext in data_config["dataset"]["allowed_extensions"]}
    suffix = Path(filename).suffix.lower()
    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{suffix}'. Allowed: {sorted(allowed_extensions)}",
        )


@router.post(
    "",
    response_model=PredictionResponseSchema,
    status_code=status.HTTP_201_CREATED,
    summary="Predict disease from an uploaded leaf image (primary endpoint)",
)
def create_prediction_from_upload(
    request: Request,
    file: UploadFile = File(..., description="Leaf image file (jpg/png)."),
    model_key: Optional[ModelKey] = Form(
        None, description="Defaults to configs/inference.yaml -> default_model_key if omitted."
    ),
    db: Session = Depends(get_db),
) -> PredictionResponseSchema:
    """Runs prediction + decision engine on an uploaded image and persists it.

    The uploaded file is processed then deleted immediately after - it is
    never stored on disk or in the database beyond its original filename
    (quyết định chốt đầu Phase 9: "ảnh xử lý xong xóa luôn").
    """
    _validate_upload_extension(file.filename or "")

    resolved_model_key = model_key or get_default_model_key()
    context = get_model_context(request, resolved_model_key)

    record = run_prediction_from_upload(db, file, resolved_model_key, context)
    return _to_response(record)


@router.post(
    "/by-path",
    response_model=PredictionResponseSchema,
    status_code=status.HTTP_201_CREATED,
    summary="Predict disease from an image path already on the server (secondary/internal use)",
)
def create_prediction_from_path(
    request: Request,
    body: PredictByPathRequestSchema,
    db: Session = Depends(get_db),
) -> PredictionResponseSchema:
    """Runs prediction + decision engine on a server-side image path.

    Intended for internal testing/automation (e.g. re-running predictions
    on files already present in data/raw/), not for external clients -
    those should use POST /api/v1/predictions with a multipart upload.
    """
    resolved_model_key = body.model_key or get_default_model_key()
    context = get_model_context(request, resolved_model_key)

    try:
        record = run_prediction_from_path(db, Path(body.image_path), resolved_model_key, context)
    except FileNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error

    return _to_response(record)


@router.get(
    "",
    response_model=PaginatedPredictionsResponseSchema,
    summary="List past predictions (History API), newest first, with optional filters",
)
def list_prediction_history(
    page: int = Query(1, ge=1),
    page_size: Optional[int] = Query(
        None, ge=1, description="Defaults to configs/api.yaml -> pagination.default_page_size"
    ),
    predicted_class: Optional[str] = None,
    priority: Optional[str] = None,
    verification_status: Optional[str] = None,
    is_healthy: Optional[bool] = None,
    model_key: Optional[ModelKey] = None,
    created_from: Optional[datetime] = None,
    created_to: Optional[datetime] = None,
    db: Session = Depends(get_db),
) -> PaginatedPredictionsResponseSchema:
    """Lists prediction history with pagination and optional exact-match filters."""
    pagination_config = load_config("api")["pagination"]
    resolved_page_size = min(
        page_size or pagination_config["default_page_size"],
        pagination_config["max_page_size"],
    )

    records, total = list_predictions(
        db,
        page=page,
        page_size=resolved_page_size,
        predicted_class=predicted_class,
        priority=priority,
        verification_status=verification_status,
        is_healthy=is_healthy,
        model_key=model_key,
        created_from=created_from,
        created_to=created_to,
    )
    items = [PredictionSummarySchema.model_validate(record) for record in records]
    return PaginatedPredictionsResponseSchema(
        items=items, total=total, page=page, page_size=resolved_page_size
    )


@router.get(
    "/{prediction_id}",
    response_model=PredictionResponseSchema,
    summary="Get the full Decision Report for one past prediction",
)
def get_prediction_detail(
    prediction_id: int, db: Session = Depends(get_db)
) -> PredictionResponseSchema:
    """Fetches one prediction's complete Decision Report by id."""
    record = get_prediction_by_id(db, prediction_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction id={prediction_id} not found.",
        )
    return _to_response(record)
