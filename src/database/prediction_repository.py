"""CRUD access to PredictionRecord - the only module allowed to build
SQLAlchemy queries against the ``predictions`` table (Low Coupling, SDS
Section 2). Callers (src.services.prediction_service, api.routers.*) must
go through these functions instead of importing PredictionRecord/Session
query internals directly.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.database.models import PredictionRecord
from src.utils.logger import setup_logger

logger: logging.Logger = setup_logger("prediction_repository")


def create_prediction_record(
    db: Session, model_key: str, decision_report: Dict[str, Any]
) -> PredictionRecord:
    """Persists one Decision Report as a new predictions row.

    Args:
        db: Active SQLAlchemy session (see src.database.session.get_db).
        model_key: Model used for this prediction ("efficientnet"/"vit").
        decision_report: Output of pipelines.decision_pipeline
            .predict_and_decide() (includes "top_k_predictions").

    Returns:
        The newly created, refreshed PredictionRecord (id/created_at
        populated by the database after commit).
    """
    prediction = decision_report["ai_prediction"]
    severity = decision_report["severity_assessment"]

    record = PredictionRecord(
        model_key=model_key,
        image_reference=decision_report["image_reference"],
        predicted_class=prediction["predicted_class"],
        plant_type=prediction["plant_type"],
        disease_name=prediction["disease_name"],
        is_healthy=prediction["is_healthy"],
        confidence_score=prediction["confidence_score"],
        confidence_tier=prediction["confidence_tier"],
        severity_level=severity["severity_level"],
        priority=decision_report["priority"],
        verification_status=decision_report["verification_status"],
        full_report=decision_report,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    logger.info(
        "Prediction record #%d persisted (model_key=%s, predicted_class=%s)",
        record.id,
        model_key,
        record.predicted_class,
    )
    return record


def get_prediction_by_id(db: Session, prediction_id: int) -> Optional[PredictionRecord]:
    """Fetches one prediction record by primary key.

    Args:
        db: Active SQLAlchemy session.
        prediction_id: Value of PredictionRecord.id.

    Returns:
        The matching record, or None if not found.
    """
    return db.get(PredictionRecord, prediction_id)


def list_predictions(
    db: Session,
    page: int,
    page_size: int,
    predicted_class: Optional[str] = None,
    priority: Optional[str] = None,
    verification_status: Optional[str] = None,
    is_healthy: Optional[bool] = None,
    model_key: Optional[str] = None,
    created_from: Optional[datetime] = None,
    created_to: Optional[datetime] = None,
) -> Tuple[List[PredictionRecord], int]:
    """Lists prediction records, newest first, with optional exact-match filters.

    All filters are optional and combined with AND when provided. Each
    filter maps 1:1 to an indexed column (see src.database.models) so
    every supported filter runs as an indexed lookup, not a table scan.

    Args:
        db: Active SQLAlchemy session.
        page: 1-indexed page number.
        page_size: Number of rows per page.
        predicted_class: Exact class_name filter.
        priority: Exact priority filter.
        verification_status: Exact verification_status filter.
        is_healthy: Filter healthy vs. diseased predictions.
        model_key: Exact model_key filter ("efficientnet"/"vit").
        created_from: Only rows created at or after this timestamp.
        created_to: Only rows created at or before this timestamp.

    Returns:
        Tuple of (page of records, total matching row count before paging).
    """
    query = select(PredictionRecord)
    count_query = select(func.count()).select_from(PredictionRecord)

    filters = []
    if predicted_class is not None:
        filters.append(PredictionRecord.predicted_class == predicted_class)
    if priority is not None:
        filters.append(PredictionRecord.priority == priority)
    if verification_status is not None:
        filters.append(PredictionRecord.verification_status == verification_status)
    if is_healthy is not None:
        filters.append(PredictionRecord.is_healthy == is_healthy)
    if model_key is not None:
        filters.append(PredictionRecord.model_key == model_key)
    if created_from is not None:
        filters.append(PredictionRecord.created_at >= created_from)
    if created_to is not None:
        filters.append(PredictionRecord.created_at <= created_to)

    for condition in filters:
        query = query.where(condition)
        count_query = count_query.where(condition)

    total = db.scalar(count_query) or 0

    query = (
        query.order_by(PredictionRecord.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    records = list(db.scalars(query).all())
    return records, total
