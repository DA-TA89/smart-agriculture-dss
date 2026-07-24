"""Business service wiring pipelines.decision_pipeline (AI + Business
layers) together with src.database (Infrastructure layer) for the REST
API (SDS Section 3: Application layer calls Business layer, never AI/Infra
directly). api/routers/predictions.py must call only the functions in
this module - never src.database.prediction_repository or
pipelines.decision_pipeline directly.
"""

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import UploadFile
from sqlalchemy.orm import Session

from pipelines.decision_pipeline import predict_and_decide
from src.database.models import PredictionRecord
from src.database.prediction_repository import create_prediction_record
from src.utils.logger import setup_logger

logger: logging.Logger = setup_logger("prediction_service")


def run_prediction_from_path(
    db: Session, image_path: Path, model_key: str, context: Dict[str, Any]
) -> PredictionRecord:
    """Runs prediction + decision engine for an image already on disk.

    Args:
        db: Active SQLAlchemy session.
        image_path: Path to an existing image file on the server.
        model_key: Model context key ("efficientnet"/"vit") - used only to
            tag the persisted record; ``context`` must already match it.
        context: Preloaded pipeline context for model_key (see
            api.dependencies.get_model_context / decision_pipeline
            .load_pipeline_context()).

    Returns:
        The persisted PredictionRecord.

    Raises:
        FileNotFoundError: If image_path does not exist.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image path not found on server: {image_path}")

    decision_report = predict_and_decide(image_path, context)
    return create_prediction_record(db, model_key, decision_report)


def run_prediction_from_upload(
    db: Session, upload_file: UploadFile, model_key: str, context: Dict[str, Any]
) -> PredictionRecord:
    """Runs prediction + decision engine for an uploaded image.

    The uploaded file is written to a temporary path, processed, then
    deleted unconditionally - images are never persisted to disk or
    database (quyết định chốt đầu Phase 9, khớp đúng docstring sẵn có của
    decision_report_builder.py: "the image itself is not stored in the
    report"). Only the original filename is kept, as ``image_reference``,
    for traceability in the History API.

    Args:
        db: Active SQLAlchemy session.
        upload_file: FastAPI/Starlette UploadFile from a multipart request.
        model_key: Model context key ("efficientnet"/"vit").
        context: Preloaded pipeline context for model_key.

    Returns:
        The persisted PredictionRecord.
    """
    original_name = upload_file.filename or "uploaded_image"
    suffix = Path(original_name).suffix

    temp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            shutil.copyfileobj(upload_file.file, temp_file)
            temp_path = Path(temp_file.name)

        # Handle đã đóng ở đây (ra khỏi khối `with`) - an toàn để PIL mở lại
        # file trên Windows (tránh lỗi file đang bị khóa bởi tiến trình khác).
        decision_report = predict_and_decide(temp_path, context)

        # Ghi đè image_reference bằng tên file gốc do client gửi lên, thay vì
        # đường dẫn file tạm vô nghĩa (vd. "/tmp/tmpabc123.jpg") - để History
        # API hiển thị tên ảnh hữu ích cho người dùng cuối.
        decision_report["image_reference"] = original_name

        return create_prediction_record(db, model_key, decision_report)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
            logger.info("Temporary upload file deleted: %s", temp_path)
