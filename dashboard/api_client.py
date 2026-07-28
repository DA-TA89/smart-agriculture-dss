"""Thin HTTP client for the Prediction & History API (Phase 9).

Dashboard (Presentation layer, SDS Section 3) talks to the API only over
HTTP - it never imports src.services / src.database / src.decision_engine
directly (quyết định chốt đầu Phase 10). This module has no Streamlit
dependency on purpose, so it stays reusable/testable on its own; caching
is applied by the pages that call it (see dashboard/pages/*.py).

ASSUMPTION (chưa xem trực tiếp api/routers/health.py, chỉ dựa theo mô tả
"GET /health - liveness check" trong PHASE9_LOG.md): endpoint không có
prefix "/api/v1". Nếu thực tế khác, chỉ cần sửa 1 dòng trong check_health().
"""

from typing import Any, Optional

import httpx

from src.config.config_loader import load_config

# api.yaml khai báo host "0.0.0.0" (bind-all - đúng cho phía server), nhưng
# client không bao giờ connect được tới 0.0.0.0 như một địa chỉ đích - phải
# trỏ về loopback thay thế.
_BIND_ALL_HOST = "0.0.0.0"
_LOCAL_CLIENT_HOST = "localhost"


class PredictionAPIError(RuntimeError):
    """Raised when the API returns an error response; message = server detail."""


def get_api_base_url() -> str:
    """Builds the API base URL from configs/api.yaml (+ optional dashboard.yaml override)."""
    override_url = load_config("dashboard")["api_client"]["base_url_override"]
    if override_url:
        return str(override_url).rstrip("/")

    api_section = load_config("api")["api"]
    host = api_section["host"]
    if host == _BIND_ALL_HOST:
        host = _LOCAL_CLIENT_HOST
    return f"http://{host}:{api_section['port']}"


def _get_timeout_seconds() -> float:
    return load_config("dashboard")["api_client"]["request_timeout_seconds"]


def _raise_for_status(response: httpx.Response) -> None:
    if response.is_error:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise PredictionAPIError(f"{response.status_code}: {detail}")


def check_health() -> bool:
    """GET /health - True nếu API đang chạy và phản hồi OK."""
    try:
        response = httpx.get(f"{get_api_base_url()}/health", timeout=_get_timeout_seconds())
        return response.status_code == 200
    except httpx.HTTPError:
        return False


def predict_from_upload(
    filename: str, file_bytes: bytes, content_type: str, model_key: Optional[str]
) -> dict[str, Any]:
    """POST /api/v1/predictions - multipart upload, trả về Decision Report đầy đủ."""
    files = {"file": (filename, file_bytes, content_type)}
    data = {"model_key": model_key} if model_key else {}
    response = httpx.post(
        f"{get_api_base_url()}/api/v1/predictions",
        files=files,
        data=data,
        timeout=_get_timeout_seconds(),
    )
    _raise_for_status(response)
    return response.json()


def list_predictions(page: int = 1, page_size: Optional[int] = None, **filters: Any) -> dict[str, Any]:
    """GET /api/v1/predictions - 1 trang History (PaginatedPredictionsResponseSchema).

    filters: predicted_class, priority, verification_status, is_healthy,
    model_key, created_from, created_to (ISO string) - khớp đúng tham số
    của api/routers/predictions.py -> list_prediction_history(). Giá trị
    None bị loại bỏ trước khi gửi (nghĩa là "không lọc theo trường này").
    """
    params: dict[str, Any] = {"page": page}
    if page_size is not None:
        params["page_size"] = page_size
    params.update({key: value for key, value in filters.items() if value is not None})

    response = httpx.get(
        f"{get_api_base_url()}/api/v1/predictions",
        params=params,
        timeout=_get_timeout_seconds(),
    )
    _raise_for_status(response)
    return response.json()


def check_database() -> bool:
    """Suy ra tình trạng kết nối Database qua 1 lệnh gọi nhẹ tới History API
    (GET .../predictions?page=1&page_size=1).

    Phase 9 không có endpoint /health riêng cho DB, và Dashboard không được
    import thẳng src.database để tự kiểm tra (chỉ gọi qua HTTP - quyết định
    đã chốt đầu Phase 10). Endpoint History bắt buộc query DB để trả kết
    quả, nên gọi thành công (kể cả total=0) đủ để coi là "DB kết nối được".
    """
    try:
        list_predictions(page=1, page_size=1)
        return True
    except (PredictionAPIError, httpx.HTTPError):
        return False


def get_prediction_detail(prediction_id: int) -> dict[str, Any]:
    """GET /api/v1/predictions/{id} - Decision Report đầy đủ của 1 prediction đã lưu."""
    response = httpx.get(
        f"{get_api_base_url()}/api/v1/predictions/{prediction_id}",
        timeout=_get_timeout_seconds(),
    )
    _raise_for_status(response)
    return response.json()