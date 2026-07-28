"""Bộ filter widget dùng chung cho trang History (và tuỳ chọn Analytics).

Danh sách giá trị priority/verification_status/model_key lấy thẳng từ
cùng file config mà API/Decision Engine đang dùng (decision_engine.yaml,
inference.yaml) - không định nghĩa lại (SDS: không Duplicate Code).

`predicted_class` không có config kiểu này (38 class name nằm trong
Knowledge Base do src.decision_engine sở hữu, mà Dashboard không được
import trực tiếp theo quyết định "chỉ gọi qua HTTP" đã chốt đầu Phase 10),
nên dùng text input khớp chính xác thay vì dropdown động.
"""

from datetime import date, datetime
from typing import Any, Optional

import streamlit as st

from src.config.config_loader import load_config

_ALL_LABEL = "Tất cả"


def get_priority_options() -> list[str]:
    return list(load_config("decision_engine")["priority"]["order"])


def get_verification_status_options() -> list[str]:
    verification = load_config("decision_engine")["verification"]
    return [verification["verified_label"], verification["needs_verification_label"]]


def get_model_key_options() -> list[str]:
    return list(load_config("inference")["inference"]["supported_model_keys"])


def _to_iso(value: Optional[date]) -> Optional[str]:
    if not value:
        return None
    return datetime(value.year, value.month, value.day).isoformat()


def render_filter_bar() -> dict[str, Any]:
    """Vẽ bộ filter và trả về dict tham số sẵn sàng truyền cho api_client.list_predictions()."""
    columns = st.columns(5)

    with columns[0]:
        predicted_class = st.text_input(
            "Predicted class (khớp chính xác)",
            placeholder="vd. Tomato___Late_blight",
        )
    with columns[1]:
        priority = st.selectbox("Priority", [_ALL_LABEL] + get_priority_options())
    with columns[2]:
        verification_status = st.selectbox(
            "Verification", [_ALL_LABEL] + get_verification_status_options()
        )
    with columns[3]:
        model_key = st.selectbox("Model", [_ALL_LABEL] + get_model_key_options())
    with columns[4]:
        is_healthy_label = st.selectbox("Healthy?", [_ALL_LABEL, "Healthy", "Diseased"])

    date_columns = st.columns(2)
    with date_columns[0]:
        created_from = st.date_input("Từ ngày", value=None)
    with date_columns[1]:
        created_to = st.date_input("Đến ngày", value=None)

    return {
        "predicted_class": predicted_class.strip() or None,
        "priority": None if priority == _ALL_LABEL else priority,
        "verification_status": None if verification_status == _ALL_LABEL else verification_status,
        "model_key": None if model_key == _ALL_LABEL else model_key,
        "is_healthy": {_ALL_LABEL: None, "Healthy": True, "Diseased": False}[is_healthy_label],
        "created_from": _to_iso(created_from),
        "created_to": _to_iso(created_to),
    }