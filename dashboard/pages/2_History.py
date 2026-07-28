"""Trang History - tra cứu lịch sử prediction qua GET /api/v1/predictions
(Phase 9), có filter + phân trang, xem chi tiết Decision Report từng bản ghi.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import streamlit as st

from dashboard.api_client import PredictionAPIError, get_prediction_detail, list_predictions
from dashboard.components.decision_report_view import render_decision_report
from dashboard.components.filters import render_filter_bar
from src.config.config_loader import load_config

st.title("📜 History")
st.caption("Lịch sử prediction đã lưu, mới nhất trước.")

_CACHE_TTL = load_config("dashboard")["cache"]["ttl_seconds"]


@st.cache_data(ttl=_CACHE_TTL, show_spinner="Đang tải lịch sử...")
def _cached_list_predictions(page: int, page_size: int, filters_key: tuple) -> dict:
    return list_predictions(page=page, page_size=page_size, **dict(filters_key))


pagination_config = load_config("api")["pagination"]

st.subheader("Bộ lọc")
filters = render_filter_bar()

col_page, col_size = st.columns(2)
with col_page:
    page = st.number_input("Trang", min_value=1, value=1, step=1)
with col_size:
    page_size_options = sorted(
        {10, 20, 50, pagination_config["default_page_size"], pagination_config["max_page_size"]}
    )
    page_size = st.selectbox(
        "Số dòng/trang",
        page_size_options,
        index=page_size_options.index(pagination_config["default_page_size"]),
    )

try:
    result = _cached_list_predictions(int(page), int(page_size), tuple(sorted(filters.items())))
except PredictionAPIError as error:
    st.error(f"Không tải được lịch sử: {error}")
    st.stop()

st.caption(f"Tổng số bản ghi khớp bộ lọc: {result['total']}")

if not result["items"]:
    st.info("Không có bản ghi nào khớp bộ lọc.")
else:
    df = pd.DataFrame(result["items"])
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("Xem chi tiết")
    selected_id = st.selectbox("Chọn ID để xem Decision Report đầy đủ", df["id"].tolist())
    if st.button("Xem chi tiết"):
        try:
            detail = get_prediction_detail(int(selected_id))
        except PredictionAPIError as error:
            st.error(f"Không tải được chi tiết: {error}")
        else:
            render_decision_report(detail)