"""Trang Analytics - thống kê tổng hợp trên toàn bộ prediction (hoặc theo
khoảng ngày). Không có endpoint /stats riêng (quyết định chốt đầu Phase
10): Dashboard tự lặp qua GET /api/v1/predictions (Phase 9) rồi tổng hợp
bằng pandas, không sửa code Phase 9.
"""

import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import streamlit as st

from dashboard.api_client import PredictionAPIError, list_predictions
from dashboard.components.charts import (
    healthy_vs_diseased_chart,
    model_comparison_chart,
    priority_distribution_chart,
    render_kpi_row,
    time_trend_chart,
    top_classes_chart,
    verification_status_chart,
)
from src.config.config_loader import load_config

st.title("📊 Analytics")
st.caption("Thống kê tổng hợp trên toàn bộ prediction (hoặc theo khoảng ngày đã chọn).")

_CACHE_TTL = load_config("dashboard")["cache"]["ttl_seconds"]
_ANALYTICS_CONFIG = load_config("dashboard")["analytics"]


def _to_iso(value: Optional[date]) -> Optional[str]:
    if not value:
        return None
    return datetime(value.year, value.month, value.day).isoformat()


col1, col2 = st.columns(2)
with col1:
    created_from = st.date_input("Từ ngày (bỏ trống = không giới hạn)", value=None)
with col2:
    created_to = st.date_input("Đến ngày (bỏ trống = không giới hạn)", value=None)


@st.cache_data(ttl=_CACHE_TTL, show_spinner="Đang tải dữ liệu để tổng hợp Analytics...")
def _fetch_all_predictions(created_from_iso: Optional[str], created_to_iso: Optional[str]):
    """Lặp toàn bộ trang của History API (page_size = max_page_size) và trả
    về (DataFrame, tổng số thật, đã bị cắt bớt hay chưa)."""
    api_pagination = load_config("api")["pagination"]
    page_size = api_pagination["max_page_size"]
    max_records = _ANALYTICS_CONFIG["max_records_fetch"]

    items: list[dict] = []
    page = 1
    total = 0
    while True:
        result = list_predictions(
            page=page,
            page_size=page_size,
            created_from=created_from_iso,
            created_to=created_to_iso,
        )
        items.extend(result["items"])
        total = result["total"]
        if not result["items"] or len(items) >= total or len(items) >= max_records:
            break
        page += 1

    df = pd.DataFrame(items)
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"])
    return df, total, len(items) < total


try:
    df, total, truncated = _fetch_all_predictions(_to_iso(created_from), _to_iso(created_to))
except PredictionAPIError as error:
    st.error(f"Không tải được dữ liệu Analytics: {error}")
    st.stop()

if truncated:
    st.warning(
        f"Đã đạt giới hạn an toàn {_ANALYTICS_CONFIG['max_records_fetch']} bản ghi "
        f"(tổng thực tế: {total}) - biểu đồ dưới đây chỉ phản ánh phần dữ liệu đã tải."
    )

render_kpi_row(df, total)

if df.empty:
    st.info("Chưa có dữ liệu prediction nào khớp bộ lọc.")
    st.stop()

row1_col1, row1_col2 = st.columns(2)
with row1_col1:
    st.plotly_chart(priority_distribution_chart(df), use_container_width=True)
with row1_col2:
    st.plotly_chart(healthy_vs_diseased_chart(df), use_container_width=True)

row2_col1, row2_col2 = st.columns(2)
with row2_col1:
    st.plotly_chart(
        top_classes_chart(df, _ANALYTICS_CONFIG["top_classes_limit"]), use_container_width=True
    )
with row2_col2:
    st.plotly_chart(model_comparison_chart(df), use_container_width=True)

row3_col1, row3_col2 = st.columns(2)
with row3_col1:
    st.plotly_chart(time_trend_chart(df), use_container_width=True)
with row3_col2:
    st.plotly_chart(verification_status_chart(df), use_container_width=True)