"""Trang Upload & Predict - 2 chế độ tách biệt (radio, không suy luận từ số
file được chọn):

- Single: 1 ảnh, hiển thị Decision Report đầy đủ (hành vi gốc từ đầu Phase 10).
- Batch: nhiều ảnh, xử lý TUẦN TỰ qua POST /api/v1/predictions (Phase 9) -
  không thêm endpoint mới, không sửa code Phase 9. Xử lý tuần tự (không
  song song) vì GPU MX550 chỉ 2GB VRAM (rủi ro đã ghi nhận từ Phase 6/7).
"""

import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import streamlit as st

from dashboard.api_client import PredictionAPIError, predict_from_upload
from dashboard.components.decision_report_view import render_decision_report
from src.config.config_loader import load_config

st.title("📤 Upload & Predict")

_inference_config = load_config("inference")["inference"]
_data_config = load_config("data")["dataset"]
_api_upload_config = load_config("api")["upload"]
_dashboard_upload_config = load_config("dashboard")["upload"]

_ALLOWED_EXTENSIONS = [ext.lstrip(".") for ext in _data_config["allowed_extensions"]]
_SUPPORTED_MODEL_KEYS = _inference_config["supported_model_keys"]


def _select_model_key() -> str:
    return st.selectbox(
        "Model",
        _SUPPORTED_MODEL_KEYS,
        index=_SUPPORTED_MODEL_KEYS.index(_inference_config["default_model_key"]),
        key="predict_model_key",
    )


def _render_single_mode(model_key: str) -> None:
    uploaded_file = st.file_uploader("Ảnh lá cây", type=_ALLOWED_EXTENSIONS, key="single_uploader")
    if uploaded_file is None:
        return

    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > _api_upload_config["max_file_size_mb"]:
        st.warning(
            f"File nặng {file_size_mb:.1f}MB, vượt giới hạn khuyến nghị "
            f"{_api_upload_config['max_file_size_mb']}MB - API có thể từ chối."
        )
    st.image(uploaded_file, caption=uploaded_file.name, width=300)

    if st.button("Predict", type="primary", key="single_predict_button"):
        with st.spinner("Đang chạy dự đoán..."):
            try:
                report = predict_from_upload(
                    filename=uploaded_file.name,
                    file_bytes=uploaded_file.getvalue(),
                    content_type=uploaded_file.type or "application/octet-stream",
                    model_key=model_key,
                )
            except PredictionAPIError as error:
                st.error(f"Dự đoán thất bại: {error}")
            else:
                st.success("Dự đoán thành công, đã lưu vào History.")
                render_decision_report(report)


def _render_batch_mode(model_key: str) -> None:
    max_files = _dashboard_upload_config["batch_max_files"]
    confirm_threshold = _inference_config["batch_confirm_threshold"]

    uploaded_files = st.file_uploader(
        "Ảnh lá cây (chọn nhiều file)",
        type=_ALLOWED_EXTENSIONS,
        accept_multiple_files=True,
        key="batch_uploader",
    )
    if not uploaded_files:
        return

    if len(uploaded_files) > max_files:
        st.error(
            f"Đã chọn {len(uploaded_files)} ảnh, vượt giới hạn tối đa {max_files} "
            "ảnh/lần cho Batch qua Dashboard. Với số lượng lớn hơn (vd. quét cả "
            "dataset), dùng CLI `pipelines/decision_pipeline.py --random`/`--limit` "
            "(Phase 8) thay thế."
        )
        return

    st.caption(f"Đã chọn {len(uploaded_files)} ảnh.")

    proceed = True
    if len(uploaded_files) > confirm_threshold:
        proceed = st.checkbox(
            f"Xác nhận xử lý {len(uploaded_files)} ảnh "
            f"(vượt ngưỡng cảnh báo {confirm_threshold} - xử lý tuần tự, có thể mất thời gian)",
            key="batch_confirm_checkbox",
        )

    if not st.button(
        "Predict Batch", type="primary", disabled=not proceed, key="batch_predict_button"
    ):
        return

    progress_bar = st.progress(0.0)
    status_text = st.empty()
    results: list[dict] = []
    errors: list[dict] = []
    start_time = time.perf_counter()

    for index, uploaded_file in enumerate(uploaded_files, start=1):
        status_text.text(f"Đang xử lý ảnh {index}/{len(uploaded_files)}: {uploaded_file.name}")
        try:
            report = predict_from_upload(
                filename=uploaded_file.name,
                file_bytes=uploaded_file.getvalue(),
                content_type=uploaded_file.type or "application/octet-stream",
                model_key=model_key,
            )
        except PredictionAPIError as error:
            errors.append({"filename": uploaded_file.name, "error": str(error)})
        else:
            report["_filename"] = uploaded_file.name
            results.append(report)
        progress_bar.progress(index / len(uploaded_files))

    elapsed_seconds = time.perf_counter() - start_time
    status_text.empty()
    progress_bar.empty()

    _render_batch_summary(results, errors, len(uploaded_files), elapsed_seconds)


def _render_batch_summary(
    results: list[dict], errors: list[dict], total_files: int, elapsed_seconds: float
) -> None:
    st.subheader("Kết quả Batch")

    success_count = len(results)
    failed_count = len(errors)
    avg_confidence = (
        sum(r["ai_prediction"]["confidence_score"] for r in results) / success_count
        if success_count
        else None
    )

    kpi_columns = st.columns(4)
    kpi_columns[0].metric("Total Processed", total_files)
    kpi_columns[1].metric("Success / Failed", f"{success_count} / {failed_count}")
    kpi_columns[2].metric(
        "Average Confidence", f"{avg_confidence:.1%}" if avg_confidence is not None else "-"
    )
    kpi_columns[3].metric("Total Processing Time", f"{elapsed_seconds:.1f}s")

    if errors:
        with st.expander(f"⚠️ {failed_count} ảnh lỗi"):
            st.dataframe(pd.DataFrame(errors), hide_index=True, use_container_width=True)

    if not results:
        st.info("Không có ảnh nào xử lý thành công.")
        return

    summary_df = pd.DataFrame(
        [
            {
                "filename": r["_filename"],
                "predicted_class": r["ai_prediction"]["predicted_class"],
                "confidence": r["ai_prediction"]["confidence_score"],
                "priority": r["priority"],
                "verification_status": r["verification_status"],
            }
            for r in results
        ]
    )
    st.dataframe(summary_df, hide_index=True, use_container_width=True)

    st.download_button(
        "Tải kết quả (CSV)",
        data=summary_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="batch_prediction_results.csv",
        mime="text/csv",
        key="batch_download_csv",
    )

    st.subheader("Xem chi tiết từng ảnh")
    selected_filename = st.selectbox(
        "Chọn ảnh", summary_df["filename"].tolist(), key="batch_detail_select"
    )
    selected_report = next(r for r in results if r["_filename"] == selected_filename)
    render_decision_report(selected_report)


_mode = st.radio("Chế độ", ["Single", "Batch"], horizontal=True, key="predict_mode")
_model_key = _select_model_key()

if _mode == "Single":
    st.caption("Tải 1 ảnh lá cây để chạy dự đoán bệnh và nhận Decision Report đầy đủ.")
    _render_single_mode(_model_key)
else:
    st.caption(
        "Tải nhiều ảnh cùng lúc - phù hợp kiểm thử hàng chục ảnh, sàng lọc ảnh nông trại, "
        "hoặc demo khả năng xử lý của hệ thống."
    )
    _render_batch_mode(_model_key)