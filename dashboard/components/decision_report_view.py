"""Render 1 Decision Report (dict khớp PredictionResponseSchema) thành UI.

Dùng chung cho trang Upload & Predict (report vừa dự đoán xong) và trang
History (report cũ xem lại chi tiết) - cùng 1 shape dữ liệu
(api/schemas/prediction_schemas.py), cùng 1 hàm render, không lặp code
layout (SDS: không Duplicate Code).
"""

from typing import Any

import pandas as pd
import streamlit as st

# Priority -> hàm alert Streamlit tương ứng - thuần trình bày, KHÔNG phải
# business rule (giá trị priority luôn lấy nguyên từ API, không tính lại ở đây).
_PRIORITY_ALERT = {
    "Critical": st.error,
    "High": st.warning,
    "Medium": st.info,
    "Low": st.info,
    "No Action Required": st.success,
}

_KB_DISCLAIMER = (
    "⚠️ Nội dung khuyến nghị xử lý/phòng ngừa bên dưới do AI soạn nháp dựa trên "
    "kiến thức bệnh học thực vật phổ biến, **chưa được chuyên gia nông nghiệp xác "
    "nhận**. Chỉ dùng tham khảo, không thay thế tư vấn "
    "chuyên môn khi ra quyết định xử lý cây trồng thật."
)


def render_decision_report(report: dict[str, Any]) -> None:
    """Vẽ 1 Decision Report đầy đủ (dict khớp PredictionResponseSchema)."""
    ai = report["ai_prediction"]
    severity = report["severity_assessment"]

    if "id" in report:
        st.caption(
            f"Prediction ID: {report['id']} · Model: {report.get('model_key')} · "
            f"Lưu lúc: {report.get('created_at')}"
        )

    alert_fn = _PRIORITY_ALERT.get(report["priority"], st.info)
    alert_fn(f"**Priority: {report['priority']}** · Verification: {report['verification_status']}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted class", ai["predicted_class"])
    col2.metric(
        "Confidence", f"{ai['confidence_score']:.2%}", help=f"Tier: {ai['confidence_tier']}"
    )
    col3.metric("Severity", severity["severity_level"])

    st.markdown(
        f"**Plant:** {ai['plant_type']} · **Disease:** {ai['disease_name']} "
        f"({'Healthy' if ai['is_healthy'] else 'Diseased'})"
    )

    with st.expander("Chi tiết bệnh học (pathogen, triệu chứng, mức lây lan)"):
        st.write(f"**Pathogen type:** {severity['pathogen_type']}")
        st.write(f"**Pathogen name:** {severity['pathogen_name']}")
        st.write(f"**Symptoms:** {severity['symptoms']}")
        st.write(f"**Spread risk:** {severity['spread_risk']}")

    st.warning(_KB_DISCLAIMER)
    st.markdown("**Khuyến nghị xử lý:**")
    st.write(report["treatment_recommendation"])
    st.markdown("**Khuyến nghị phòng ngừa:**")
    st.write(report["prevention_recommendation"])

    if report.get("top_k_predictions"):
        st.markdown("**Top-k dự đoán:**")
        top_k_df = pd.DataFrame(report["top_k_predictions"]).rename(
            columns={"class_name": "Class", "confidence_score": "Confidence"}
        )
        st.dataframe(top_k_df, hide_index=True, use_container_width=True)

    st.caption(f"Ảnh: {report['image_reference']} · Sinh lúc: {report['generated_at']}")