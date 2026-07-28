"""Chart builders (Plotly) + hàng KPI nhỏ cho trang Analytics.

Thuần trình bày: nhận vào 1 pandas DataFrame các PredictionSummarySchema
đã fetch sẵn (qua History API, xem dashboard/pages/3_Analytics.py) và trả
về Figure/vẽ trực tiếp. Không gọi API, không chứa business logic ở đây.
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config.config_loader import load_config

# Cùng thứ tự severity mà Decision Engine dùng (configs/decision_engine.yaml),
# tái sử dụng ở đây chỉ để sắp xếp/tô màu biểu đồ nhất quán - không tính lại.
_PRIORITY_ORDER = load_config("decision_engine")["priority"]["order"]
_PRIORITY_COLORS = {
    "No Action Required": "#2ecc71",
    "Low": "#3498db",
    "Medium": "#f1c40f",
    "High": "#e67e22",
    "Critical": "#e74c3c",
}


def render_kpi_row(df: pd.DataFrame, total: int) -> None:
    """Hàng KPI nhỏ tóm tắt nhanh dữ liệu hiện đang tải."""
    columns = st.columns(5)
    columns[0].metric("Tổng số prediction", total)
    if df.empty:
        return
    healthy_pct = df["is_healthy"].mean() * 100
    needs_verification_pct = (df["verification_status"] != "Verified").mean() * 100
    critical_count = int((df["priority"] == "Critical").sum())
    avg_confidence = df["confidence_score"].mean() * 100
    columns[1].metric("% Healthy", f"{healthy_pct:.1f}%")
    columns[2].metric("Cần xác minh", f"{needs_verification_pct:.1f}%")
    columns[3].metric("Critical", critical_count)
    columns[4].metric("Confidence TB", f"{avg_confidence:.1f}%")


def priority_distribution_chart(df: pd.DataFrame):
    counts = df["priority"].value_counts().reindex(_PRIORITY_ORDER, fill_value=0)
    return px.bar(
        x=counts.index,
        y=counts.values,
        color=counts.index,
        color_discrete_map=_PRIORITY_COLORS,
        labels={"x": "Priority", "y": "Số lượng"},
        title="Phân bố Priority",
    )


def healthy_vs_diseased_chart(df: pd.DataFrame):
    labels = df["is_healthy"].map({True: "Healthy", False: "Diseased"})
    counts = labels.value_counts()
    return px.pie(
        names=counts.index,
        values=counts.values,
        title="Tỷ lệ Healthy vs Diseased",
        color=counts.index,
        color_discrete_map={"Healthy": "#2ecc71", "Diseased": "#e74c3c"},
    )


def top_classes_chart(df: pd.DataFrame, top_n: int):
    counts = df["predicted_class"].value_counts().head(top_n)
    return px.bar(
        x=counts.values,
        y=counts.index,
        orientation="h",
        labels={"x": "Số lượng", "y": "Predicted class"},
        title=f"Top {top_n} class được dự đoán nhiều nhất",
    )


def model_comparison_chart(df: pd.DataFrame):
    summary = (
        df.groupby("model_key")
        .agg(count=("id", "count"), avg_confidence=("confidence_score", "mean"))
        .reset_index()
    )
    summary["avg_confidence"] = summary["avg_confidence"] * 100
    return px.bar(
        summary,
        x="model_key",
        y="count",
        color="model_key",
        hover_data={"avg_confidence": ":.1f"},
        labels={"model_key": "Model", "count": "Số lượng prediction"},
        title="So sánh EfficientNet vs ViT (số lượng · hover xem confidence TB %)",
    )


def time_trend_chart(df: pd.DataFrame):
    daily_counts = df.set_index("created_at").resample("D").size().reset_index(name="count")
    return px.line(
        daily_counts,
        x="created_at",
        y="count",
        markers=True,
        labels={"created_at": "Ngày", "count": "Số prediction"},
        title="Xu hướng số lượng prediction theo thời gian",
    )


def verification_status_chart(df: pd.DataFrame):
    counts = df["verification_status"].value_counts()
    return px.pie(
        names=counts.index, values=counts.values, title="Tỷ lệ Verified vs Need Verification"
    )