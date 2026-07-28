"""Dashboard entry point.

Chạy bằng: `streamlit run dashboard/app.py` (từ project root, venv đã activate,
API - Phase 9 - đang chạy ở configs/api.yaml -> api.host/api.port).

Dùng st.navigation/st.Page (Streamlit >=1.36, dự án đang pin streamlit==1.40.2)
để khai báo tường minh 3 trang + icon, thay vì dựa vào emoji trong tên file
(tránh rủi ro encode filename trên Windows/Git). set_page_config() chỉ được
gọi 1 lần duy nhất ở đây - các file trong dashboard/pages/ không được gọi lại.

Bảng màu/theme tổng thể (áp dụng cho MỌI trang) nằm ở .streamlit/config.toml
(cơ chế theming gốc của Streamlit). Phần CSS trong file này CHỈ tạo thêm vài
khối trang trí cho riêng Home (hero banner, status pill, tech badge, leaf
accent, footer) - không override widget nào của Streamlit, không dùng
framework/thư viện UI ngoài.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from dashboard.api_client import check_database, check_health, get_api_base_url
from src.config.config_loader import load_config

dashboard_config = load_config("dashboard")["dashboard"]

st.set_page_config(
    page_title=dashboard_config["page_title"],
    page_icon=dashboard_config["page_icon"],
    layout="wide",
)

# Logo thương hiệu ở đầu sidebar (st.logo - Streamlit >=1.37, native, không
# CSS hack). logo_full.png: hiện khi sidebar mở rộng (navy, nền trong suốt,
# chỉ lấy phần mark - bỏ tagline "DATA INSIGHT IMPACT" để gọn/cân đối trong
# khung sidebar hẹp). logo_icon.png: badge trắng-trên-navy bo góc, hiện thu
# gọn khi sidebar bị đóng. Gọi 1 lần ở đây (giống set_page_config) là đủ cho
# mọi trang, vì app.py chạy lại trước mỗi lần điều hướng (st.navigation).
#
# 2 file trong dashboard/assets/ được xử lý (crop sát bbox + đệm 8%, tách nền
# trong suốt, tô lại màu navy #111827/trắng) từ nguồn ảnh gốc độ phân giải
# cao (1536x1024) người dùng cung cấp.
_ASSETS_DIR = _PROJECT_ROOT / "dashboard" / "assets"
st.logo(
    image=str(_ASSETS_DIR / "logo_full.png"),
    icon_image=str(_ASSETS_DIR / "logo_icon.png"),
    link="https://github.com/DA-TA89",
    size="large",
)

_HOME_CSS = """
<style>
.hero-banner {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, #1B4332 0%, #2D6A4F 55%, #40916C 100%);
    border-radius: 16px;
    padding: 2.4rem 3rem;
    margin-bottom: 1.4rem;
}
.hero-title {
    color: #F7FAF9;
    font-size: 2.1rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin: 0 0 0.3rem 0;
}
.hero-subtitle {
    color: #D8F3DC;
    font-size: 1.05rem;
    font-weight: 400;
    margin: 0 0 1.1rem 0;
}
.tech-badges { display: flex; flex-wrap: wrap; gap: 0.5rem; max-width: 80%; }
.tech-badge {
    background: rgba(255, 255, 255, 0.14);
    border: 1px solid rgba(255, 255, 255, 0.28);
    color: #FEF3C7;
    padding: 0.28rem 0.75rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 500;
}
.leaf-accent {
    position: absolute;
    top: 1.3rem;
    right: 2.2rem;
    font-size: 4rem;
    line-height: 1;
    filter: drop-shadow(0 10px 14px rgba(0, 0, 0, 0.25));
    animation: leaf-float 4s ease-in-out infinite;
}
@keyframes leaf-float {
    0%, 100% { transform: translateY(0) rotate(-8deg) rotateY(0deg); }
    50% { transform: translateY(-12px) rotate(6deg) rotateY(20deg); }
}
@media (prefers-reduced-motion: reduce) {
    .leaf-accent { animation: none; }
}
.status-row {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.75rem;
    margin: 0.1rem 0 1.6rem 0;
}
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.3rem 0.75rem;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 500;
    border: 1px solid #E2E8E4;
}
.status-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.pill-ok { color: #14532D; background: #F0FDF4; border-color: #BBF7D0; }
.pill-ok .status-dot { background: #22C55E; }
.pill-down { color: #7F1D1D; background: #FEF2F2; border-color: #FECACA; }
.pill-down .status-dot { background: #EF4444; }
.docs-link {
    margin-left: auto;
    font-size: 0.85rem;
    font-weight: 500;
    color: #2D6A4F;
    text-decoration: none;
}
.docs-link:hover { text-decoration: underline; }
.app-footer {
    text-align: center;
    color: #6B7280;
    font-size: 0.82rem;
    padding-top: 0.4rem;
    line-height: 1.6;
}
.app-footer a { color: #2D6A4F; text-decoration: none; font-weight: 500; }
.app-footer a:hover { text-decoration: underline; }
</style>
"""


def _status_pill_html(label: str, ok: bool, ok_text: str, down_text: str) -> str:
    state_class = "pill-ok" if ok else "pill-down"
    text = ok_text if ok else down_text
    return (
        f'<span class="status-pill {state_class}">'
        f'<span class="status-dot"></span>{label}: {text}</span>'
    )


def _render_home() -> None:
    st.markdown(_HOME_CSS, unsafe_allow_html=True)

    st.markdown(
        '<div class="hero-banner">'
        '<span class="leaf-accent">🍃</span>'
        '<div class="hero-title">🌿 Smart Agriculture DSS</div>'
        '<div class="hero-subtitle">Decision Support System for Plant Disease Diagnosis</div>'
        '<div class="tech-badges">'
        '<span class="tech-badge">Deep Learning</span>'
        '<span class="tech-badge">Vision Transformer</span>'
        '<span class="tech-badge">FastAPI</span>'
        '<span class="tech-badge">PostgreSQL</span>'
        "</div></div>",
        unsafe_allow_html=True,
    )

    backend_ok = check_health()
    database_ok = check_database() if backend_ok else False
    pills_html = _status_pill_html(
        "Backend", backend_ok, "Running", "Not running"
    ) + _status_pill_html("Database", database_ok, "Connected", "Not connected")
    st.markdown(
        f'<div class="status-row">{pills_html}'
        f'<a class="docs-link" href="{get_api_base_url()}/docs" target="_blank">'
        "Xem chi tiết API (Swagger) →</a></div>",
        unsafe_allow_html=True,
    )
    if not backend_ok:
        st.error(
            f"Không kết nối được API tại `{get_api_base_url()}`. Hãy chắc chắn API "
            "(Phase 9, `uvicorn api.main:app`) đang chạy trước khi dùng Dashboard."
        )

    st.subheader("Bắt đầu nhanh")
    feature_cards = [
        (
            upload_page,
            "📤",
            "Upload & Predict",
            "Tải ảnh lá cây, chạy dự đoán bệnh và nhận khuyến nghị xử lý ngay lập tức.",
        ),
        (
            history_page,
            "📜",
            "History",
            "Tra cứu lại các lần dự đoán trước, lọc theo priority, model, khoảng thời gian.",
        ),
        (
            analytics_page,
            "📊",
            "Analytics",
            "Thống kê tổng hợp: phân bố mức độ ưu tiên, so sánh mô hình, xu hướng theo thời gian.",
        ),
    ]
    columns = st.columns(3)
    for column, (page, icon, title, description) in zip(columns, feature_cards):
        with column:
            with st.container(border=True):
                st.markdown(f"#### {icon} {title}")
                st.caption(description)
                st.page_link(page, label="Mở trang")

    st.divider()
    st.markdown(
        '<div class="app-footer">'
        "Smart Agriculture DSS · Deep Learning Decision Support System · Dataset: "
        '<a href="https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset" '
        'target="_blank">PlantVillage</a>'
        "<br>Author: Phạm Tuấn Anh · "
        '<a href="https://github.com/DA-TA89" target="_blank">github.com/DA-TA89</a>'
        "</div>",
        unsafe_allow_html=True,
    )


home_page = st.Page(_render_home, title="Home", icon="🏠", default=True)
upload_page = st.Page("pages/1_Upload_Predict.py", title="Upload & Predict", icon="📤")
history_page = st.Page("pages/2_History.py", title="History", icon="📜")
analytics_page = st.Page("pages/3_Analytics.py", title="Analytics", icon="📊")

navigation = st.navigation([home_page, upload_page, history_page, analytics_page])
navigation.run()