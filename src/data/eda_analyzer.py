"""EDA analysis functions cho Smart Agriculture DSS (Phase 4).

Cung cấp các hàm TÍNH TOÁN (không vẽ biểu đồ) phục vụ
notebooks/01_eda_plantvillage.ipynb. Việc trực quan hóa (Matplotlib/Plotly)
được thực hiện trực tiếp trong notebook vì đây là phần khám phá/trình bày,
không cần tái sử dụng ở module khác — tương tự cách phân chia trách nhiệm
giữa validator.py (tính toán + sinh report) và etl_pipeline.py (orchestrate)
đã dùng ở Phase 3.

Input: data/processed/metadata_clean.csv (sinh ra từ Phase 3).

Lưu ý về tên cột: các hằng số COLUMN_* bên dưới giả định theo đúng mô tả
trong PHASE3_LOG.md. Nếu schema thực tế trong metadata_schema.md khác,
chỉ cần sửa lại giá trị các hằng số này ở một chỗ duy nhất.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config.config_loader import get_project_root, load_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# ---- Tên cột trong metadata_clean.csv (điều chỉnh tại đây nếu schema khác) ----
COLUMN_CLASS_NAME = "class_name"
COLUMN_PLANT_TYPE = "plant_type"
COLUMN_DISEASE_NAME = "disease_name"
COLUMN_IS_HEALTHY = "is_healthy"
COLUMN_WIDTH = "width"
COLUMN_HEIGHT = "height"
COLUMN_IS_DUPLICATE = "is_duplicate"
COLUMN_IS_CORRUPT = "is_corrupt"

ROUND_DECIMALS = 2


def load_clean_metadata(metadata_path: Path | None = None) -> pd.DataFrame:
    """Đọc metadata_clean.csv sinh ra từ Phase 3.

    Args:
        metadata_path: Đường dẫn file metadata_clean.csv. Nếu None, tự suy ra
            từ configs/base.yaml (paths.data_processed_dir) và
            configs/data.yaml (metadata.clean_output_filename).

    Returns:
        DataFrame chứa metadata đã làm sạch.
    """
    if metadata_path is None:
        base_config = load_config("base")
        data_config = load_config("data")
        processed_dir = get_project_root() / base_config["paths"]["data_processed_dir"]
        metadata_path = processed_dir / data_config["metadata"]["clean_output_filename"]

    logger.info("Đang đọc metadata từ %s", metadata_path)
    dataframe = pd.read_csv(metadata_path)
    logger.info("Đã đọc %d dòng metadata", len(dataframe))
    return dataframe


def compute_class_distribution(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Tính số lượng ảnh theo từng class gốc (plant + disease).

    Args:
        dataframe: Metadata đã làm sạch.

    Returns:
        DataFrame 2 cột [class_name, image_count], sắp xếp giảm dần.
    """
    distribution = (
        dataframe.groupby(COLUMN_CLASS_NAME)
        .size()
        .reset_index(name="image_count")
        .sort_values("image_count", ascending=False)
        .reset_index(drop=True)
    )
    return distribution


def compute_plant_type_distribution(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Tính số lượng ảnh theo loại cây trồng (plant_type)."""
    distribution = (
        dataframe.groupby(COLUMN_PLANT_TYPE)
        .size()
        .reset_index(name="image_count")
        .sort_values("image_count", ascending=False)
        .reset_index(drop=True)
    )
    return distribution


def compute_healthy_vs_diseased(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Tính số lượng ảnh khỏe mạnh (healthy) so với ảnh có bệnh (diseased)."""
    counts = dataframe[COLUMN_IS_HEALTHY].value_counts().rename(
        index={True: "healthy", False: "diseased"}
    )
    result = counts.reset_index()
    result.columns = ["status", "image_count"]
    return result


def compute_imbalance_ratio(
    distribution: pd.DataFrame, count_column: str = "image_count"
) -> float:
    """Tính tỷ lệ chênh lệch giữa class nhiều ảnh nhất và ít ảnh nhất."""
    max_count = distribution[count_column].max()
    min_count = distribution[count_column].min()
    ratio = max_count / min_count
    return round(float(ratio), ROUND_DECIMALS)


def compute_image_dimension_stats(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Tính thống kê mô tả (min/max/mean/std...) cho width, height, aspect_ratio."""
    stats_frame = dataframe[[COLUMN_WIDTH, COLUMN_HEIGHT]].copy()
    stats_frame["aspect_ratio"] = stats_frame[COLUMN_WIDTH] / stats_frame[COLUMN_HEIGHT]
    return stats_frame.describe().round(ROUND_DECIMALS)


def compute_duplicate_summary(dataframe: pd.DataFrame) -> dict[str, int]:
    """Xác nhận lại kết quả loại trùng lặp đã thực hiện ở Phase 3.

    metadata_clean.csv về lý thuyết không còn dòng nào có is_duplicate=True.
    Hàm này chỉ kiểm tra chéo (sanity check), KHÔNG tính lại hash MD5
    (việc đó đã tốn ~18 phút ở Phase 3, không cần lặp lại).
    """
    remaining_duplicates = int(dataframe[COLUMN_IS_DUPLICATE].sum())
    return {
        "total_rows": len(dataframe),
        "remaining_duplicates": remaining_duplicates,
    }


def compute_missing_analysis(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Tính số lượng và tỷ lệ giá trị thiếu (NaN) theo từng cột.

    Returns:
        DataFrame chỉ gồm các cột có ít nhất 1 giá trị thiếu. Rỗng nếu
        không có giá trị thiếu nào.
    """
    missing_count = dataframe.isna().sum()
    missing_percent = (missing_count / len(dataframe) * 100).round(ROUND_DECIMALS)
    result = pd.DataFrame(
        {"missing_count": missing_count, "missing_percent": missing_percent}
    )
    return result[result["missing_count"] > 0].sort_values(
        "missing_count", ascending=False
    )


def build_eda_summary(
    dataframe: pd.DataFrame, imbalance_warning_threshold: float
) -> dict[str, Any]:
    """Gom toàn bộ kết quả phân tích EDA thành một dict tổng hợp.

    Args:
        dataframe: Metadata đã làm sạch.
        imbalance_warning_threshold: Ngưỡng tỷ lệ để cảnh báo mất cân bằng
            (đọc từ configs/data.yaml -> eda.imbalance_warning_threshold).

    Returns:
        Dict chứa toàn bộ bảng/kết quả con, dùng để sinh báo cáo Markdown.
    """
    class_distribution = compute_class_distribution(dataframe)
    imbalance_ratio = compute_imbalance_ratio(class_distribution)

    summary: dict[str, Any] = {
        "total_images": len(dataframe),
        "total_classes": dataframe[COLUMN_CLASS_NAME].nunique(),
        "total_plant_types": dataframe[COLUMN_PLANT_TYPE].nunique(),
        "class_distribution": class_distribution,
        "plant_type_distribution": compute_plant_type_distribution(dataframe),
        "healthy_vs_diseased": compute_healthy_vs_diseased(dataframe),
        "imbalance_ratio": imbalance_ratio,
        "is_imbalanced": imbalance_ratio >= imbalance_warning_threshold,
        "dimension_stats": compute_image_dimension_stats(dataframe),
        "duplicate_summary": compute_duplicate_summary(dataframe),
        "missing_analysis": compute_missing_analysis(dataframe),
    }
    logger.info(
        "EDA summary: %d ảnh, %d class, imbalance ratio = %.2fx",
        summary["total_images"],
        summary["total_classes"],
        imbalance_ratio,
    )
    return summary


def generate_eda_report(summary: dict[str, Any], output_path: Path) -> None:
    """Sinh báo cáo EDA dạng Markdown từ dict tổng hợp.

    Args:
        summary: Kết quả trả về từ build_eda_summary().
        output_path: Đường dẫn file .md sẽ được ghi ra
            (thường là docs/data_dictionary/eda_report.md).
    """
    most_common = summary["class_distribution"].iloc[0]
    least_common = summary["class_distribution"].iloc[-1]
    imbalance_note = (
        "CẢNH BÁO — cần xử lý ở Phase 5/6"
        if summary["is_imbalanced"]
        else "chấp nhận được"
    )

    lines = [
        "# EDA Report — Smart Agriculture DSS",
        "",
        f"- Tổng số ảnh: **{summary['total_images']:,}**",
        f"- Tổng số class: **{summary['total_classes']}**",
        f"- Tổng số loại cây trồng (plant_type): **{summary['total_plant_types']}**",
        f"- Tỷ lệ mất cân bằng (max/min): **{summary['imbalance_ratio']}x** ({imbalance_note})",
        "",
        "## Phân bố Class",
        "",
        f"- Class nhiều ảnh nhất: `{most_common[COLUMN_CLASS_NAME]}` — "
        f"{most_common['image_count']:,} ảnh",
        f"- Class ít ảnh nhất: `{least_common[COLUMN_CLASS_NAME]}` — "
        f"{least_common['image_count']:,} ảnh",
        "",
        "## Duplicate Analysis",
        "",
        f"- Tổng số dòng: {summary['duplicate_summary']['total_rows']:,}",
        f"- Duplicate còn sót lại: {summary['duplicate_summary']['remaining_duplicates']} "
        "(kỳ vọng = 0, đã loại bỏ ở Phase 3)",
        "",
        "## Missing Analysis",
        "",
    ]

    if summary["missing_analysis"].empty:
        lines.append("Không phát hiện giá trị thiếu (NaN) ở bất kỳ cột nào.")
    else:
        lines.append("| Cột | Số lượng thiếu | Tỷ lệ (%) |")
        lines.append("|---|---|---|")
        for column_name, row in summary["missing_analysis"].iterrows():
            lines.append(
                f"| {column_name} | {int(row['missing_count'])} | {row['missing_percent']}% |"
            )

    lines += [
        "",
        "## Khuyến nghị cho Phase 5/6",
        "",
        "- Xử lý mất cân bằng class bằng class weighting, oversampling hoặc "
        "augmentation có trọng số cho các class ít ảnh.",
        "- Kiểm tra lại các class có số ảnh thấp trước khi chia "
        "train/val/test để đảm bảo mỗi tập vẫn có đại diện đầy đủ (dùng "
        "stratified split).",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Đã ghi EDA report vào %s", output_path)