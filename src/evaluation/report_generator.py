"""Report generation utilities for Phase 7: confusion matrix plots and
Markdown evaluation / comparison reports.

Follows the same "sinh report tự động vào docs/" pattern established by
``eda_analyzer.py`` (Phase 4) and ``splitter.py`` (Phase 5).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils.logger import setup_logger

logger: logging.Logger = setup_logger("evaluation_report_generator")

_SUMMARY_ROW_KEYS = ("accuracy", "macro avg", "weighted avg")


def plot_confusion_matrix(
    conf_matrix: np.ndarray, class_names: List[str], output_path: Path, model_key: str
) -> None:
    """Renders and saves a confusion matrix heatmap.

    Args:
        conf_matrix: Square confusion matrix, shape (num_classes, num_classes).
        class_names: Ordered class names matching matrix row/column order.
        output_path: File path to save the PNG image.
        model_key: Model identifier, used in the plot title.
    """
    num_classes = len(class_names)
    fig_size = max(12, num_classes * 0.4)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    image = ax.imshow(conf_matrix, cmap="Blues")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_yticklabels(class_names, fontsize=6)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion Matrix - {model_key}")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix plot: %s", output_path)


def _render_overall_metrics_section(result: Dict[str, Any]) -> List[str]:
    """Builds the Markdown lines for the overall-metrics table.

    Args:
        result: Output dict from evaluator.evaluate_model().

    Returns:
        List of Markdown lines.
    """
    return [
        "## Overall Metrics (Test Set)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Accuracy | {result['accuracy']:.4f} |",
        f"| Precision (macro) | {result['precision']:.4f} |",
        f"| Recall (macro) | {result['recall']:.4f} |",
        f"| F1-score (macro) | {result['macro_f1']:.4f} |",
        f"| ROC-AUC (macro, OvR) | {result['roc_auc']:.4f} |",
        "",
    ]


def _render_small_sample_section(result: Dict[str, Any]) -> List[str]:
    """Builds the Markdown lines warning about low-support classes.

    Args:
        result: Output dict from evaluator.evaluate_model().

    Returns:
        List of Markdown lines (empty if no small-sample classes).
    """
    if not result["small_sample_classes"]:
        return []

    lines = [
        "## ⚠️ Small-Sample Classes",
        "",
        "The following classes have fewer test samples than the configured "
        "warning threshold (`configs/data.yaml -> split.min_samples_per_class_warning`). "
        "Interpret their per-class precision/recall with caution — small "
        "support means these numbers can swing a lot from a handful of "
        "misclassifications.",
        "",
    ]
    for class_name in result["small_sample_classes"]:
        support = result["per_class_report"][class_name]["support"]
        lines.append(f"- `{class_name}` (support={int(support)})")
    lines.append("")
    return lines


def _render_per_class_table(result: Dict[str, Any]) -> List[str]:
    """Builds the Markdown lines for the full per-class metrics table.

    Args:
        result: Output dict from evaluator.evaluate_model().

    Returns:
        List of Markdown lines.
    """
    lines = [
        "## Per-Class Report",
        "",
        "| Class | Precision | Recall | F1-score | Support |",
        "|---|---|---|---|---|",
    ]
    for class_name, class_metrics in result["per_class_report"].items():
        if class_name in _SUMMARY_ROW_KEYS:
            continue
        lines.append(
            f"| {class_name} | {class_metrics['precision']:.4f} | "
            f"{class_metrics['recall']:.4f} | {class_metrics['f1-score']:.4f} | "
            f"{int(class_metrics['support'])} |"
        )
    lines.append("")
    return lines


def generate_evaluation_report(result: Dict[str, Any], output_path: Path) -> None:
    """Writes a per-model Markdown evaluation report.

    Args:
        result: Output dict from evaluator.evaluate_model().
        output_path: File path to write the Markdown report.
    """
    lines: List[str] = [
        f"# Evaluation Report — {result['model_key']}",
        "",
        f"- Checkpoint: `{result['checkpoint_path']}`",
        f"- Total parameters: {result['total_parameters']:,}",
        f"- Trainable parameters: {result['trainable_parameters']:,}",
        f"- Model size: {result['model_size_mb']:.2f} MB",
        f"- Inference time: {result['inference_time_ms_per_image']:.3f} ms/image",
        "",
    ]
    lines.extend(_render_overall_metrics_section(result))
    lines.extend(_render_small_sample_section(result))
    lines.extend(_render_per_class_table(result))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved evaluation report: %s", output_path)


def generate_comparison_report(results: List[Dict[str, Any]], output_path: Path) -> None:
    """Writes a Markdown report comparing all evaluated models.

    Args:
        results: List of result dicts from evaluator.evaluate_model(), one
            per model.
        output_path: File path to write the Markdown report.
    """
    metric_rows = [
        ("Accuracy", "accuracy", "{:.4f}"),
        ("Precision (macro)", "precision", "{:.4f}"),
        ("Recall (macro)", "recall", "{:.4f}"),
        ("F1-score (macro)", "macro_f1", "{:.4f}"),
        ("ROC-AUC (macro, OvR)", "roc_auc", "{:.4f}"),
        ("Inference time (ms/image)", "inference_time_ms_per_image", "{:.3f}"),
        ("Model size (MB)", "model_size_mb", "{:.2f}"),
        ("Total parameters", "total_parameters", "{:,}"),
        ("Trainable parameters", "trainable_parameters", "{:,}"),
    ]

    lines: List[str] = [
        "# Model Comparison Report",
        "",
        "> Both models were trained with **linear probing (frozen backbone)** "
        "— see PHASE6_LOG.md. This was a VRAM-safety choice for the 2GB "
        "MX550 GPU, not an inherent architecture limitation. Full "
        "fine-tuning could narrow or reverse the gap below.",
        "",
        "| Metric | " + " | ".join(result["model_key"] for result in results) + " |",
        "|---|" + "---|" * len(results),
    ]
    for label, key, fmt in metric_rows:
        row_values = [fmt.format(result[key]) for result in results]
        lines.append(f"| {label} | " + " | ".join(row_values) + " |")
    lines.append("")

    best_result = max(results, key=lambda result: result["macro_f1"])
    lines.append(
        f"**Best model by macro-F1:** `{best_result['model_key']}` "
        f"(macro_f1={best_result['macro_f1']:.4f}) — macro-F1 is used as the "
        f"selection criterion (not accuracy) due to the 36.23x class "
        f"imbalance confirmed in Phase 4."
    )
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved model comparison report: %s", output_path)