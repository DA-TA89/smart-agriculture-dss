"""Assembles the final structured Decision Report for a single prediction.

``build_decision_report()`` returns plain structured data only — it never
composes derived warning text by combining priority and verification_status;
that presentation decision belongs to the Dashboard/API (Phase 9/10).
``format_decision_report_markdown()`` is a thin, unconditional renderer used
only by pipelines/decision_pipeline.py for CLI demo/debug output: it lists
every field as-is, with no branching logic based on field values.
"""

import logging
from datetime import datetime
from typing import Any, Dict

from src.utils.logger import setup_logger

logger: logging.Logger = setup_logger("decision_report_builder")


def build_decision_report(
    image_reference: str,
    predicted_class: str,
    confidence_score: float,
    confidence_tier: str,
    disease_info: Dict[str, Any],
    priority: str,
    verification_status: str,
) -> Dict[str, Any]:
    """Builds the final decision report dictionary for one prediction.

    Args:
        image_reference: Filename or path of the input leaf image (for
            traceability; the image itself is not stored in the report).
        predicted_class: Predicted class_name (matches class_to_idx.json).
        confidence_score: Predicted class probability, in [0, 1].
        confidence_tier: Output of classify_confidence_tier().
        disease_info: Knowledge base entry for predicted_class.
        priority: Output of priority_calculator.resolve_priority().
        verification_status: Output of
            confidence_classifier.determine_verification_status().

    Returns:
        A JSON-serializable decision report dictionary. "priority" and
        "verification_status" are returned as independent, unmodified
        fields — no combined/derived warning text is generated here.
    """
    report: Dict[str, Any] = {
        "image_reference": image_reference,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "ai_prediction": {
            "predicted_class": predicted_class,
            "plant_type": disease_info["plant_type"],
            "disease_name": disease_info["disease_name"],
            "is_healthy": disease_info["is_healthy"],
            "confidence_score": round(confidence_score, 4),
            "confidence_tier": confidence_tier,
        },
        "severity_assessment": {
            "severity_level": disease_info["severity_level"],
            "pathogen_type": disease_info["pathogen_type"],
            "pathogen_name": disease_info["pathogen_name"],
            "symptoms": disease_info["symptoms"],
            "spread_risk": disease_info["spread_risk"],
        },
        "priority": priority,
        "verification_status": verification_status,
        "treatment_recommendation": disease_info["treatment_recommendation"],
        "prevention_recommendation": disease_info["prevention_recommendation"],
    }
    logger.info(
        "Decision report built for '%s': priority=%s, verification_status=%s",
        predicted_class,
        priority,
        verification_status,
    )
    return report


def format_decision_report_markdown(report: Dict[str, Any]) -> str:
    """Formats a decision report dictionary as a human-readable Markdown string.

    Unconditional renderer: every field is listed as-is, in the same order
    for every report, regardless of priority/verification_status values.
    Intended for CLI demo/debug output only (pipelines/decision_pipeline.py).

    Args:
        report: Output of build_decision_report().

    Returns:
        Markdown-formatted decision report text.
    """
    prediction = report["ai_prediction"]
    severity = report["severity_assessment"]

    lines = [
        f"# Decision Report — {prediction['disease_name']}",
        "",
        f"- **Ảnh:** {report['image_reference']}",
        f"- **Thời gian:** {report['generated_at']}",
        f"- **Cây trồng:** {prediction['plant_type']}",
        f"- **Kết quả dự đoán:** {prediction['disease_name']} "
        f"({'Khỏe mạnh' if prediction['is_healthy'] else 'Có bệnh'})",
        f"- **Độ tin cậy:** {prediction['confidence_score']:.2%} "
        f"(mức: {prediction['confidence_tier']})",
        f"- **Mức độ nguy hiểm (severity):** {severity['severity_level']}",
        f"- **Priority:** {report['priority']}",
        f"- **Trạng thái xác minh:** {report['verification_status']}",
        "",
        "## Triệu chứng",
        severity["symptoms"],
        "",
        "## Khả năng lây lan",
        severity["spread_risk"],
        "",
        "## Khuyến nghị xử lý",
        report["treatment_recommendation"],
        "",
        "## Biện pháp phòng ngừa",
        report["prevention_recommendation"],
        "",
    ]
    return "\n".join(lines)