"""Decision Engine orchestrator — the module's single public entry point.

Per SDS Section 6: decision_engine "Nhận Prediction, sinh Decision. Truy
xuất Knowledge Base để tìm mức độ nghiêm trọng và giải pháp." This module
wires together knowledge_base_loader, confidence_classifier,
priority_calculator, and decision_report_builder without exposing their
internals to callers (Low Coupling, SDS Section 2). Phase 9 (REST API)
should call ``run_decision_engine`` directly instead of re-implementing
this flow.

priority and verification_status are computed independently (see
priority_calculator.py and confidence_classifier.py respectively) and
returned as-is in the report — this module does not combine them into any
derived message; that is a presentation-layer decision for Phase 9/10.
"""

import logging
from typing import Any, Dict

from src.decision_engine.confidence_classifier import (
    classify_confidence_tier,
    determine_verification_status,
)
from src.decision_engine.decision_report_builder import build_decision_report
from src.decision_engine.knowledge_base_loader import get_disease_info
from src.decision_engine.priority_calculator import resolve_priority
from src.utils.logger import setup_logger

logger: logging.Logger = setup_logger("decision_engine")


def run_decision_engine(
    image_reference: str,
    predicted_class: str,
    confidence_score: float,
    knowledge_base: Dict[str, Dict[str, Any]],
    decision_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Runs the full decision flow for a single AI prediction.

    Args:
        image_reference: Filename or path of the input leaf image.
        predicted_class: Predicted class_name from src.inference.
        confidence_score: Predicted class probability, in [0, 1].
        knowledge_base: Output of knowledge_base_loader.load_knowledge_base().
        decision_config: Parsed content of configs/decision_engine.yaml.

    Returns:
        The final decision report dictionary (see
        decision_report_builder.build_decision_report()).
    """
    disease_info = get_disease_info(knowledge_base, predicted_class)

    confidence_tier = classify_confidence_tier(
        confidence_score, decision_config["confidence_thresholds"]
    )
    priority = resolve_priority(disease_info, decision_config["priority"])
    verification_status = determine_verification_status(
        confidence_tier, decision_config["verification"]
    )

    report = build_decision_report(
        image_reference=image_reference,
        predicted_class=predicted_class,
        confidence_score=confidence_score,
        confidence_tier=confidence_tier,
        disease_info=disease_info,
        priority=priority,
        verification_status=verification_status,
    )
    return report