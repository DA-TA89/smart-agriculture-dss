"""Resolves the Priority Level directly from the Knowledge Base severity.

Priority reflects only the domain danger of the diagnosed disease if the
prediction is correct — it is intentionally NEVER modified by model
confidence. Model uncertainty is represented as a fully separate signal
(``verification_status``, see confidence_classifier.py); the two are only
combined for display purposes at the presentation layer (Dashboard/API),
not here (SDS Section 2 — Separation of Concerns).
"""

import logging
from typing import Any, Dict

from src.utils.logger import setup_logger

logger: logging.Logger = setup_logger("priority_calculator")


def resolve_priority(disease_info: Dict[str, Any], priority_config: Dict[str, Any]) -> str:
    """Resolves the priority level for a predicted class.

    Healthy predictions are intentionally NOT collapsed into the "Low"
    severity tier: "no disease present" is a different situation from "a
    disease is present but not urgent". The distinct label configured in
    ``priority.healthy_label`` is returned instead, preserving the
    Knowledge Base's own ``severity_level="None"`` semantics for healthy
    classes (confirmed design decision — see PHASE8_LOG.md).

    Args:
        disease_info: Knowledge base entry for the predicted class.
        priority_config: Parsed ``priority`` section of
            configs/decision_engine.yaml.

    Returns:
        Either ``priority_config["healthy_label"]`` (for healthy
        predictions) or the disease's ``severity_level`` as stored in the
        knowledge base, unchanged.
    """
    if disease_info["is_healthy"]:
        priority = priority_config["healthy_label"]
    else:
        priority = disease_info["severity_level"]

    logger.info(
        "Priority resolved (is_healthy=%s): '%s'", disease_info["is_healthy"], priority
    )
    return priority