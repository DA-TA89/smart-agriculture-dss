"""Classifies model confidence and derives a verification status from it.

Both functions in this module answer ONE question only: "how much should we
trust this specific prediction's label?" They intentionally carry no
information about how dangerous the predicted disease is — that is the
separate, independent concern of ``priority_calculator.resolve_priority()``,
which reads only the Knowledge Base and is never touched by confidence.

Future Improvement (explicitly out of scope for Phase 8): confidence_score is
the model's raw softmax output, not a calibrated probability. Neural networks
trained with cross-entropy/softmax are commonly overconfident (Guo et al.,
2017, "On Calibration of Modern Neural Networks"). Before relying on
``configs/decision_engine.yaml -> confidence_thresholds`` operationally, a
production system should validate them against held-out accuracy — e.g. a
reliability diagram, Expected Calibration Error (ECE), or Temperature
Scaling on the Phase 7 test set. This module deliberately does not implement
any of that; it only classifies whatever confidence_score the model provides.
"""

import logging
from typing import Any, Dict

from src.utils.logger import setup_logger

logger: logging.Logger = setup_logger("confidence_classifier")


def classify_confidence_tier(confidence_score: float, thresholds: Dict[str, Any]) -> str:
    """Classifies a confidence score into "high", "medium", or "low".

    Args:
        confidence_score: Predicted class probability, in [0, 1].
        thresholds: Parsed ``confidence_thresholds`` section of
            configs/decision_engine.yaml (keys "high", "medium").

    Returns:
        One of "high", "medium", "low".

    Raises:
        ValueError: If confidence_score is not in [0, 1].
    """
    if not 0.0 <= confidence_score <= 1.0:
        raise ValueError(f"confidence_score must be in [0, 1], got {confidence_score}")

    if confidence_score >= thresholds["high"]:
        tier = "high"
    elif confidence_score >= thresholds["medium"]:
        tier = "medium"
    else:
        tier = "low"

    logger.info("confidence_score=%.4f classified as tier='%s'", confidence_score, tier)
    return tier


def determine_verification_status(
    confidence_tier: str, verification_config: Dict[str, Any]
) -> str:
    """Determines whether a prediction should be flagged for human verification.

    This is a direct, unconditional function of confidence_tier alone. It
    does not read severity, priority, or any Knowledge Base data — combining
    this status with priority for display purposes is a presentation-layer
    decision left to the Dashboard/API (Phase 9/10), not to this module.

    Args:
        confidence_tier: Output of classify_confidence_tier()
            ("high", "medium", or "low").
        verification_config: Parsed ``verification`` section of
            configs/decision_engine.yaml.

    Returns:
        ``verification_config["verified_label"]`` if confidence_tier equals
        ``verification_config["confidence_tier_required"]``, otherwise
        ``verification_config["needs_verification_label"]``.
    """
    required_tier = verification_config["confidence_tier_required"]
    if confidence_tier == required_tier:
        status = verification_config["verified_label"]
    else:
        status = verification_config["needs_verification_label"]

    logger.info(
        "confidence_tier='%s' -> verification_status='%s'", confidence_tier, status
    )
    return status