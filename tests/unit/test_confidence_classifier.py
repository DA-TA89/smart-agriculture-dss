"""Unit tests for src.decision_engine.confidence_classifier.

Covers classify_confidence_tier() (boundary behavior at the "high"/"medium"
thresholds, and the [0, 1] validation) and determine_verification_status()
(the direct tier -> label mapping, independent of any Knowledge Base data -
see the module's own docstring on why this is deliberately decoupled from
severity/priority, PHASE8_LOG.md section 3).
"""

import pytest

from src.decision_engine.confidence_classifier import (
    classify_confidence_tier,
    determine_verification_status,
)

# Matches configs/decision_engine.yaml -> confidence_thresholds (Phase 8).
THRESHOLDS = {"high": 0.85, "medium": 0.60}

# Matches configs/decision_engine.yaml -> verification (Phase 8).
VERIFICATION_CONFIG = {
    "confidence_tier_required": "high",
    "verified_label": "Verified",
    "needs_verification_label": "Need Verification",
}


class TestClassifyConfidenceTier:
    def test_at_or_above_high_threshold_is_high(self):
        assert classify_confidence_tier(0.85, THRESHOLDS) == "high"
        assert classify_confidence_tier(0.98, THRESHOLDS) == "high"
        assert classify_confidence_tier(1.0, THRESHOLDS) == "high"

    def test_at_or_above_medium_below_high_is_medium(self):
        assert classify_confidence_tier(0.60, THRESHOLDS) == "medium"
        assert classify_confidence_tier(0.6830, THRESHOLDS) == "medium"
        assert classify_confidence_tier(0.8499, THRESHOLDS) == "medium"

    def test_below_medium_threshold_is_low(self):
        assert classify_confidence_tier(0.0, THRESHOLDS) == "low"
        assert classify_confidence_tier(0.4468, THRESHOLDS) == "low"
        assert classify_confidence_tier(0.5999, THRESHOLDS) == "low"

    @pytest.mark.parametrize("invalid_score", [-0.01, 1.01, -5.0, 2.0])
    def test_raises_value_error_out_of_range(self, invalid_score):
        with pytest.raises(ValueError):
            classify_confidence_tier(invalid_score, THRESHOLDS)


class TestDetermineVerificationStatus:
    def test_high_tier_is_verified(self):
        status = determine_verification_status("high", VERIFICATION_CONFIG)
        assert status == "Verified"

    @pytest.mark.parametrize("tier", ["medium", "low"])
    def test_non_high_tier_needs_verification(self, tier):
        status = determine_verification_status(tier, VERIFICATION_CONFIG)
        assert status == "Need Verification"

    def test_independent_of_severity_or_priority(self):
        """determine_verification_status() takes only confidence_tier + the
        verification config - it must not require/consult any disease or
        priority information (Separation of Concerns, PHASE8_LOG.md 3.4)."""
        status = determine_verification_status("low", VERIFICATION_CONFIG)
        assert status == VERIFICATION_CONFIG["needs_verification_label"]
