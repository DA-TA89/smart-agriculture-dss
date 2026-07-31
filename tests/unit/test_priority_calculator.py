"""Unit tests for src.decision_engine.priority_calculator.

The single most important behavior to regression-test here (see
PHASE8_LOG.md section 3 - the rejected "escalate priority by confidence"
design, and section 3.4 - the final decoupled design):

1. priority == severity_level from the Knowledge Base, verbatim - it must
   NEVER be derived from or altered by confidence_score.
2. is_healthy=True must resolve to priority_config["healthy_label"]
   ("No Action Required"), NOT to "Low" - "no disease present" is
   deliberately kept distinct from "a disease is present but low priority".
"""

from src.decision_engine.priority_calculator import resolve_priority

# Matches configs/decision_engine.yaml -> priority (Phase 8).
PRIORITY_CONFIG = {
    "order": ["No Action Required", "Low", "Medium", "High", "Critical"],
    "healthy_label": "No Action Required",
}


class TestResolvePriority:
    def test_healthy_maps_to_healthy_label_not_low(self):
        disease_info = {"is_healthy": True, "severity_level": "Low"}
        priority = resolve_priority(disease_info, PRIORITY_CONFIG)
        assert priority == "No Action Required"
        assert priority != "Low"

    def test_diseased_returns_severity_level_verbatim(self):
        for severity in ["Low", "Medium", "High", "Critical"]:
            disease_info = {"is_healthy": False, "severity_level": severity}
            assert resolve_priority(disease_info, PRIORITY_CONFIG) == severity

    def test_critical_case_orange_huanglongbing(self):
        """Regression test for the one class allowed to reach "Critical"
        (Orange___Haunglongbing_(Citrus_greening) - PHASE8_LOG.md section 9)."""
        disease_info = {"is_healthy": False, "severity_level": "Critical"}
        assert resolve_priority(disease_info, PRIORITY_CONFIG) == "Critical"

    def test_priority_ignores_confidence_related_keys(self):
        """Even if a disease_info dict happened to carry extra keys that look
        confidence-related, resolve_priority() must not consult them - it
        only ever reads is_healthy and severity_level."""
        disease_info = {
            "is_healthy": False,
            "severity_level": "High",
            "confidence_score": 0.10,  # deliberately very low, must be ignored
        }
        assert resolve_priority(disease_info, PRIORITY_CONFIG) == "High"
