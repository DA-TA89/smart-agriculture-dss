"""Unit tests for src.decision_engine.knowledge_base_loader.

Uses pytest's tmp_path fixture to write small, self-contained JSON fixtures
instead of depending on the real src/decision_engine/knowledge_base/
disease_kb.json (38 real classes) - keeps these tests fast, isolated from
content edits, and independent of the still-pending expert review noted in
PHASE8_LOG.md section 9.
"""

import json

import pytest

from src.decision_engine.knowledge_base_loader import (
    get_disease_info,
    load_knowledge_base,
    validate_against_class_names,
)

VALID_ENTRY = {
    "plant_type": "Tomato",
    "disease_name": "Late_blight",
    "is_healthy": False,
    "pathogen_type": "Oomycete",
    "pathogen_name": "Phytophthora infestans",
    "symptoms": "Dark water-soaked lesions on leaves.",
    "severity_level": "High",
    "spread_risk": "High",
    "treatment_recommendation": "Apply fungicide.",
    "prevention_recommendation": "Avoid overhead irrigation.",
}


def _write_kb(tmp_path, kb_dict):
    kb_path = tmp_path / "disease_kb.json"
    kb_path.write_text(json.dumps(kb_dict), encoding="utf-8")
    return kb_path


class TestLoadKnowledgeBase:
    def test_loads_valid_kb(self, tmp_path):
        kb_path = _write_kb(tmp_path, {"Tomato___Late_blight": VALID_ENTRY})
        kb = load_knowledge_base(kb_path)
        assert "Tomato___Late_blight" in kb
        assert kb["Tomato___Late_blight"]["severity_level"] == "High"

    def test_missing_file_raises_file_not_found(self, tmp_path):
        missing_path = tmp_path / "does_not_exist.json"
        with pytest.raises(FileNotFoundError):
            load_knowledge_base(missing_path)

    def test_entry_missing_required_field_raises_value_error(self, tmp_path):
        incomplete_entry = dict(VALID_ENTRY)
        del incomplete_entry["severity_level"]
        kb_path = _write_kb(tmp_path, {"Tomato___Late_blight": incomplete_entry})
        with pytest.raises(ValueError):
            load_knowledge_base(kb_path)


class TestValidateAgainstClassNames:
    def test_matching_sets_pass_silently(self):
        kb = {"Tomato___Late_blight": VALID_ENTRY}
        # Should not raise.
        validate_against_class_names(kb, ["Tomato___Late_blight"])

    def test_missing_in_kb_raises_value_error(self):
        kb = {"Tomato___Late_blight": VALID_ENTRY}
        with pytest.raises(ValueError):
            validate_against_class_names(
                kb, ["Tomato___Late_blight", "Potato___healthy"]
            )

    def test_extra_in_kb_raises_value_error(self):
        kb = {
            "Tomato___Late_blight": VALID_ENTRY,
            "Potato___healthy": VALID_ENTRY,
        }
        with pytest.raises(ValueError):
            validate_against_class_names(kb, ["Tomato___Late_blight"])


class TestGetDiseaseInfo:
    def test_returns_entry_for_known_class(self):
        kb = {"Tomato___Late_blight": VALID_ENTRY}
        info = get_disease_info(kb, "Tomato___Late_blight")
        assert info == VALID_ENTRY

    def test_raises_key_error_for_unknown_class(self):
        kb = {"Tomato___Late_blight": VALID_ENTRY}
        with pytest.raises(KeyError):
            get_disease_info(kb, "Unknown___Class")
