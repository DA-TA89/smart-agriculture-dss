"""Loads and validates the disease knowledge base used by the Decision Engine.

The knowledge base is a hand-curated JSON resource
(``src/decision_engine/knowledge_base/disease_kb.json``) mapping each
PlantVillage class_name to disease information: severity level, symptoms,
treatment and prevention recommendations. This module is the single point
of access to that resource — no other module should read the JSON file
directly (Low Coupling, SDS Section 2).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from src.utils.logger import setup_logger

logger: logging.Logger = setup_logger("knowledge_base_loader")

REQUIRED_FIELDS = (
    "plant_type",
    "disease_name",
    "is_healthy",
    "pathogen_type",
    "pathogen_name",
    "symptoms",
    "severity_level",
    "spread_risk",
    "treatment_recommendation",
    "prevention_recommendation",
)


def load_knowledge_base(kb_path: Path) -> Dict[str, Dict[str, Any]]:
    """Loads the disease knowledge base JSON file.

    Args:
        kb_path: Absolute path to disease_kb.json.

    Returns:
        Mapping from class_name to its disease information dictionary.

    Raises:
        FileNotFoundError: If kb_path does not exist.
        ValueError: If any entry is missing a required field.
    """
    if not kb_path.exists():
        raise FileNotFoundError(f"Knowledge base file not found: {kb_path}")

    with open(kb_path, "r", encoding="utf-8") as kb_file:
        knowledge_base: Dict[str, Dict[str, Any]] = json.load(kb_file)

    _validate_knowledge_base(knowledge_base)
    logger.info("Loaded knowledge base with %d classes from %s", len(knowledge_base), kb_path)
    return knowledge_base


def _validate_knowledge_base(knowledge_base: Dict[str, Dict[str, Any]]) -> None:
    """Validates that every entry contains all required fields.

    Args:
        knowledge_base: Parsed knowledge base dictionary.

    Raises:
        ValueError: If any entry is missing a required field.
    """
    for class_name, entry in knowledge_base.items():
        missing_fields = [field for field in REQUIRED_FIELDS if field not in entry]
        if missing_fields:
            raise ValueError(
                f"Knowledge base entry '{class_name}' is missing fields: {missing_fields}"
            )


def validate_against_class_names(
    knowledge_base: Dict[str, Dict[str, Any]], class_names: List[str]
) -> None:
    """Cross-checks knowledge base keys against the model's class names.

    Ensures the knowledge base does not silently drift out of sync with
    ``data/processed/class_to_idx.json`` (e.g. after a dataset update).

    Args:
        knowledge_base: Parsed knowledge base dictionary.
        class_names: Ordered class names from class_to_idx.json.

    Raises:
        ValueError: If there is any mismatch between the two sets.
    """
    kb_classes = set(knowledge_base.keys())
    model_classes = set(class_names)

    missing_in_kb = model_classes - kb_classes
    extra_in_kb = kb_classes - model_classes

    if missing_in_kb or extra_in_kb:
        raise ValueError(
            "Knowledge base out of sync with class_to_idx.json. "
            f"Missing in KB: {sorted(missing_in_kb)}. "
            f"Extra in KB: {sorted(extra_in_kb)}."
        )
    logger.info("Knowledge base validated against %d model class names.", len(class_names))


def get_disease_info(
    knowledge_base: Dict[str, Dict[str, Any]], class_name: str
) -> Dict[str, Any]:
    """Looks up disease information for a predicted class.

    Args:
        knowledge_base: Parsed knowledge base dictionary.
        class_name: Predicted class name (must match a class_to_idx.json key).

    Returns:
        The disease information dictionary for class_name.

    Raises:
        KeyError: If class_name is not present in the knowledge base.
    """
    if class_name not in knowledge_base:
        raise KeyError(f"No knowledge base entry found for class_name '{class_name}'")
    return knowledge_base[class_name]