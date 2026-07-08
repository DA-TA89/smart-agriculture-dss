"""Metric computation utilities for training and validation loops."""

import logging
from src.utils.logger import setup_logger
logger: logging.Logger = setup_logger("training_metrics")
from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes accuracy and macro F1-score.

    Macro-F1 is used as the primary model-selection metric (not accuracy) because
    of the 36.23x class imbalance confirmed in Phase 4/5 — see PHASE5_LOG.md.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.

    Returns:
        Dictionary with keys "accuracy" and "macro_f1".
    """
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": float(accuracy), "macro_f1": float(macro_f1)}