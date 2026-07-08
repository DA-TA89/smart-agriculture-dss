"""Factory for building classification models from configuration."""

import logging
from src.utils.logger import setup_logger
logger: logging.Logger = setup_logger("model_factory")

from src.config.config_loader import load_config
from src.models.base_classifier import DEFAULT_DROPOUT_RATE, TimmClassifier



SUPPORTED_MODEL_KEYS = ("efficientnet", "vit")


def build_model(model_key: str, num_classes: int) -> TimmClassifier:
    """Builds a TimmClassifier for the given model key using configs/model.yaml.

    Args:
        model_key: One of "efficientnet" or "vit".
        num_classes: Number of output classes.

    Returns:
        An initialized TimmClassifier instance.

    Raises:
        ValueError: If model_key is not supported.
    """
    if model_key not in SUPPORTED_MODEL_KEYS:
        raise ValueError(
            f"Unsupported model_key '{model_key}'. Must be one of {SUPPORTED_MODEL_KEYS}."
        )
    model_config = load_config("model")
    entry = model_config["models"][model_key]
    dropout_rate = model_config.get("head", {}).get("dropout_rate", DEFAULT_DROPOUT_RATE)

    model = TimmClassifier(
        timm_name=entry["timm_name"],
        num_classes=num_classes,
        pretrained=entry.get("pretrained", True),
        freeze_backbone=entry.get("freeze_backbone", True),
        dropout_rate=dropout_rate,
    )
    logger.info("Model '%s' built via factory: %s", model_key, entry["timm_name"])
    return model


def get_checkpoint_prefix(model_key: str) -> str:
    """Returns the checkpoint filename prefix for a model key.

    Args:
        model_key: One of "efficientnet" or "vit".

    Returns:
        The checkpoint_prefix string from configs/model.yaml.
    """
    model_config = load_config("model")
    return model_config["models"][model_key]["checkpoint_prefix"]