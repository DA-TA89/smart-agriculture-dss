"""Loads a trained model and runs disease prediction on a single new image.

Reuses Phase 6/7 infrastructure directly rather than duplicating it
(Reusability, SDS Section 2):
- ``src.evaluation.evaluator.find_checkpoint_path()`` /
  ``load_trained_model()`` for locating and loading the best checkpoint of
  a given architecture (same convention as Phase 7 test-set evaluation).
- ``src.features.transforms.build_eval_transform()`` for the exact same
  resize+normalize pipeline used at validation/test time (no augmentation).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image
from torch import Tensor, nn

from src.evaluation.evaluator import find_checkpoint_path, load_trained_model
from src.features.transforms import build_eval_transform
from src.models.model_factory import get_checkpoint_prefix
from src.utils.logger import setup_logger

logger: logging.Logger = setup_logger("predictor")


def load_class_names(class_to_idx_path: Path) -> List[str]:
    """Builds an index-ordered list of class names from class_to_idx.json.

    Args:
        class_to_idx_path: Path to data/processed/class_to_idx.json.

    Returns:
        List of class names such that class_names[i] is the class for
        label index i.
    """
    with open(class_to_idx_path, "r", encoding="utf-8") as mapping_file:
        class_to_idx: Dict[str, int] = json.load(mapping_file)

    class_names = [""] * len(class_to_idx)
    for class_name, index in class_to_idx.items():
        class_names[index] = class_name
    logger.info("Loaded %d class names from %s", len(class_names), class_to_idx_path)
    return class_names


def load_inference_model(
    model_key: str, models_dir: Path, num_classes: int, device: torch.device
) -> nn.Module:
    """Loads the best checkpoint for model_key, ready for inference.

    Args:
        model_key: One of model_factory.SUPPORTED_MODEL_KEYS
            ("efficientnet" or "vit").
        models_dir: Directory containing .pth checkpoint files.
        num_classes: Number of output classes.
        device: Device to load the model onto.

    Returns:
        The model in eval() mode.
    """
    checkpoint_prefix = get_checkpoint_prefix(model_key)
    checkpoint_path = find_checkpoint_path(models_dir, checkpoint_prefix)
    return load_trained_model(model_key, checkpoint_path, num_classes, device)


def preprocess_image(image_path: Path, data_config: Dict[str, Any]) -> Tensor:
    """Loads and transforms a single image for inference.

    Args:
        image_path: Path to a leaf image file (jpg/png).
        data_config: Parsed content of configs/data.yaml.

    Returns:
        A single-image batch tensor of shape (1, 3, H, W).
    """
    eval_transform = build_eval_transform(data_config)
    with Image.open(image_path) as image_file:
        image = image_file.convert("RGB")
    image_tensor = eval_transform(image)
    return image_tensor.unsqueeze(0)


def predict_image(
    image_path: Path,
    model: nn.Module,
    class_names: List[str],
    data_config: Dict[str, Any],
    device: torch.device,
    top_k: int = 3,
) -> Dict[str, Any]:
    """Predicts the disease class for a single new leaf image.

    Args:
        image_path: Path to a leaf image file.
        model: Model returned by load_inference_model(), in eval() mode.
        class_names: Index-ordered class names (see load_class_names()).
        data_config: Parsed content of configs/data.yaml.
        device: Device the model lives on.
        top_k: Number of top predictions to include for context.

    Returns:
        Dictionary with "predicted_class", "confidence_score", and
        "top_k_predictions" (list of {class_name, confidence_score}).
    """
    image_batch = preprocess_image(image_path, data_config).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(image_batch)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)

    top_probabilities, top_indices = torch.topk(probabilities, k=top_k)
    top_k_predictions = [
        {"class_name": class_names[index], "confidence_score": float(probability)}
        for probability, index in zip(top_probabilities.tolist(), top_indices.tolist())
    ]

    result = {
        "predicted_class": top_k_predictions[0]["class_name"],
        "confidence_score": top_k_predictions[0]["confidence_score"],
        "top_k_predictions": top_k_predictions,
    }
    logger.info(
        "Predicted '%s' with confidence %.4f for image %s",
        result["predicted_class"],
        result["confidence_score"],
        image_path,
    )
    return result