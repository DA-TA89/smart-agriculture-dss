"""Image transform pipelines for training, validation, and test datasets.

This module builds torchvision transform pipelines based on parameters
declared in ``configs/data.yaml`` (sections ``preprocessing`` and
``augmentation``). No image-processing parameter is hard-coded here; every
value is read from configuration to comply with SDS Section 8
(Configuration Strategy) and Section 9 (No Magic Number).
"""

import logging
from typing import Any, Dict

from torchvision import transforms

from src.utils.logger import setup_logger

logger: logging.Logger = setup_logger("feature_transforms")


def build_train_transform(config: Dict[str, Any]) -> transforms.Compose:
    """Build the training transform pipeline (resize + augmentation + normalize).

    Args:
        config: Parsed content of ``configs/data.yaml``.

    Returns:
        A torchvision ``Compose`` combining resize, augmentation, tensor
        conversion, and normalization steps.
    """
    preprocessing_cfg = config["preprocessing"]
    augmentation_cfg = config["augmentation"]

    image_size = preprocessing_cfg["image_size"]
    normalize_mean = preprocessing_cfg["normalize_mean"]
    normalize_std = preprocessing_cfg["normalize_std"]

    pipeline = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=augmentation_cfg["horizontal_flip_prob"]),
            transforms.RandomRotation(degrees=augmentation_cfg["rotation_degrees"]),
            transforms.ColorJitter(
                brightness=augmentation_cfg["color_jitter_brightness"],
                contrast=augmentation_cfg["color_jitter_contrast"],
                saturation=augmentation_cfg["color_jitter_saturation"],
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std),
        ]
    )
    logger.info("Built train transform pipeline with image_size=%s", image_size)
    return pipeline


def build_eval_transform(config: Dict[str, Any]) -> transforms.Compose:
    """Build the evaluation transform pipeline (resize + normalize only).

    Used for both the validation and the test set. No augmentation is
    applied so evaluation metrics reflect model behaviour on unmodified
    images.

    Args:
        config: Parsed content of ``configs/data.yaml``.

    Returns:
        A torchvision ``Compose`` combining resize, tensor conversion, and
        normalization steps.
    """
    preprocessing_cfg = config["preprocessing"]

    image_size = preprocessing_cfg["image_size"]
    normalize_mean = preprocessing_cfg["normalize_mean"]
    normalize_std = preprocessing_cfg["normalize_std"]

    pipeline = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std),
        ]
    )
    logger.info("Built eval transform pipeline with image_size=%s", image_size)
    return pipeline