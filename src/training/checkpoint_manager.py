"""Checkpoint saving utilities following the SDS naming convention."""

import logging
from src.utils.logger import setup_logger
logger: logging.Logger = setup_logger("checkpoint_manager")
from pathlib import Path

import torch
import torch.nn as nn


DATASET_TAG: str = "plantvillage"


def build_checkpoint_filename(checkpoint_prefix: str, metric_value: float) -> str:
    """Builds a checkpoint filename per SDS Section 7 naming convention.

    Format: {architecture}_{dataset}_{metric}.pth
    Example: vit_tiny_plantvillage_f1_0.87.pth

    Args:
        checkpoint_prefix: Architecture prefix (e.g. "vit_tiny").
        metric_value: Validation macro F1-score achieved.

    Returns:
        The formatted checkpoint filename.
    """
    return f"{checkpoint_prefix}_{DATASET_TAG}_f1_{metric_value:.2f}.pth"

def _remove_existing_checkpoints(checkpoint_dir: Path, checkpoint_prefix: str) -> None:
    """Removes previously saved checkpoint files for this architecture.

    Only one "best" checkpoint per architecture should exist at a time, since
    the filename does not encode epoch number (SDS Section 7). Without this
    cleanup, every improved epoch leaves a stale, no-longer-best file behind.

    Args:
        checkpoint_dir: Directory containing checkpoint files.
        checkpoint_prefix: Architecture prefix used to match old files.
    """
    pattern = f"{checkpoint_prefix}_{DATASET_TAG}_f1_*.pth"
    for stale_path in checkpoint_dir.glob(pattern):
        stale_path.unlink()
        logger.info("Removed stale checkpoint: %s", stale_path)

def save_checkpoint(
    model: nn.Module,
    checkpoint_dir: str,
    checkpoint_prefix: str,
    metric_value: float,
    epoch: int,
) -> Path:
    """Saves the model's state_dict to the models/ directory, keeping only the best."""
    output_dir = Path(checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _remove_existing_checkpoints(output_dir, checkpoint_prefix)
    output_path = output_dir / build_checkpoint_filename(checkpoint_prefix, metric_value)
    torch.save(model.state_dict(), output_path)
    logger.info("Checkpoint saved at epoch %d: %s (macro_f1=%.4f)", epoch, output_path, metric_value)
    return output_path
