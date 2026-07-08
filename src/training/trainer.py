"""Core training and validation loop logic."""

import logging
from src.utils.logger import setup_logger
logger: logging.Logger = setup_logger("trainer")
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from src.training.metrics import compute_classification_metrics



def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[GradScaler],
    use_amp: bool,
    log_interval_batches: int,
) -> float:
    """Runs one training epoch.

    Args:
        model: Model to train.
        dataloader: Training DataLoader.
        optimizer: Optimizer updating trainable parameters.
        criterion: Loss function.
        device: Device to run computation on.
        scaler: GradScaler for mixed precision, or None if AMP is disabled.
        use_amp: Whether to use automatic mixed precision.
        log_interval_batches: How often (in batches) to log progress.

    Returns:
        Average training loss over the epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = len(dataloader)

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        if use_amp and scaler is not None:
            with autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        if (batch_idx + 1) % log_interval_batches == 0:
            logger.info("Batch %d/%d - loss: %.4f", batch_idx + 1, num_batches, loss.item())

    return running_loss / num_batches


def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """Runs one validation epoch.

    Args:
        model: Model to evaluate.
        dataloader: Validation DataLoader.
        criterion: Loss function.
        device: Device to run computation on.

    Returns:
        Tuple of (average validation loss, metrics dictionary).
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = running_loss / len(dataloader)
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    metrics = compute_classification_metrics(y_true, y_pred)
    return avg_loss, metrics