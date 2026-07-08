"""Builds optimizer and loss function objects from configuration."""

import logging
from src.utils.logger import setup_logger
logger: logging.Logger = setup_logger("optimizer_factory")
from typing import Iterable

import torch
import torch.nn as nn


SUPPORTED_OPTIMIZERS = ("adam", "sgd")
SUPPORTED_LOSSES = ("cross_entropy",)
SGD_MOMENTUM: float = 0.9


def build_optimizer(
    trainable_params: Iterable[nn.Parameter],
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Builds an optimizer for the given trainable parameters.

    Args:
        trainable_params: Iterable of parameters to optimize.
        optimizer_name: One of "adam" or "sgd".
        learning_rate: Learning rate.
        weight_decay: L2 weight decay coefficient.

    Returns:
        A configured torch optimizer.

    Raises:
        ValueError: If optimizer_name is not supported.
    """
    if optimizer_name == "adam":
        return torch.optim.Adam(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            trainable_params, lr=learning_rate, weight_decay=weight_decay, momentum=SGD_MOMENTUM
        )
    raise ValueError(f"Unsupported optimizer '{optimizer_name}'. Must be one of {SUPPORTED_OPTIMIZERS}.")


def build_loss_function(loss_name: str) -> nn.Module:
    """Builds a loss function.

    Args:
        loss_name: Currently only "cross_entropy" is supported.

    Returns:
        A configured loss module.

    Raises:
        ValueError: If loss_name is not supported.
    """
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()
    raise ValueError(f"Unsupported loss '{loss_name}'. Must be one of {SUPPORTED_LOSSES}.")