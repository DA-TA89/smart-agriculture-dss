"""Generic timm-based image classifier with optional backbone freezing."""

import logging
from src.utils.logger import setup_logger
logger: logging.Logger = setup_logger("base_classifier")
from typing import List

import timm
import torch
import torch.nn as nn

DEFAULT_DROPOUT_RATE: float = 0.2


class TimmClassifier(nn.Module):
    """Wraps any timm backbone with a trainable linear classification head."""

    def __init__(
        self,
        timm_name: str,
        num_classes: int,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        dropout_rate: float = DEFAULT_DROPOUT_RATE,
    ) -> None:
        """Builds the classifier.

        Args:
            timm_name: timm model identifier (e.g. "efficientnet_b0").
            num_classes: Number of output disease classes.
            pretrained: Whether to load ImageNet pretrained weights.
            freeze_backbone: Whether to freeze backbone parameters.
            dropout_rate: Dropout probability applied before the linear head.
        """
        super().__init__()
        self.timm_name = timm_name
        self.freeze_backbone = freeze_backbone
        self.backbone = timm.create_model(timm_name, pretrained=pretrained, num_classes=0)
        feature_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(feature_dim, num_classes),
        )
        if freeze_backbone:
            self._freeze_backbone_parameters()
        logger.info(
            "Built %s: feature_dim=%d, num_classes=%d, freeze_backbone=%s",
            timm_name, feature_dim, num_classes, freeze_backbone,
        )

    def _freeze_backbone_parameters(self) -> None:
        """Disables gradients for backbone parameters and sets eval mode."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Computes class logits for a batch of images.

        Args:
            images: Input batch, shape (batch, 3, H, W).

        Returns:
            Logits tensor, shape (batch, num_classes).
        """
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.backbone(images)
        else:
            features = self.backbone(images)
        return self.head(features)

    def trainable_parameters(self) -> List[nn.Parameter]:
        """Returns the parameters that require gradients (for the optimizer)."""
        return [p for p in self.parameters() if p.requires_grad]

    def train(self, mode: bool = True) -> "TimmClassifier":
        """Keeps a frozen backbone in eval mode even when the model is set to train.

        This matters for EfficientNet's BatchNorm layers: without this override,
        their running statistics would keep updating during the forward pass
        even though the weights themselves are frozen.

        Args:
            mode: True for training mode, False for eval mode.

        Returns:
            self, for chaining.
        """
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self