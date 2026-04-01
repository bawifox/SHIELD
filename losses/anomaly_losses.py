"""
Anomaly Segmentation Loss
- BCE + Dice Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import LOSS_REGISTRY
from .base import BaseLoss


@LOSS_REGISTRY.register()
class AnomalyBCEDiceLoss(BaseLoss):
    """
    Binary Cross Entropy + Dice Loss for Anomaly Segmentation
    """

    def __init__(
        self,
        num_classes: int = 1,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        bce_reduction: str = 'mean',
        **kwargs
    ):
        super().__init__(num_classes)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_reduction = bce_reduction
        self.bce = nn.BCEWithLogitsLoss(reduction=bce_reduction)

    def forward(self, predictions, targets):
        """
        Args:
            predictions: [B, 1, H, W] - raw logits (before sigmoid)
            targets: {'anomaly_mask': [B, 1, H, W]} - binary mask (0 or 1)

        Returns:
            loss: scalar loss
        """
        if isinstance(targets, dict):
            anomaly_mask = targets['anomaly_mask']
        else:
            anomaly_mask = targets

        # BCE Loss
        bce_loss = self.bce(predictions, anomaly_mask)

        # Dice Loss
        pred_probs = torch.sigmoid(predictions)
        dice_loss = self._dice_loss(pred_probs, anomaly_mask)

        # Total loss
        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss

        return total_loss

    def _dice_loss(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
        """
        Compute Dice Loss

        Args:
            pred: [B, 1, H, W] - predicted probabilities (0-1)
            target: [B, 1, H, W] - binary ground truth (0 or 1)

        Returns:
            dice_loss: scalar
        """
        pred = pred.flatten(1)
        target = target.flatten(1)

        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)

        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice.mean()

        return dice_loss


@LOSS_REGISTRY.register()
class AnomalyBCELoss(BaseLoss):
    """
    Binary Cross Entropy Loss for Anomaly Segmentation
    """

    def __init__(self, num_classes: int = 1, **kwargs):
        super().__init__(num_classes)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, predictions, targets):
        """
        Args:
            predictions: [B, 1, H, W] - raw logits
            targets: {'anomaly_mask': [B, 1, H, W]} or [B, 1, H, W]
        """
        if isinstance(targets, dict):
            anomaly_mask = targets['anomaly_mask']
        else:
            anomaly_mask = targets

        return self.bce(predictions, anomaly_mask)


@LOSS_REGISTRY.register()
class AnomalyDiceLoss(BaseLoss):
    """
    Dice Loss for Anomaly Segmentation
    """

    def __init__(self, num_classes: int = 1, smooth: float = 1.0, **kwargs):
        super().__init__(num_classes)
        self.smooth = smooth

    def forward(self, predictions, targets):
        """
        Args:
            predictions: [B, 1, H, W] - raw logits
            targets: {'anomaly_mask': [B, 1, H, W]} or [B, 1, H, W]
        """
        if isinstance(targets, dict):
            anomaly_mask = targets['anomaly_mask']
        else:
            anomaly_mask = targets

        pred_probs = torch.sigmoid(predictions)

        pred = pred_probs.flatten(1)
        target = anomaly_mask.flatten(1)

        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()

        return dice_loss
