"""
Demo 损失函数实现
用于演示项目结构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import LOSS_REGISTRY
from .base import BaseLoss


@LOSS_REGISTRY.register()
class CrossEntropyLoss(BaseLoss):
    """
    交叉熵损失函数。
    """

    def __init__(self, num_classes: int = 19, ignore_index: int = 255, **kwargs):
        super().__init__(num_classes)
        self.ignore_index = ignore_index
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, predictions, targets):
        """
        Args:
            predictions: [B, num_classes, H, W]
            targets: {'label': [B, H, W]}
        """
        labels = targets['label']
        # 调整预测形状 [B, C, H, W] -> [B, H, W, C] -> [B*H*W, C]
        B, C, H, W = predictions.shape
        predictions = predictions.permute(0, 2, 3, 1).contiguous().view(-1, C)
        labels = labels.view(-1)

        loss = self.criterion(predictions, labels)
        return loss


@LOSS_REGISTRY.register()
class DiceLoss(BaseLoss):
    """
    Dice 损失函数，用于医学图像或小目标分割。
    """

    def __init__(self, num_classes: int = 19, smooth: float = 1.0, **kwargs):
        super().__init__(num_classes)
        self.smooth = smooth

    def forward(self, predictions, targets):
        """
        Args:
            predictions: [B, num_classes, H, W]
            targets: {'label': [B, H, W]}
        """
        labels = targets['label']
        B, C, H, W = predictions.shape

        # Softmax
        predictions = F.softmax(predictions, dim=1)

        # One-hot 编码标签
        labels_one_hot = F.one_hot(labels, num_classes=C).permute(0, 3, 1, 2).float()

        # 计算 Dice
        intersection = (predictions * labels_one_hot).sum(dim=(2, 3))
        union = predictions.sum(dim=(2, 3)) + labels_one_hot.sum(dim=(2, 3))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()

        return dice_loss
