# Losses package - 损失函数模块
from .registry import LOSS_REGISTRY
from .base import BaseLoss
from .demo_losses import CrossEntropyLoss, DiceLoss
from .anomaly_losses import AnomalyBCEDiceLoss, AnomalyBCELoss, AnomalyDiceLoss

__all__ = [
    'LOSS_REGISTRY',
    'BaseLoss',
    'CrossEntropyLoss',
    'DiceLoss',
    'AnomalyBCEDiceLoss',
    'AnomalyBCELoss',
    'AnomalyDiceLoss',
]
