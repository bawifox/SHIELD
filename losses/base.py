"""
基础损失函数类 - 所有损失函数的基类
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional


class BaseLoss(nn.Module):
    """
    损失函数的基类，定义了通用的接口。
    """

    def __init__(self, num_classes: int = 19, **kwargs):
        """
        初始化基类。

        Args:
            num_classes: 分割类别数
            **kwargs: 其他损失函数参数
        """
        super().__init__()
        self.num_classes = num_classes

    def forward(self, predictions: torch.Tensor, targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算损失，由子类实现。

        Args:
            predictions: 模型预测 [B, num_classes, H, W]
            targets: 目标字典，包含 'label' 等

        Returns:
            损失值
        """
        raise NotImplementedError("Subclass must implement forward()")

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> 'BaseLoss':
        """
        从配置构建损失函数，由子类实现。

        Args:
            config: 损失函数配置字典

        Returns:
            损失函数实例
        """
        raise NotImplementedError("Subclass must implement build_from_config()")
