"""
基础分割模型类 - 所有分割模型的基类
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional


class BaseSegmentationModel(nn.Module):
    """
    语义分割模型的基类，定义了通用的接口和行为。
    """

    def __init__(self, num_classes: int, in_channels: int = 3):
        """
        初始化基类。

        Args:
            num_classes: 分割类别数
            in_channels: 输入图像通道数
        """
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，由子类实现。

        Args:
            x: 输入图像 [B, C, H, W]

        Returns:
            分割 logits [B, num_classes, H, W]
        """
        raise NotImplementedError("Subclass must implement forward()")

    def get_params_groups(self) -> list:
        """
        获取参数组，用于不同的学习率设置。

        Returns:
            参数组列表
        """
        # 简单实现：所有参数为一组
        return [{'params': self.parameters()}]

    def load_pretrained(self, pretrained_path: str, strict: bool = True):
        """
        加载预训练权重。

        Args:
            pretrained_path: 预训练权重路径
            strict: 是否严格匹配参数名
        """
        state_dict = torch.load(pretrained_path, map_location='cpu')
        self.load_state_dict(state_dict, strict=strict)

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> 'BaseSegmentationModel':
        """
        从配置构建模型，由子类实现。

        Args:
            config: 模型配置字典

        Returns:
            模型实例
        """
        raise NotImplementedError("Subclass must implement build_from_config()")
