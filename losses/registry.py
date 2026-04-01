"""
损失函数注册器 - 统一管理所有损失函数
"""

from utils.registry import Registry, LOSS_REGISTRY
from .base import BaseLoss


__all__ = ['LOSS_REGISTRY', 'BaseLoss']
