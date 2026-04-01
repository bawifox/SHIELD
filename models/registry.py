"""
模型注册器 - 统一管理所有分割模型
"""

from utils.registry import Registry, MODEL_REGISTRY
from .base import BaseSegmentationModel


__all__ = ['MODEL_REGISTRY', 'BaseSegmentationModel']
