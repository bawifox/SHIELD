"""
数据集注册器 - 统一管理所有数据集
"""

from utils.registry import Registry, DATASET_REGISTRY
from .base import BaseDataset


__all__ = ['DATASET_REGISTRY', 'BaseDataset']
