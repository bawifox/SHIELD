"""
基础数据集类 - 所有数据集的基类
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, Any, Tuple, Optional, Callable, List
from PIL import Image
import os


class BaseDataset(Dataset):
    """
    分割数据集的基类，定义了通用的接口和行为。
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        num_classes: int = 19
    ):
        """
        初始化基类。

        Args:
            data_root: 数据根目录
            split: 数据集划分 (train/val/test)
            transform: 数据增强transforms
            num_classes: 分割类别数
        """
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.num_classes = num_classes
        self.data_list = self.load_data_list()

    def load_data_list(self) -> List[Dict[str, str]]:
        """
        加载数据列表，由子类实现。

        Returns:
            数据列表，每项包含图像和标签路径
        """
        raise NotImplementedError("Subclass must implement load_data_list()")

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        获取单个样本。

        Args:
            idx: 样本索引

        Returns:
            (图像, 标签字典)
        """
        data_info = self.data_list[idx]
        image = Image.open(data_info['image_path']).convert('RGB')

        label = None
        if 'label_path' in data_info and data_info['label_path']:
            label = Image.open(data_info['label_path'])

        if self.transform:
            transformed = self.transform(image=image, label=label)
            image = transformed['image']
            label = transformed.get('label')

        # 转换为tensor
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float()

        result = {'image': image}
        if label is not None:
            if not isinstance(label, torch.Tensor):
                label = torch.from_numpy(label).long()
            result['label'] = label

        return result

    @staticmethod
    def build_from_config(config: Dict[str, Any], split: str) -> 'BaseDataset':
        """
        从配置构建数据集，由子类实现。

        Args:
            config: 数据集配置字典
            split: 数据集划分

        Returns:
            数据集实例
        """
        raise NotImplementedError("Subclass must implement build_from_config()")
