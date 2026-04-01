"""
Cityscapes 数据集实现
支持读取 leftImg8bit 和 gtFine，返回 image 和 semantic segmentation label
支持训练集、验证集划分，自动进行 label 映射
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from typing import Dict, Any, List, Callable, Optional

from .registry import DATASET_REGISTRY
from .base import BaseDataset
from .cityscapes_labels import CITYSCAPES_REMAP_ARRAY


@DATASET_REGISTRY.register()
class CityscapesDataset(BaseDataset):
    """
    Cityscapes 数据集实现。

    数据目录结构:
        data_root/
            leftImg8bit/
                train/
                val/
            gtFine/
                train/
                val/

    支持:
    - 读取 leftImg8bit 图像
    - 读取 gtFine 标签 (labelIds 格式)
    - 自动将 35 类原始标签映射到 19 类训练标签
    - 训练集/验证集划分
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        num_classes: int = 19,
        remap: bool = True
    ):
        """
        Args:
            data_root: 数据根目录
            split: 数据集划分 (train/val)
            transform: 数据增强
            num_classes: 类别数
            remap: 是否进行 label 映射
        """
        self.img_dir = os.path.join(data_root, 'leftImg8bit', split)
        self.label_dir = os.path.join(data_root, 'gtFine', split)
        self.remap = remap
        super().__init__(data_root, split, transform, num_classes)

    def load_data_list(self) -> List[Dict[str, str]]:
        """加载数据列表"""
        data_list = []
        if not os.path.exists(self.img_dir):
            print(f"Warning: Image directory not found: {self.img_dir}")
            return data_list

        for city in sorted(os.listdir(self.img_dir)):
            city_img_dir = os.path.join(self.img_dir, city)
            if not os.path.isdir(city_img_dir):
                continue

            for img_name in sorted(os.listdir(city_img_dir)):
                if not img_name.endswith('_leftImg8bit.png'):
                    continue

                img_path = os.path.join(city_img_dir, img_name)
                base_name = img_name.replace('_leftImg8bit.png', '')
                label_name = f"{base_name}_gtFine_labelIds.png"
                label_path = os.path.join(self.label_dir, city, label_name)

                data_list.append({
                    'image_path': img_path,
                    'label_path': label_path if os.path.exists(label_path) else None
                })

        return data_list

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本。

        Args:
            idx: 样本索引

        Returns:
            {'image': tensor, 'label': tensor}
        """
        data_info = self.data_list[idx]

        # 读取图像
        image = Image.open(data_info['image_path']).convert('RGB')

        # 读取标签
        label = None
        if 'label_path' in data_info and data_info['label_path']:
            label = Image.open(data_info['label_path'])

            # 转换为 numpy 并进行 label 映射
            label_np = np.array(label)
            if self.remap:
                # 使用预创建的映射数组进行快速映射
                label_np = np.take(CITYSCAPES_REMAP_ARRAY, label_np)
            label = Image.fromarray(label_np.astype(np.uint8))

        # 应用 transform
        if self.transform:
            data = self.transform({'image': image, 'label': label})
            image = data['image']
            label = data.get('label')

        result = {'image': image}

        if label is not None:
            result['label'] = label

        return result


def build_cityscapes_dataloader(
    data_root: str,
    split: str = 'train',
    batch_size: int = 8,
    num_workers: int = 4,
    transform: Optional[Callable] = None,
    num_classes: int = 19,
    shuffle: bool = True,
    remap: bool = True
):
    """
    构建 Cityscapes 数据加载器。

    Args:
        data_root: 数据根目录
        split: 数据集划分
        batch_size: 批次大小
        num_workers: 工作进程数
        transform: 数据增强
        num_classes: 类别数
        shuffle: 是否打乱
        remap: 是否进行 label 映射

    Returns:
        DataLoader
    """
    from torch.utils.data import DataLoader

    dataset = CityscapesDataset(
        data_root=data_root,
        split=split,
        transform=transform,
        num_classes=num_classes,
        remap=remap
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )

    return loader
