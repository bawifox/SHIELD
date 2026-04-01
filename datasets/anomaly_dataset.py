"""
Anomaly Segmentation Dataset
支持读取道路场景图像和二值 anomaly mask
支持训练/验证划分，输出 image, anomaly_mask, meta_info
预留 mixed-scale benchmark 统计信息字段
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from typing import Dict, Any, List, Optional, Callable

from .registry import DATASET_REGISTRY
from .base import BaseDataset


@DATASET_REGISTRY.register()
class AnomalySegmentationDataset(BaseDataset):
    """
    Anomaly Segmentation Dataset

    数据目录结构 (默认 - 有子目录):
        data_root/
            images/
                train/
                    xxx.png
                val/
                    xxx.png
            anomaly_masks/
                train/
                    xxx.png
                val/
                    xxx.png

    或扁平目录结构:
        data_root/
            synthetic_images/
                xxx.jpg
            synthetic_masks/
                xxx.png

    输出:
        {
            'image': tensor [3, H, W],
            'anomaly_mask': tensor [1, H, W],  # 二值 mask (0=正常, 1=异常)
            'meta_info': {
                'image_path': str,
                'mask_path': str,
                'image_name': str,
                'image_id': int,
                # 预留字段 (兼容 mixed-scale benchmark)
                'scale_info': {
                    'original_size': tuple,  # 原始图像尺寸
                    'scale_factor': float,   # 缩放因子
                },
                'statistics': {
                    'anomaly_ratio': float,  # 异常区域占比
                    'num_components': int,  # 异常连通域数量
                }
            }
        }
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        img_dir: str = 'images',
        mask_dir: str = 'anomaly_masks',
        mask_thresh: int = 1,
        validate_mask_exists: bool = True,
        use_flat_dir: bool = False,
        train_ratio: float = 0.9,
        seed: int = 42
    ):
        """
        Args:
            data_root: 数据根目录
            split: 数据集划分 (train/val/test)
            transform: 数据增强
            img_dir: 图像目录名
            mask_dir: mask 目录名
            mask_thresh: mask 阈值，大于等于该值视为异常
            validate_mask_exists: 是否验证 mask 文件存在
            use_flat_dir: 是否使用扁平目录结构 (图像和 mask 在同一目录)
            train_ratio: 训练集比例 (当 use_flat_dir=True 时有效)
            seed: 随机种子 (当 use_flat_dir=True 时有效)
        """
        self.img_dir = os.path.join(data_root, img_dir)
        self.mask_dir = os.path.join(data_root, mask_dir)
        self.mask_thresh = mask_thresh
        self.validate_mask_exists = validate_mask_exists
        self.use_flat_dir = use_flat_dir
        self.train_ratio = train_ratio
        self.seed = seed
        super().__init__(data_root, split, transform, num_classes=2)

    def load_data_list(self) -> List[Dict[str, str]]:
        """加载数据列表"""
        data_list = []

        # 根据目录结构类型选择加载方式
        if self.use_flat_dir:
            # 扁平目录结构: 图像和 mask 在同一目录
            data_list = self._load_flat_dir_list()
        else:
            # 子目录结构: images/train, images/val, masks/train, masks/val
            data_list = self._load_subdir_list()

        return data_list

    def _load_subdir_list(self) -> List[Dict[str, str]]:
        """加载子目录结构的数据列表"""
        data_list = []
        split_img_dir = os.path.join(self.img_dir, self.split)
        split_mask_dir = os.path.join(self.mask_dir, self.split)

        if not os.path.exists(split_img_dir):
            print(f"Warning: Image directory not found: {split_img_dir}")
            return data_list

        for img_name in sorted(os.listdir(split_img_dir)):
            if not img_name.endswith(('.png', '.jpg', '.jpeg')):
                continue

            img_path = os.path.join(split_img_dir, img_name)
            base_name = os.path.splitext(img_name)[0]

            # 查找 mask 文件
            mask_path = None
            for ext in ['.png', '.jpg', '.jpeg']:
                potential_mask_path = os.path.join(split_mask_dir, base_name + ext)
                if os.path.exists(potential_mask_path):
                    mask_path = potential_mask_path
                    break

            if self.validate_mask_exists and mask_path is None:
                continue

            data_list.append({
                'image_path': img_path,
                'mask_path': mask_path,
                'image_name': base_name,
                'image_id': len(data_list)
            })

        return data_list

    def _load_flat_dir_list(self) -> List[Dict[str, str]]:
        """加载扁平目录结构的数据列表"""
        import random
        random.seed(self.seed)

        data_list = []

        if not os.path.exists(self.img_dir):
            print(f"Warning: Image directory not found: {self.img_dir}")
            return data_list

        # 获取所有图像文件
        all_files = []
        for img_name in os.listdir(self.img_dir):
            if not img_name.endswith(('.png', '.jpg', '.jpeg')):
                continue

            img_path = os.path.join(self.img_dir, img_name)
            base_name = os.path.splitext(img_name)[0]

            # 查找对应的 mask 文件
            mask_path = None
            for ext in ['.png', '.jpg', '.jpeg']:
                potential_mask_path = os.path.join(self.mask_dir, base_name + ext)
                if os.path.exists(potential_mask_path):
                    mask_path = potential_mask_path
                    break

            if self.validate_mask_exists and mask_path is None:
                continue

            all_files.append({
                'image_path': img_path,
                'mask_path': mask_path,
                'image_name': base_name,
            })

        # 随机划分训练集和验证集
        random.shuffle(all_files)
        split_idx = int(len(all_files) * self.train_ratio)

        if self.split == 'train':
            selected_files = all_files[:split_idx]
        else:  # val
            selected_files = all_files[split_idx:]

        # 添加 image_id
        for idx, file_info in enumerate(selected_files):
            file_info['image_id'] = idx
            data_list.append(file_info)

        return data_list

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本。

        Args:
            idx: 样本索引

        Returns:
            {
                'image': tensor [3, H, W],
                'anomaly_mask': tensor [1, H, W],
                'meta_info': dict
            }
        """
        data_info = self.data_list[idx]

        # 读取图像
        image = Image.open(data_info['image_path']).convert('RGB')
        original_size = image.size  # (W, H)

        # 读取 anomaly mask
        anomaly_mask = None
        if data_info['mask_path'] and os.path.exists(data_info['mask_path']):
            mask = Image.open(data_info['mask_path'])
            mask_np = np.array(mask)

            # 二值化: >= mask_thresh 为异常 (1), 否则为正常 (0)
            anomaly_mask_np = (mask_np >= self.mask_thresh).astype(np.uint8)
            anomaly_mask = Image.fromarray(anomaly_mask_np)
        else:
            # 如果没有 mask，创建全零 mask
            anomaly_mask = Image.fromarray(np.zeros(image.size[::-1], dtype=np.uint8))

        # 记录统计信息
        mask_np = np.array(anomaly_mask)
        anomaly_ratio = mask_np.sum() / mask_np.size if mask_np.size > 0 else 0.0
        num_components = self._count_components(mask_np)

        # 应用 transform
        if self.transform:
            # Compose 期望字典格式: {'image': ..., 'mask': ...}
            transformed = self.transform(
                {'image': image, 'mask': anomaly_mask}
            )
            image = transformed['image']
            anomaly_mask = transformed.get('mask')

        # 转换为 tensor
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()

        if not isinstance(anomaly_mask, torch.Tensor):
            anomaly_mask = torch.from_numpy(np.array(anomaly_mask)).unsqueeze(0).float()

        # 构建 meta_info
        meta_info = {
            'image_path': data_info['image_path'],
            'mask_path': data_info.get('mask_path'),
            'image_name': data_info['image_name'],
            'image_id': data_info['image_id'],
            # 预留字段 (兼容 mixed-scale benchmark)
            'scale_info': {
                'original_size': original_size,  # (W, H)
                'scale_factor': 1.0,
            },
            'statistics': {
                'anomaly_ratio': float(anomaly_ratio),
                'num_components': num_components,
            }
        }

        return {
            'image': image,
            'anomaly_mask': anomaly_mask,
            'meta_info': meta_info
        }

    def _count_components(self, mask: np.ndarray) -> int:
        """计算连通域数量 (简单的连通域计数)"""
        if mask is None or mask.size == 0:
            return 0

        from scipy import ndimage
        labeled_array, num_features = ndimage.label(mask)
        return int(num_features)

    @staticmethod
    def build_from_config(config: Dict[str, Any], split: str) -> 'AnomalySegmentationDataset':
        """
        从配置构建数据集。

        Args:
            config: 数据集配置
                dataset:
                  data_root: str
                  img_dir: str (可选)
                  mask_dir: str (可选)
                  mask_thresh: int (可选)
            split: 数据集划分

        Returns:
            AnomalySegmentationDataset 实例
        """
        dataset_cfg = config.get('dataset', {})
        anomaly_cfg = config.get('anomaly_dataset', {})

        data_root = dataset_cfg.get('data_root', 'anomaly_data')

        return AnomalySegmentationDataset(
            data_root=data_root,
            split=split,
            transform=None,  # transform 在外部构建
            img_dir=anomaly_cfg.get('img_dir', 'images'),
            mask_dir=anomaly_cfg.get('mask_dir', 'anomaly_masks'),
            mask_thresh=anomaly_cfg.get('mask_thresh', 1),
            validate_mask_exists=anomaly_cfg.get('validate_mask_exists', True)
        )


def build_anomaly_dataloader(
    data_root: str,
    split: str = 'train',
    batch_size: int = 8,
    num_workers: int = 4,
    transform: Optional[Callable] = None,
    img_dir: str = 'images',
    mask_dir: str = 'anomaly_masks',
    shuffle: bool = True,
    mask_thresh: int = 1
):
    """
    构建 Anomaly Segmentation DataLoader。

    Args:
        data_root: 数据根目录
        split: 数据集划分
        batch_size: 批次大小
        num_workers: 工作进程数
        transform: 数据增强
        img_dir: 图像目录名
        mask_dir: mask 目录名
        shuffle: 是否打乱
        mask_thresh: mask 阈值

    Returns:
        DataLoader
    """
    from torch.utils.data import DataLoader

    dataset = AnomalySegmentationDataset(
        data_root=data_root,
        split=split,
        transform=transform,
        img_dir=img_dir,
        mask_dir=mask_dir,
        mask_thresh=mask_thresh
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
