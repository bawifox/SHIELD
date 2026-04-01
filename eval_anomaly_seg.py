"""
Anomaly Segmentation Evaluation Script
支持按不同子集统计评估结果
"""

import os
import sys
import argparse
import json
import csv
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config
from utils.logger import setup_logger
from utils.anomaly_metrics import compute_anomaly_metrics
from models.anomaly_segmentation import AnomalySegmentationModel, AnomalySegmentationModelWithPrior
from datasets import AnomalySegmentationDataset, build_val_transforms


@dataclass
class EvaluationResult:
    """评估结果数据结构"""
    subset_name: str
    
    # Pixel-level metrics
    pixel_accuracy: float = 0.0
    pixel_precision: float = 0.0
    pixel_recall: float = 0.0
    pixel_f1: float = 0.0
    
    # Region-level metrics
    iou: float = 0.0
    
    # Threshold-invariant metrics
    auroc: float = 0.0
    aupr: float = 0.0
    ap: float = 0.0
    
    # Additional info
    num_samples: int = 0
    num_anomaly_pixels: int = 0
    total_pixels: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'subset': self.subset_name,
            'pixel_accuracy': self.pixel_accuracy,
            'pixel_precision': self.pixel_precision,
            'pixel_recall': self.pixel_recall,
            'pixel_f1': self.pixel_f1,
            'iou': self.iou,
            'auroc': self.auroc,
            'aupr': self.aupr,
            'ap': self.ap,
            'num_samples': self.num_samples,
            'num_anomaly_pixels': self.num_anomaly_pixels,
            'total_pixels': self.total_pixels
        }
    
    def to_csv_row(self) -> Dict[str, Any]:
        return self.to_dict()
    
    def __str__(self) -> str:
        return (f"{self.subset_name:20s} | "
                f"Acc: {self.pixel_accuracy:.4f} | "
                f"Prec: {self.pixel_precision:.4f} | "
                f"Rec: {self.pixel_recall:.4f} | "
                f"F1: {self.pixel_f1:.4f} | "
                f"IoU: {self.iou:.4f} | "
                f"AUROC: {self.auroc:.4f} | "
                f"AUPR: {self.aupr:.4f} | "
                f"AP: {self.ap:.4f}")


@dataclass
class SampleMetadata:
    """样本元数据，用于子集划分"""
    image_path: str
    mask_path: str
    image_name: str
    image_id: int
    
    # 统计信息
    anomaly_area: int = 0
    anomaly_ratio: float = 0.0
    num_components: int = 0
    
    # 可选的额外信息
    extra: Dict[str, Any] = field(default_factory=dict)


class SubsetClassifier:
    """
    子集分类器
    用于将样本划分到不同的子集
    """
    
    def __init__(
        self,
        small_threshold: int = 500,
        medium_threshold: int = 5000,
        mixed_scale_subsets: Optional[List[str]] = None
    ):
        """
        Args:
            small_threshold: 小目标面积阈值 (像素数)
            medium_threshold: 中目标面积阈值 (像素数)
            mixed_scale_subsets: mixed-scale benchmark 的子集名称列表
        """
        self.small_threshold = small_threshold
        self.medium_threshold = medium_threshold
        self.mixed_scale_subsets = mixed_scale_subsets or []
    
    def classify_by_size(self, anomaly_area: int) -> str:
        """
        根据异常区域面积分类
        
        Args:
            anomaly_area: 异常区域面积 (像素数)
            
        Returns:
            'small', 'medium', 或 'large'
        """
        if anomaly_area < self.small_threshold:
            return 'small'
        elif anomaly_area < self.medium_threshold:
            return 'medium'
        else:
            return 'large'
    
    def get_subsets(self, metadata: SampleMetadata) -> List[str]:
        """
        获取样本所属的所有子集
        
        Args:
            metadata: 样本元数据
            
        Returns:
            子集名称列表
        """
        subsets = []
        
        # 1. 整体 (all)
        subsets.append('all')
        
        # 2. 按大小分类
        if metadata.anomaly_area > 0:
            size_subset = self.classify_by_size(metadata.anomaly_area)
            subsets.append(size_subset)
            
            # 可选: small + medium + large
            subsets.extend(['small', 'medium', 'large'])
        
        # 3. Mixed-scale benchmark 分类 (如果有)
        # 这里可以扩展: 根据 extra 信息或文件名匹配
        if metadata.extra.get('benchmark_type'):
            subsets.append(metadata.extra['benchmark_type'])
        
        # 4. 其他自定义分类可以在这里添加
        # 例如: 根据 anomaly_ratio 分类
        if metadata.anomaly_ratio > 0 and metadata.anomaly_ratio < 0.05:
            subsets.append('rare')
        
        return list(set(subsets))  # 去重


class AnomalySegEvaluator:
    """
    Anomaly Segmentation 评估器
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = 'cuda',
        threshold: float = 0.5,
        subset_classifier: Optional[SubsetClassifier] = None
    ):
        """
        Args:
            model: 模型
            device: 设备
            threshold: 二值化阈值
            subset_classifier: 子集分类器
        """
        self.model = model
        self.device = device
        self.threshold = threshold
        self.subset_classifier = subset_classifier or SubsetClassifier()
        
        # 存储所有样本的预测和标签
        self.all_logits = []
        self.all_targets = []
        self.all_metadata = []
    
    def add_batch(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        metadata: Optional[List[SampleMetadata]] = None
    ):
        """
        添加一个 batch 的结果
        
        Args:
            logits: 预测 logits [B, 1, H, W]
            targets: GT masks [B, 1, H, W]
            metadata: 样本元数据列表
        """
        self.all_logits.append(logits.detach().cpu())
        self.all_targets.append(targets.detach().cpu())
        
        if metadata is not None:
            self.all_metadata.extend(metadata)
    
    def evaluate(self) -> Dict[str, EvaluationResult]:
        """
        执行评估
        
        Returns:
            子集名称 -> 评估结果 的字典
        """
        if len(self.all_logits) == 0:
            return {}
        
        # 合并所有数据
        all_logits = torch.cat(self.all_logits, dim=0)
        all_targets = torch.cat(self.all_targets, dim=0)
        
        # 初始化结果
        results = {}
        
        # 1. 评估整体
        results['all'] = self._compute_metrics(
            all_logits, all_targets, 'all'
        )
        
        # 2. 按子集评估 (如果有元数据)
        if len(self.all_metadata) > 0:
            # 创建子集索引映射
            subset_indices = self._build_subset_indices()
            
            for subset_name, indices in subset_indices.items():
                if len(indices) == 0:
                    continue
                    
                subset_logits = all_logits[indices]
                subset_targets = all_targets[indices]
                
                results[subset_name] = self._compute_metrics(
                    subset_logits, subset_targets, subset_name
                )
        
        return results
    
    def _compute_metrics(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        subset_name: str
    ) -> EvaluationResult:
        """计算指标"""
        metrics = compute_anomaly_metrics(logits, targets, self.threshold)
        
        # 计算异常像素统计
        targets_np = targets.numpy()
        num_anomaly = int((targets_np == 1).sum())
        total_pixels = int(targets_np.size)
        
        return EvaluationResult(
            subset_name=subset_name,
            pixel_accuracy=metrics['accuracy'],
            pixel_precision=metrics['precision'],
            pixel_recall=metrics['recall'],
            pixel_f1=metrics['f1'],
            iou=metrics['iou'],
            auroc=metrics['auroc'],
            aupr=metrics['aupr'],
            ap=metrics['ap'],
            num_samples=logits.shape[0],
            num_anomaly_pixels=num_anomaly,
            total_pixels=total_pixels
        )
    
    def _build_subset_indices(self) -> Dict[str, List[int]]:
        """构建子集索引映射"""
        subset_indices = {}
        
        for idx, meta in enumerate(self.all_metadata):
            subsets = self.subset_classifier.get_subsets(meta)
            
            for subset in subsets:
                if subset not in subset_indices:
                    subset_indices[subset] = []
                subset_indices[subset].append(idx)
        
        return subset_indices
    
    def reset(self):
        """重置评估器"""
        self.all_logits = []
        self.all_targets = []
        self.all_metadata = []


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Anomaly Segmentation Evaluation")
    
    # Model & Config
    parser.add_argument("--config", type=str, default="configs/shield_lite.yaml",
                       help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use")
    
    # Data
    parser.add_argument("--split", type=str, default="val",
                       help="Dataset split (train/val/test)")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of workers")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to evaluate")
    
    # Threshold
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Binary threshold")
    
    # Subset classification
    parser.add_argument("--small_threshold", type=int, default=500,
                       help="Small anomaly area threshold (pixels)")
    parser.add_argument("--medium_threshold", type=int, default=5000,
                       help="Medium anomaly area threshold (pixels)")
    
    # Output
    parser.add_argument("--output", type=str, default=None,
                       help="Output file path")
    parser.add_argument("--output_format", type=str, default="json",
                       choices=["json", "csv", "both"],
                       help="Output format")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed results")
    
    args = parser.parse_args()
    return args


def collate_fn(batch):
    """自定义 collate 函数"""
    image = torch.stack([item['image'] for item in batch])
    anomaly_mask = torch.stack([item['anomaly_mask'] for item in batch])
    meta_info = [item['meta_info'] for item in batch]
    return {
        'image': image,
        'anomaly_mask': anomaly_mask,
        'meta_info': meta_info
    }


def build_dataloader(config, split='val', batch_size=4, num_workers=4):
    """构建数据加载器"""
    dataset_cfg = config.get('dataset', {})
    anomaly_cfg = config.get('anomaly_dataset', {})
    data_root = anomaly_cfg.get('data_root', dataset_cfg.get('data_root', 'train_set'))

    transform = build_val_transforms(config) if config.get('transforms', {}).get('val') else None

    dataset = AnomalySegmentationDataset(
        data_root=data_root,
        split=split,
        transform=transform,
        img_dir=anomaly_cfg.get('img_dir', 'synthetic_images'),
        mask_dir=anomaly_cfg.get('mask_dir', 'synthetic_masks'),
        mask_thresh=anomaly_cfg.get('mask_thresh', 1),
        use_flat_dir=anomaly_cfg.get('use_flat_dir', True),
        train_ratio=anomaly_cfg.get('train_ratio', 0.9),
        seed=anomaly_cfg.get('seed', 42)
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    return loader


def build_model(config, checkpoint_path: str, device: str):
    """构建模型"""
    model_cfg = config.get('model', {})
    anomaly_cfg = config.get('anomaly_model', {})
    
    pretrained_path = anomaly_cfg.get('pretrained_backbone_path')
    if not pretrained_path and 'checkpoint' in config:
        save_dir = config['checkpoint'].get('save_dir', 'checkpoints/cityscapes')
        pretrained_path = f"{save_dir}/backbone_weights.pth"
    
    # 尝试加载带 prior 的模型
    try:
        model = AnomalySegmentationModelWithPrior(
            in_channels=model_cfg.get('in_channels', 3),
            decoder_dim=anomaly_cfg.get('decoder_dim', 256),
            pretrained_backbone_path=pretrained_path,
            use_hazard_scorer=True,
            use_small_hazard_prior=True,
            prior_fusion_mode='add'
        )
    except:
        # 回退到基础模型
        model = AnomalySegmentationModel(
            in_channels=model_cfg.get('in_channels', 3),
            decoder_dim=anomaly_cfg.get('decoder_dim', 256),
            pretrained_backbone_path=pretrained_path
        )
    
    # 加载 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


def extract_metadata(batch, meta_info_list: List[dict]) -> List[SampleMetadata]:
    """从 batch 中提取元数据"""
    metadata = []
    
    for i, meta in enumerate(meta_info_list):
        # 计算 anomaly area
        anomaly_mask = batch['anomaly_mask'][i].numpy()
        anomaly_area = int((anomaly_mask == 1).sum())
        total_pixels = anomaly_mask.size
        anomaly_ratio = anomaly_area / total_pixels if total_pixels > 0 else 0.0
        
        metadata.append(SampleMetadata(
            image_path=meta.get('image_path', ''),
            mask_path=meta.get('mask_path', ''),
            image_name=meta.get('image_name', ''),
            image_id=meta.get('image_id', i),
            anomaly_area=anomaly_area,
            anomaly_ratio=anomaly_ratio,
            num_components=meta.get('statistics', {}).get('num_components', 0),
            extra={
                'scale_info': meta.get('scale_info', {}),
                'statistics': meta.get('statistics', {})
            }
        ))
    
    return metadata


def save_results(results: Dict[str, EvaluationResult], output_path: str, format: str = 'json'):
    """保存评估结果"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if format in ['json', 'both']:
        json_path = output_path if format == 'json' else output_path.replace('.csv', '.json')
        results_dict = {name: result.to_dict() for name, result in results.items()}
        with open(json_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        print(f"Results saved to {json_path}")
    
    if format in ['csv', 'both']:
        csv_path = output_path if format == 'csv' else output_path.replace('.json', '.csv')
        df = pd.DataFrame([result.to_csv_row() for result in results.values()])
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")


def main():
    """主函数"""
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设备
    device = args.device or config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # 日志
    logger = setup_logger('eval', log_file=None)
    
    logger.info("=" * 60)
    logger.info("Anomaly Segmentation Evaluation")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Device: {device}")
    logger.info(f"Threshold: {args.threshold}")
    
    # 构建数据加载器
    logger.info("Building dataloader...")
    val_loader = build_dataloader(
        config,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    logger.info(f"Total samples: {len(val_loader.dataset)}")
    
    # 构建模型
    logger.info("Building model...")
    model = build_model(config, args.checkpoint, device)
    
    # 子集分类器
    subset_classifier = SubsetClassifier(
        small_threshold=args.small_threshold,
        medium_threshold=args.medium_threshold
    )
    
    # 评估器
    evaluator = AnomalySegEvaluator(
        model=model,
        device=device,
        threshold=args.threshold,
        subset_classifier=subset_classifier
    )
    
    # 评估
    logger.info("Running evaluation...")
    pbar = tqdm(val_loader, desc="Evaluating")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            if args.max_samples is not None and batch_idx * args.batch_size >= args.max_samples:
                break
            
            images = batch['image'].to(device)
            anomaly_masks = batch['anomaly_mask'].to(device)
            
            # Forward
            if hasattr(model, 'forward_with_hazard_prior'):
                # 需要 candidate masks，简化处理
                outputs = model(images, return_details=False)
            else:
                outputs = model(images)
            
            # 提取元数据
            metadata = extract_metadata(batch, batch['meta_info'])
            
            # 添加到评估器
            evaluator.add_batch(
                outputs.final_logits if hasattr(outputs, 'final_logits') else outputs[1],
                anomaly_masks,
                metadata
            )
    
    # 计算结果
    logger.info("Computing metrics...")
    results = evaluator.evaluate()
    
    # 打印结果
    print("\n" + "=" * 100)
    print("Evaluation Results")
    print("=" * 100)
    
    # 按优先级排序输出
    priority_subsets = ['all', 'small', 'medium', 'large']
    other_subsets = sorted([s for s in results.keys() if s not in priority_subsets])
    ordered_subsets = [s for s in priority_subsets if s in results] + other_subsets
    
    for subset_name in ordered_subsets:
        result = results[subset_name]
        print(result)
    
    # 保存结果
    if args.output:
        save_results(results, args.output, args.output_format)
    
    logger.info("Evaluation finished!")


if __name__ == '__main__':
    main()
