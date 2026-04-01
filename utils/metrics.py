"""
分割指标计算工具
"""

import torch
import numpy as np
from typing import List, Dict


def compute_miou(
    pred: torch.Tensor,
    label: torch.Tensor,
    num_classes: int = 19,
    ignore_index: int = 255
) -> Dict[str, float]:
    """
    计算 mIoU (mean Intersection over Union)

    Args:
        pred: 预测结果 [N, H, W] 或 [N, C, H, W] (logits or argmax)
        label: 标签 [N, H, W]
        num_classes: 类别数
        ignore_index: 忽略的类别ID

    Returns:
        mIoU 和各类别 IoU
    """
    # 如果是 logits，先 argmax
    if pred.dim() == 4:
        pred = pred.argmax(dim=1)

    pred = pred.flatten()
    label = label.flatten()

    # 过滤掉 ignore_index
    mask = (label != ignore_index)
    pred = pred[mask]
    label = label[mask]

    # 计算每个类别的像素数
    iou_list = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        label_cls = (label == cls)

        intersection = (pred_cls & label_cls).sum().item()
        union = (pred_cls | label_cls).sum().item()

        if union == 0:
            iou = float('nan')  # 该类别不存在于标签中
        else:
            iou = intersection / union

        iou_list.append(iou)

    # 计算 mIoU (忽略 NaN)
    valid_ious = [iou for iou in iou_list if not np.isnan(iou)]
    miou = np.mean(valid_ious) if valid_ious else 0.0

    return {
        'mIoU': miou,
        'iou_per_class': iou_list
    }


def compute_pixel_accuracy(
    pred: torch.Tensor,
    label: torch.Tensor,
    ignore_index: int = 255
) -> float:
    """
    计算像素准确率 (Pixel Accuracy)

    Args:
        pred: 预测结果 [N, H, W] 或 [N, C, H, W]
        label: 标签 [N, H, W]
        ignore_index: 忽略的类别ID

    Returns:
        像素准确率
    """
    # 如果是 logits，先 argmax
    if pred.dim() == 4:
        pred = pred.argmax(dim=1)

    pred = pred.flatten()
    label = label.flatten()

    # 过滤掉 ignore_index
    mask = (label != ignore_index)
    pred = pred[mask]
    label = label[mask]

    correct = (pred == label).sum().item()
    total = label.numel()

    if total == 0:
        return 0.0

    return correct / total


def compute_segmentation_metrics(
    preds: List[torch.Tensor],
    labels: List[torch.Tensor],
    num_classes: int = 19,
    ignore_index: int = 255
) -> Dict[str, float]:
    """
    计算分割任务的完整指标

    Args:
        preds: 预测结果列表
        labels: 标签列表
        num_classes: 类别数
        ignore_index: 忽略的类别ID

    Returns:
        指标字典
    """
    # 合并所有 batch 的预测和标签
    all_preds = torch.cat(preds, dim=0)
    all_labels = torch.cat(labels, dim=0)

    # 计算 mIoU
    miou_result = compute_miou(all_preds, all_labels, num_classes, ignore_index)

    # 计算像素准确率
    pixel_acc = compute_pixel_accuracy(all_preds, all_labels, ignore_index)

    # 返回结果
    return {
        'mIoU': miou_result['mIoU'],
        'pixel_acc': pixel_acc,
        'iou_per_class': miou_result['iou_per_class']
    }


class MetricsTracker:
    """
    指标跟踪器 - 用于跟踪训练过程中的指标变化
    """

    def __init__(self, num_classes: int = 19, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        """重置所有状态"""
        self.all_preds = []
        self.all_labels = []
        self.total_loss = 0.0
        self.num_samples = 0

    def update(self, preds: torch.Tensor, labels: torch.Tensor, loss: float = None):
        """
        更新指标

        Args:
            preds: 预测结果 [N, H, W] 或 [N, C, H, W]
            labels: 标签 [N, H, W]
            loss: 损失值
        """
        self.all_preds.append(preds.cpu())
        self.all_labels.append(labels.cpu())

        if loss is not None:
            self.total_loss += loss
            self.num_samples += preds.shape[0]

    def compute(self) -> Dict[str, float]:
        """
        计算当前累积的指标

        Returns:
            指标字典
        """
        if len(self.all_preds) == 0:
            return {'mIoU': 0.0, 'pixel_acc': 0.0, 'loss': 0.0, 'iou_per_class': []}

        # 合并所有预测和标签
        all_preds = torch.cat(self.all_preds, dim=0)
        all_labels = torch.cat(self.all_labels, dim=0)

        # 计算指标
        metrics = compute_segmentation_metrics(
            [all_preds],
            [all_labels],
            self.num_classes,
            self.ignore_index
        )

        # 保存每类 IoU 供外部访问
        self.iou_per_class = metrics.get('iou_per_class', [])
        self.mIoU = metrics.get('mIoU', 0.0)

        # 添加损失
        metrics['loss'] = self.total_loss / max(self.num_samples, 1)
        if self.num_samples > 0:
            metrics['loss'] = self.total_loss / self.num_samples

        return metrics
