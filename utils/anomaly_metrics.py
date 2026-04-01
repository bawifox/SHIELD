"""
Anomaly Segmentation Metrics
- AP (Average Precision)
- AUROC (Area Under ROC Curve)
- AUPR (Area Under Precision-Recall Curve)
"""

import torch
import numpy as np
from typing import List, Dict, Tuple


def compute_ap(precision: np.ndarray, recall: np.ndarray) -> float:
    """
    计算 Average Precision (AP)

    Args:
        precision: precision values
        recall: recall values

    Returns:
        AP value
    """
    # 添加首尾点
    recall = np.concatenate([[0.], recall, [1.]])
    precision = np.concatenate([[0.], precision, [0.]])

    # 降序排列
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    # 计算 AP
    indices = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])

    return ap


def compute_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    计算 AUROC (Area Under ROC Curve)

    Args:
        y_true: ground truth binary labels (0 or 1)
        y_score: predicted scores/probabilities

    Returns:
        AUROC value
    """
    y_true = y_true.flatten()
    y_score = y_score.flatten()

    # 排序
    sorted_indices = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[sorted_indices]

    # 计算 TPR 和 FPR
    pos_count = y_true.sum()
    neg_count = len(y_true) - pos_count

    if pos_count == 0 or neg_count == 0:
        return 0.0

    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)

    tpr = tps / pos_count
    fpr = fps / neg_count

    # 添加起点
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])

    # 计算 AUC
    auroc = np.trapz(tpr, fpr)

    return auroc


def compute_aupr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    计算 AUPR (Area Under Precision-Recall Curve)

    Args:
        y_true: ground truth binary labels (0 or 1)
        y_score: predicted scores/probabilities

    Returns:
        AUPR value
    """
    y_true = y_true.flatten()
    y_score = y_score.flatten()

    # 排序
    sorted_indices = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[sorted_indices]

    # 计算 precision 和 recall
    tp = np.cumsum(y_true_sorted)
    fp = np.cumsum(1 - y_true_sorted)

    precision = tp / (tp + fp)
    recall = tp / y_true.sum()

    # 添加起点
    precision = np.concatenate([[1], precision])
    recall = np.concatenate([[0], recall])

    # 计算 AUPR
    aupr = np.trapz(precision, recall)

    return aupr


def compute_fpr95(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    计算 FPR@95 (False Positive Rate at 95% True Positive Rate)

    Args:
        y_true: ground truth binary labels (0 or 1)
        y_score: predicted scores/probabilities

    Returns:
        FPR@95 value
    """
    y_true = y_true.flatten()
    y_score = y_score.flatten()

    # 排序
    sorted_indices = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[sorted_indices]

    # 计算 TPR 和 FPR
    pos_count = y_true.sum()
    neg_count = len(y_true) - pos_count

    if pos_count == 0 or neg_count == 0:
        return 1.0 if pos_count == 0 else 0.0

    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)

    tpr = tps / pos_count
    fpr = fps / neg_count

    # 找到 TPR >= 0.95 的最小阈值点
    idx = np.searchsorted(tpr, 0.95, side='right')
    if idx >= len(fpr):
        return fpr[-1]
    return float(fpr[idx])


def compute_anomaly_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    计算 anomaly segmentation 指标

    Args:
        logits: 预测 logits [B, 1, H, W]
        targets: ground truth masks [B, 1, H, W]
        threshold: 二值化阈值

    Returns:
        指标字典
    """
    # 转换为 numpy
    logits_np = logits.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()

    # 转换为概率 (sigmoid)
    probs = 1 / (1 + np.exp(-logits_np))

    # 二值化预测
    preds = (probs >= threshold).astype(np.float32)

    # 计算各项指标
    # Pixel-level metrics
    tp = ((preds == 1) & (targets_np == 1)).sum()
    fp = ((preds == 1) & (targets_np == 0)).sum()
    tn = ((preds == 0) & (targets_np == 0)).sum()
    fn = ((preds == 0) & (targets_np == 1)).sum()

    # Accuracy
    accuracy = (tp + tn) / (tp + fp + tn + fn + 1e-8)

    # Precision, Recall, F1
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # IoU
    iou = tp / (tp + fp + fn + 1e-8)

    # AUROC, AUPR, FPR95
    auroc = compute_auroc(targets_np, probs)
    aupr = compute_aupr(targets_np, probs)
    fpr95 = compute_fpr95(targets_np, probs)

    # AP (使用多个阈值)
    thresholds = np.linspace(0, 1, 101)
    precisions = []
    recalls = []
    for thresh in thresholds:
        preds_thresh = (probs >= thresh).astype(np.float32)
        tp_t = ((preds_thresh == 1) & (targets_np == 1)).sum()
        fp_t = ((preds_thresh == 1) & (targets_np == 0)).sum()
        fn_t = ((preds_thresh == 0) & (targets_np == 1)).sum()

        p = tp_t / (tp_t + fp_t + 1e-8)
        r = tp_t / (tp_t + fn_t + 1e-8)

        precisions.append(p)
        recalls.append(r)

    ap = compute_ap(np.array(precisions), np.array(recalls))

    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'iou': float(iou),
        'auroc': float(auroc),
        'aupr': float(aupr),
        'ap': float(ap),
        'fpr95': float(fpr95)
    }


class AnomalyMetricsTracker:
    """
    Anomaly Metrics Tracker
    用于跟踪训练过程中的指标变化
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """重置所有状态"""
        self.all_logits = []
        self.all_targets = []
        self.total_loss = 0.0
        self.num_samples = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor, loss: float = None):
        """
        更新指标

        Args:
            logits: 预测 logits [B, 1, H, W]
            targets: ground truth masks [B, 1, H, W]
            loss: 损失值
        """
        self.all_logits.append(logits.detach().cpu())
        self.all_targets.append(targets.detach().cpu())

        if loss is not None:
            self.total_loss += loss
            self.num_samples += logits.shape[0]

    def compute(self) -> Dict[str, float]:
        """
        计算当前累积的指标

        Returns:
            指标字典
        """
        if len(self.all_logits) == 0:
            return {
                'loss': 0.0,
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'iou': 0.0,
                'auroc': 0.0,
                'aupr': 0.0,
                'ap': 0.0,
                'fpr95': 1.0
            }

        # 合并所有预测和标签
        all_logits = torch.cat(self.all_logits, dim=0)
        all_targets = torch.cat(self.all_targets, dim=0)

        # 计算指标
        metrics = compute_anomaly_metrics(all_logits, all_targets)

        # 添加损失
        if self.num_samples > 0:
            metrics['loss'] = self.total_loss / self.num_samples

        return metrics
