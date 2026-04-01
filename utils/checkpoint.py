"""
训练检查点工具 - 保存和加载模型权重
"""

import os
import torch
from typing import Dict, Any, Optional


def save_checkpoint(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any = None,
    config: Dict[str, Any] = None,
    save_path: str = "checkpoints",
    filename: str = None,
    **kwargs
):
    """
    保存训练检查点。

    Args:
        epoch: 当前 epoch 数
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器（可选）
        config: 配置字典（可选）
        save_path: 保存目录
        filename: 文件名（可选）
        **kwargs: 其他要保存的内容
    """
    os.makedirs(save_path, exist_ok=True)

    if filename is None:
        filename = f"checkpoint_epoch_{epoch}.pth"

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    if config is not None:
        checkpoint['config'] = config

    checkpoint.update(kwargs)

    save_file = os.path.join(save_path, filename)
    torch.save(checkpoint, save_file)
    return save_file


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Any = None,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    加载训练检查点。

    Args:
        checkpoint_path: 检查点路径
        model: 模型
        optimizer: 优化器（可选）
        scheduler: 学习率调度器（可选）
        device: 设备

    Returns:
        检查点字典
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint
