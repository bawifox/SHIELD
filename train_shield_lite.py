"""
SHIELD-Lite Training Script
支持完整训练：coarse branch + hazard scorer + dynamic threshold + small-hazard prior
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from tqdm import tqdm
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config
from utils.logger import setup_logger
from utils.anomaly_metrics import AnomalyMetricsTracker
from utils.candidate_extractor import CandidateExtractor, CandidateExtractorBatch
from models.anomaly_segmentation import AnomalySegmentationModelWithPrior, AnomalySegmentationOutput
from datasets import AnomalySegmentationDataset, build_train_transforms, build_val_transforms
from losses.hazard_losses import AnomalySegmentationTotalLoss


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="SHIELD-Lite Training")
    parser.add_argument("--config", type=str, default="configs/anomaly_seg.yaml",
                       help="Path to config file")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode (visualize every N batches)")
    args, unknown = parser.parse_known_args()
    
    # 从环境变量获取 local_rank (torchrun 会设置这个)
    import os
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ.get('LOCAL_RANK', -1))
    else:
        args.local_rank = -1
    
    # 初始化
    args.local_rank = getattr(args, 'local_rank', -1)
    
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


def build_dataloader(config, split='train', rank=-1, world_size=1):
    """构建数据加载器
    
    Args:
        config: 配置字典
        split: 'train' or 'val'
        rank: 当前进程编号 (分布式训练)
        world_size: 总进程数 (分布式训练)
    """
    dataset_cfg = config.get('dataset', {})
    anomaly_cfg = config.get('anomaly_dataset', {})
    data_root = anomaly_cfg.get('data_root', dataset_cfg.get('data_root', 'train_set'))

    if split == 'train':
        transform = build_train_transforms(config) if config.get('transforms', {}).get('train') else None
    else:
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

    batch_size = config['train'].get('batch_size', 4)
    num_workers = config['train'].get('num_workers', 4)

    # 分布式训练使用 DistributedSampler
    if world_size > 1 and split == 'train':
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=config.get('anomaly_dataset', {}).get('seed', 42)
        )
        shuffle = False
    elif split == 'train':
        sampler = RandomSampler(dataset)
        shuffle = True
    else:
        sampler = SequentialSampler(dataset)
        shuffle = False

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train'),
        collate_fn=collate_fn,
        sampler=sampler
    )

    return loader


def build_model(config, ablation_config=None):
    """构建 SHIELD-Lite 模型
    
    Args:
        config: 完整配置字典
        ablation_config: AblationConfig 对象，若为 None 则从 config 读取
    """
    model_cfg = config.get('model', {})
    anomaly_cfg = config.get('anomaly_model', {})
    shield_cfg = config.get('shield_lite', {})
    
    # 处理 ablation 配置
    if ablation_config is None:
        ablation_cfg = config.get('ablation', {})
        ablation_name = config.get('_ablation_name')
        
        # 如果指定了预设名称，则使用预设
        if ablation_name:
            from configs.ablation_config import get_ablation_config
            ablation_config = get_ablation_config(ablation_name)
        elif ablation_cfg:
            # 使用手动配置
            from configs.ablation_config import AblationConfig
            ablation_config = AblationConfig(
                enable_small_candidate_extraction=ablation_cfg.get('enable_small_candidate_extraction', True),
                enable_text_guided_hazard=ablation_cfg.get('enable_text_guided_hazard', False),
                enable_adaptive_threshold=ablation_cfg.get('enable_adaptive_threshold', True),
                enable_small_hazard_prior=ablation_cfg.get('enable_small_hazard_prior', True),
                prior_fusion_mode=ablation_cfg.get('prior_fusion_mode', 'add'),
                hazard_score_type=ablation_cfg.get('hazard_score_type', 'area'),
                lambda_coarse=shield_cfg.get('lambda_coarse', 0.5),
                lambda_hazard=shield_cfg.get('lambda_hazard', 0.5)
            )
        else:
            # 默认配置
            from configs.ablation_config import AblationConfig
            ablation_config = AblationConfig()
    
    pretrained_path = anomaly_cfg.get('pretrained_backbone_path')
    if not pretrained_path and 'checkpoint' in config:
        save_dir = config['checkpoint'].get('save_dir', 'checkpoints/cityscapes')
        pretrained_path = f"{save_dir}/backbone_weights.pth"

    model = AnomalySegmentationModelWithPrior(
        in_channels=model_cfg.get('in_channels', 3),
        decoder_dim=anomaly_cfg.get('decoder_dim', 256),
        pretrained_backbone_path=pretrained_path,
        
        # 根据 ablation_config 设置各个组件
        # Hazard Scorer (包含 adaptive threshold)
        use_hazard_scorer=ablation_config.enable_adaptive_threshold or ablation_config.enable_small_candidate_extraction,
        hazard_beta=shield_cfg.get('hazard_beta', 10.0),
        base_threshold=shield_cfg.get('base_threshold', 0.01),
        
        # Small Hazard Prior
        use_small_hazard_prior=ablation_config.enable_small_hazard_prior,
        prior_fusion_mode=ablation_config.prior_fusion_mode,
        
        # 训练细节
        enable_training_details=True
    )
    
    return model, ablation_config


def build_optimizer(model, config):
    """构建优化器"""
    opt_cfg = config['optimizer']
    opt_type = opt_cfg.get('type', 'AdamW')

    if hasattr(model, 'get_params_groups'):
        param_groups = model.get_params_groups()
    else:
        param_groups = [{'params': model.parameters(), 'lr_mult': 1.0}]

    base_lr = opt_cfg.get('lr', 1e-4)
    params = []
    for group in param_groups:
        params.append({
            'params': group['params'],
            'lr': base_lr * group.get('lr_mult', 1.0),
            'weight_decay': opt_cfg.get('weight_decay', 5e-4)
        })

    if opt_type == 'AdamW':
        optimizer = optim.AdamW(params)
    elif opt_type == 'SGD':
        optimizer = optim.SGD(params, momentum=opt_cfg.get('momentum', 0.9), nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")

    return optimizer


def build_scheduler(optimizer, config):
    """构建学习率调度器"""
    sched_cfg = config.get('scheduler')
    if sched_cfg is None:
        return None

    sched_type = sched_cfg.get('type', 'PolynomialLR')
    num_epochs = config['train'].get('num_epochs', 30)

    if sched_type == 'PolynomialLR':
        from torch.optim.lr_scheduler import PolynomialLR
        scheduler = PolynomialLR(optimizer, total_iters=num_epochs, power=sched_cfg.get('power', 0.9))
    elif sched_type == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif sched_type == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    else:
        scheduler = None

    return scheduler


def build_loss(config, ablation_config=None):
    """构建损失函数
    
    Args:
        config: 完整配置字典
        ablation_config: AblationConfig 对象
    """
    loss_cfg = config.get('loss', {})
    shield_cfg = config.get('shield_lite', {})
    
    # 如果没有传入 ablation_config，从配置读取
    if ablation_config is None:
        ablation_name = config.get('_ablation_name')
        if ablation_name:
            from configs.ablation_config import get_ablation_config
            ablation_config = get_ablation_config(ablation_name)
        else:
            from configs.ablation_config import AblationConfig
            ablation_config = AblationConfig()
    
    # 根据 ablation 配置设置 lambda 权重
    lambda_coarse = ablation_config.lambda_coarse if ablation_config else loss_cfg.get('lambda_coarse', 0.5)
    lambda_hazard = ablation_config.lambda_hazard if ablation_config else loss_cfg.get('lambda_hazard', 0.5)
    
    return AnomalySegmentationTotalLoss(
        bce_weight=loss_cfg.get('bce_weight', 0.5),
        dice_weight=loss_cfg.get('dice_weight', 0.5),
        lambda_coarse=lambda_coarse,
        lambda_hazard=lambda_hazard,
        hazard_overlap_threshold=shield_cfg.get('hazard_overlap_threshold', 0.3),
        hazard_loss_type=loss_cfg.get('hazard_loss_type', 'bce'),
        hazard_pos_weight=loss_cfg.get('hazard_pos_weight', 1.0),
        hazard_neg_weight=loss_cfg.get('hazard_neg_weight', 0.5)
    )


def build_candidate_extractor(config, ablation_config=None):
    """构建候选区域提取器
    
    Args:
        config: 完整配置字典
        ablation_config: AblationConfig 对象
    """
    shield_cfg = config.get('shield_lite', {})
    
    # 如果没有传入 ablation_config，从配置读取
    if ablation_config is None:
        ablation_name = config.get('_ablation_name')
        if ablation_name:
            from configs.ablation_config import get_ablation_config
            ablation_config = get_ablation_config(ablation_name)
        else:
            from configs.ablation_config import AblationConfig
            ablation_config = AblationConfig()
    
    # 如果禁用小候选区域提取，返回 None
    if ablation_config and not ablation_config.enable_small_candidate_extraction:
        return None
    
    return CandidateExtractor(
        threshold_high=shield_cfg.get('candidate_threshold_high', 0.5),
        threshold_small=shield_cfg.get('candidate_threshold_small', 0.3),
        tau_small=shield_cfg.get('candidate_tau_small', 500),
        local_response_thresh=shield_cfg.get('candidate_local_response_thresh', 0.6),
        N_max=shield_cfg.get('max_candidates', 10),
        min_area=shield_cfg.get('min_candidate_area', 50)
    )


def extract_candidates_batch(coarse_prob, extractor, device):
    """
    从 coarse probability map 批量提取候选区域
    
    Args:
        coarse_prob: [B, 1, H, W] tensor
        extractor: CandidateExtractor
        device: torch device
        
    Returns:
        candidate_masks: [B, N, H, W] tensor (padding to N_max)
        area_ratios: [B, N] tensor
    """
    B = coarse_prob.shape[0]
    N_max = extractor.N_max
    
    batch_size = coarse_prob.shape[0]
    H, W = coarse_prob.shape[2], coarse_prob.shape[3]
    
    # 初始化
    candidate_masks = torch.zeros(B, N_max, H, W, device=device)
    area_ratios = torch.zeros(B, N_max, device=device)
    
    for b in range(B):
        prob_map = coarse_prob[b, 0].cpu().numpy()
        candidates = extractor(prob_map)
        
        n_cands = min(len(candidates), N_max)
        for i, cand in enumerate(candidates[:n_cands]):
            candidate_masks[b, i] = torch.from_numpy(cand.mask).float().to(device)
            area_ratios[b, i] = cand.area_ratio
    
    return candidate_masks, area_ratios


def visualize_outputs(outputs, targets, batch_idx, vis_dir, prefix='train', rank=0, epoch=0):
    """
    可视化训练过程中的中间结果
    
    Args:
        outputs: AnomalySegmentationOutput
        targets: dict with 'anomaly_mask'
        batch_idx: batch index
        vis_dir: visualization directory
        prefix: 'train' or 'val'
        rank: 进程编号 (只有 rank 0 执行可视化)
    """
    # 只有主进程执行可视化
    if rank != 0:
        return
    
    import matplotlib.pyplot as plt
    
    B = outputs.coarse_logits.shape[0]
    
    for b in range(min(B, 2)):  # 只可视化前2张
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # 1. Coarse prediction
        coarse_prob = torch.sigmoid(outputs.coarse_logits[b, 0]).cpu().numpy()
        axes[0, 0].imshow(coarse_prob, cmap='jet')
        axes[0, 0].set_title('Coarse Map')
        axes[0, 0].axis('off')
        
        # 2. Small hazard prior
        if outputs.small_hazard_prior is not None:
            prior = outputs.small_hazard_prior[b, 0].cpu().numpy()
            im = axes[0, 1].imshow(prior, cmap='jet')
            axes[0, 1].set_title('Small Hazard Prior')
            axes[0, 1].axis('off')
            plt.colorbar(im, ax=axes[0, 1])
        else:
            axes[0, 1].text(0.5, 0.5, 'No Prior', ha='center', va='center')
            axes[0, 1].set_title('Small Hazard Prior')
            axes[0, 1].axis('off')
        
        # 3. Final prediction
        final_prob = torch.sigmoid(outputs.final_logits[b, 0]).cpu().numpy()
        im = axes[0, 2].imshow(final_prob, cmap='jet')
        axes[0, 2].set_title('Final Prediction')
        axes[0, 2].axis('off')
        plt.colorbar(im, ax=axes[0, 2])
        
        # 4. GT anomaly mask
        gt_mask = targets['anomaly_mask'][b, 0].cpu().numpy()
        axes[1, 0].imshow(gt_mask, cmap='gray')
        axes[1, 0].set_title('GT Anomaly Mask')
        axes[1, 0].axis('off')
        
        # 5. Hazard scores (bar plot)
        if outputs.hazard_scores is not None:
            scores = outputs.hazard_scores[b].cpu().numpy()
            axes[1, 1].bar(range(len(scores)), scores)
            axes[1, 1].set_title(f'Hazard Scores (Δt={outputs.delta_t[b].item():.4f})')
            axes[1, 1].set_xlabel('Candidate')
            axes[1, 1].set_ylabel('Score')
        else:
            axes[1, 1].text(0.5, 0.5, 'No Scores', ha='center', va='center')
            axes[1, 1].set_title('Hazard Scores')
        
        # 6. Delta t
        if outputs.delta_t is not None:
            delta_t_str = f"Delta_t: {outputs.delta_t[b].item():.4f}"
        else:
            delta_t_str = "Delta_t: N/A"
        axes[1, 2].text(0.5, 0.5, delta_t_str, ha='center', va='center', fontsize=14)
        axes[1, 2].set_title('Threshold Offset')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # 保存 (添加 epoch 编号)
        epoch_vis_dir = os.path.join(vis_dir, f'epoch_{epoch:03d}')
        os.makedirs(epoch_vis_dir, exist_ok=True)
        save_path = os.path.join(epoch_vis_dir, f'{prefix}_batch{batch_idx}_img{b}.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()


def train_epoch(model, train_loader, criterion, optimizer, device, logger, epoch, config, candidate_extractor=None, vis_dir=None, debug=False, rank=0):
    """训练一个 epoch"""
    model.train()

    # 损失权重
    loss_cfg = config.get('loss', {})
    shield_cfg = config.get('shield_lite', {})
    
    use_hazard = shield_cfg.get('use_hazard_scorer', True)
    use_prior = shield_cfg.get('use_small_hazard_prior', True)
    vis_interval = config['train'].get('vis_interval', 100)

    # 指标跟踪
    total_metrics = AnomalyMetricsTracker()
    coarse_metrics = AnomalyMetricsTracker()
    final_metrics = AnomalyMetricsTracker()
    
    # 损失统计
    loss_stats = {
        'total': [], 'final': [], 'coarse': [], 'hazard': [],
        'final_bce': [], 'final_dice': [], 'coarse_bce': [], 'coarse_dice': []
    }
    delta_t_list = []
    delta_t_std_list = []
    num_pos_list = []
    num_neg_list = []
    candidate_count_list = []
    small_candidate_count_list = []

    # 只有主进程显示进度条
    if rank == 0:
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        use_pbar = True
    else:
        pbar = iter(train_loader)
        use_pbar = False
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        anomaly_masks = batch['anomaly_mask'].to(device)
        
        # 准备 targets
        targets = {
            'anomaly_mask': anomaly_masks,
            'small_anomaly_mask': anomaly_masks  # 简化：使用同一 mask
        }
        
        optimizer.zero_grad()

        # ===== Forward =====
        if use_hazard and use_prior:
            # 需要先提取候选区域
            with torch.no_grad():
                # 处理 DDP 模型
                if hasattr(model, 'module'):
                    backbone = model.module.backbone
                    decoder = model.module.decoder
                    forward_with_hazard_prior = model.module.forward_with_hazard_prior
                else:
                    backbone = model.backbone
                    decoder = model.decoder
                    forward_with_hazard_prior = model.forward_with_hazard_prior
                
                # 第一次 forward 获取 coarse map
                features = backbone(images)
                coarse_logits, _ = decoder(features, target_size=(images.shape[2], images.shape[3]), small_hazard_prior=None)
                coarse_prob = torch.sigmoid(coarse_logits)
                
                # 提取候选区域
                candidate_masks, area_ratios = extract_candidates_batch(coarse_prob, candidate_extractor, device)
            
            # 完整 forward (带 hazard scoring 和 prior)
            outputs = forward_with_hazard_prior(
                images,
                candidate_masks=candidate_masks,
                area_ratios=area_ratios,
                text_similarities=None
            )
        else:
            # 处理 DDP 模型
            if hasattr(model, 'module'):
                outputs_raw = model.module(images, return_details=False)
            else:
                outputs_raw = model(images, return_details=False)
            outputs = AnomalySegmentationOutput(
                coarse_logits=outputs_raw.coarse_logits,
                final_logits=outputs_raw.final_logits,
                candidates=[],
                hazard_scores=None,
                delta_t=None,
                small_hazard_prior=None,
                candidate_masks=None,
                area_ratios=None
            )

        # ===== Loss =====
        loss_dict = criterion(outputs, targets)
        total_loss = loss_dict['total_loss']

        # ===== Backward =====
        total_loss.backward()
        optimizer.step()

        # ===== Metrics =====
        total_metrics.update(outputs.final_logits, anomaly_masks, total_loss.item())
        coarse_metrics.update(outputs.coarse_logits, anomaly_masks, loss_dict['coarse_loss'].item())
        final_metrics.update(outputs.final_logits, anomaly_masks, loss_dict['final_loss'].item())
        
        # 收集损失统计
        loss_stats['total'].append(total_loss.item())
        loss_stats['final'].append(loss_dict['final_loss'].item())
        loss_stats['coarse'].append(loss_dict['coarse_loss'].item())
        loss_stats['hazard'].append(loss_dict['hazard_loss'].item())
        loss_stats['final_bce'].append(loss_dict['final_bce'].item())
        loss_stats['final_dice'].append(loss_dict['final_dice'].item())
        loss_stats['coarse_bce'].append(loss_dict['coarse_bce'].item())
        loss_stats['coarse_dice'].append(loss_dict['coarse_dice'].item())
        
        if outputs.delta_t is not None:
            delta_t_list.append(outputs.delta_t.mean().item())
            delta_t_std_list.append(outputs.delta_t.std().item())
        
        # 跟踪候选区域数量
        if outputs.candidate_masks is not None:
            # 计算每个样本的有效候选数
            cand_mask = outputs.candidate_masks  # [B, N, H, W]
            valid_cands = (cand_mask.sum(dim=(2, 3)) > 0).sum(dim=1)  # [B]
            candidate_count_list.extend(valid_cands.cpu().tolist())
            
            # 小候选区 (面积比例 < 0.01)
            if outputs.area_ratios is not None:
                small_mask = outputs.area_ratios < 0.01  # [B, N]
                small_cands = (small_mask & (valid_cands.unsqueeze(1) > 0)).sum(dim=1)
                small_candidate_count_list.extend(small_cands.cpu().tolist())
        
        num_pos_list.append(loss_dict['num_positives'])
        num_neg_list.append(loss_dict['num_negatives'])

        # ===== Visualization =====
        if debug and batch_idx % vis_interval == 0 and vis_dir is not None:
            visualize_outputs(outputs, targets, batch_idx, vis_dir, prefix='train', rank=rank, epoch=epoch)

        # 更新进度条
        if use_pbar:
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'coarse': f"{loss_dict['coarse_loss'].item():.4f}",
                'final': f"{loss_dict['final_loss'].item():.4f}",
                'hazard': f"{loss_dict['hazard_loss'].item():.4f}"
            })

    # ===== Epoch Stats =====
    total_stats = total_metrics.compute()
    coarse_stats = coarse_metrics.compute()
    final_stats = final_metrics.compute()
    
    avg_loss = {
        'total': np.mean(loss_stats['total']),
        'final': np.mean(loss_stats['final']),
        'coarse': np.mean(loss_stats['coarse']),
        'hazard': np.mean(loss_stats['hazard']),
        'final_bce': np.mean(loss_stats['final_bce']),
        'final_dice': np.mean(loss_stats['final_dice']),
        'coarse_bce': np.mean(loss_stats['coarse_bce']),
        'coarse_dice': np.mean(loss_stats['coarse_dice'])
    }
    
    avg_delta_t = np.mean(delta_t_list) if delta_t_list else 0.0
    avg_delta_t_std = np.mean(delta_t_std_list) if delta_t_std_list else 0.0
    avg_num_pos = np.mean(num_pos_list) if num_pos_list else 0.0
    avg_num_neg = np.mean(num_neg_list) if num_neg_list else 0.0
    avg_cand_count = np.mean(candidate_count_list) if candidate_count_list else 0.0
    avg_small_cand_count = np.mean(small_candidate_count_list) if small_candidate_count_list else 0.0

    # 只有主进程打印日志
    if logger is not None and rank == 0:
        logger.info(f"Epoch {epoch} [Train]")
        logger.info(f"  Loss - Total: {avg_loss['total']:.4f}, Final: {avg_loss['final']:.4f}, Coarse: {avg_loss['coarse']:.4f}, Hazard: {avg_loss['hazard']:.4f}")
        logger.info(f"  Final - BCE: {avg_loss['final_bce']:.4f}, Dice: {avg_loss['final_dice']:.4f}, IoU: {final_stats['iou']:.4f}, AUROC: {final_stats['auroc']:.4f}, AP: {final_stats['ap']:.4f}, FPR95: {final_stats['fpr95']:.4f}")
        logger.info(f"  Coarse - BCE: {avg_loss['coarse_bce']:.4f}, Dice: {avg_loss['coarse_dice']:.4f}, IoU: {coarse_stats['iou']:.4f}, AUROC: {coarse_stats['auroc']:.4f}")
        logger.info(f"  Delta_t mean: {avg_delta_t:.4f}, std: {avg_delta_t_std:.4f}, Avg candidates: {avg_cand_count:.1f}, Avg small candidates: {avg_small_cand_count:.1f}")
        logger.info(f"  Num pos: {avg_num_pos:.1f}, Num neg: {avg_num_neg:.1f}")

    return avg_loss, final_stats


def validate_epoch(model, val_loader, criterion, device, logger, epoch, config, candidate_extractor=None, vis_dir=None, max_batches=None, rank=0):
    """验证一个 epoch"""
    model.eval()

    loss_cfg = config.get('loss', {})
    shield_cfg = config.get('shield_lite', {})
    
    use_hazard = shield_cfg.get('use_hazard_scorer', True)
    use_prior = shield_cfg.get('use_small_hazard_prior', True)

    total_metrics = AnomalyMetricsTracker()
    coarse_metrics = AnomalyMetricsTracker()
    final_metrics = AnomalyMetricsTracker()
    
    loss_stats = {'total': [], 'final': [], 'coarse': [], 'hazard': []}
    delta_t_list = []

    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = batch['image'].to(device)
            anomaly_masks = batch['anomaly_mask'].to(device)
            
            targets = {
                'anomaly_mask': anomaly_masks,
                'small_anomaly_mask': anomaly_masks
            }

            # Forward
            if use_hazard and use_prior:
                # 处理 DDP 模型
                if hasattr(model, 'module'):
                    backbone = model.module.backbone
                    decoder = model.module.decoder
                    forward_with_hazard_prior = model.module.forward_with_hazard_prior
                else:
                    backbone = model.backbone
                    decoder = model.decoder
                    forward_with_hazard_prior = model.forward_with_hazard_prior
                
                features = backbone(images)
                coarse_logits, _ = decoder(features, target_size=(images.shape[2], images.shape[3]), small_hazard_prior=None)
                coarse_prob = torch.sigmoid(coarse_logits)
                
                candidate_masks, area_ratios = extract_candidates_batch(coarse_prob, candidate_extractor, device)
                
                outputs = forward_with_hazard_prior(
                    images,
                    candidate_masks=candidate_masks,
                    area_ratios=area_ratios,
                    text_similarities=None
                )
            else:
                # 处理 DDP 模型
                if hasattr(model, 'module'):
                    outputs_raw = model.module(images, return_details=False)
                else:
                    outputs_raw = model(images, return_details=False)
                outputs = AnomalySegmentationOutput(
                    coarse_logits=outputs_raw.coarse_logits,
                    final_logits=outputs_raw.final_logits,
                    candidates=[],
                    hazard_scores=None,
                    delta_t=None,
                    small_hazard_prior=None,
                    candidate_masks=None,
                    area_ratios=None
                )

            # Loss
            loss_dict = criterion(outputs, targets)
            
            # Metrics
            total_metrics.update(outputs.final_logits, anomaly_masks, loss_dict['total_loss'].item())
            coarse_metrics.update(outputs.coarse_logits, anomaly_masks, loss_dict['coarse_loss'].item())
            final_metrics.update(outputs.final_logits, anomaly_masks, loss_dict['final_loss'].item())
            
            loss_stats['total'].append(loss_dict['total_loss'].item())
            loss_stats['final'].append(loss_dict['final_loss'].item())
            loss_stats['coarse'].append(loss_dict['coarse_loss'].item())
            loss_stats['hazard'].append(loss_dict['hazard_loss'].item())
            
            if outputs.delta_t is not None:
                delta_t_list.append(outputs.delta_t.mean().item())
            
            # Visualization
            if vis_dir is not None and batch_idx == 0:
                visualize_outputs(outputs, targets, batch_idx, vis_dir, prefix='val', rank=rank, epoch=epoch)

            pbar.set_postfix({
                'loss': f"{loss_dict['total_loss'].item():.4f}"
            })

    # Epoch stats
    total_stats = total_metrics.compute()
    coarse_stats = coarse_metrics.compute()
    final_stats = final_metrics.compute()
    
    avg_loss = {
        'total': np.mean(loss_stats['total']),
        'final': np.mean(loss_stats['final']),
        'coarse': np.mean(loss_stats['coarse']),
        'hazard': np.mean(loss_stats['hazard'])
    }
    
    avg_delta_t = np.mean(delta_t_list) if delta_t_list else 0.0

    # 只有主进程打印日志
    if logger is not None and rank == 0:
        logger.info(f"Epoch {epoch} [Val]")
        logger.info(f"  Loss - Total: {avg_loss['total']:.4f}, Final: {avg_loss['final']:.4f}, Coarse: {avg_loss['coarse']:.4f}, Hazard: {avg_loss['hazard']:.4f}")
        logger.info(f"  Final - IoU: {final_stats['iou']:.4f}, AUROC: {final_stats['auroc']:.4f}, AP: {final_stats['ap']:.4f}, FPR95: {final_stats['fpr95']:.4f}")
        logger.info(f"  Coarse - IoU: {coarse_stats['iou']:.4f}, AUROC: {coarse_stats['auroc']:.4f}")
        logger.info(f"  Delta_t mean: {avg_delta_t:.4f}")

    # 返回完整指标
    val_metrics = {
        'loss': avg_loss['total'],
        'final_iou': final_stats['iou'],
        'final_auroc': final_stats['auroc'],
        'final_ap': final_stats['ap'],
        'final_fpr95': final_stats['fpr95'],
        'coarse_iou': coarse_stats['iou'],
        'coarse_auroc': coarse_stats['auroc'],
        'delta_t': avg_delta_t
    }
    return val_metrics, avg_loss


def save_checkpoint(model_state_dict, optimizer, epoch, metrics, loss_stats, save_path, is_best=False, save_latest=True):
    """保存 checkpoint
    
    Args:
        model_state_dict: 模型 state dict (DDP 包装前)
        optimizer: 优化器
        epoch: 当前 epoch
        metrics: 验证指标
        loss_stats: 损失统计
        save_path: 保存路径
        is_best: 是否是最佳模型
        save_latest: 是否保存 latest
    """
    save_dir = os.path.dirname(save_path)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'loss_stats': loss_stats
    }
    
    # 保存每个 epoch 的 checkpoint (带 epoch 编号)
    epoch_path = os.path.join(save_dir, f'epoch_{epoch:03d}.pth')
    torch.save(checkpoint, epoch_path)
    
    # 保存 latest
    if save_latest:
        latest_path = os.path.join(save_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)
    
    # 保存 best
    if is_best:
        best_path = os.path.join(save_dir, 'best.pth')
        torch.save(checkpoint, best_path)
        print(f"Best checkpoint saved to {best_path}")


def main():
    """主函数"""
    args = parse_args()
    
    config = load_config(args.config)
    
    if args.device:
        config['device'] = args.device
    
    # 分布式训练设置
    local_rank = args.local_rank
    use_distributed = local_rank >= 0
    
    if use_distributed:
        # 初始化分布式训练
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        
        # 从配置获取 GPU 设备列表
        device_ids = config['train'].get('devices', [4, 5, 6, 7])
        if rank < len(device_ids):
            torch.cuda.set_device(device_ids[rank])
        device = torch.device(f'cuda:{device_ids[rank]}' if rank < len(device_ids) else 'cuda:0')
    else:
        world_size = 1
        rank = 0
        device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # 路径配置
    checkpoint_cfg = config.get('checkpoint', {})
    log_dir = checkpoint_cfg.get('log_dir', 'logs/shield_lite')
    save_dir = checkpoint_cfg.get('save_dir', 'checkpoints/shield_lite')
    vis_dir = checkpoint_cfg.get('vis_dir', 'visualizations/shield_lite')
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    # 只有主进程创建 logger
    if rank == 0:
        logger = setup_logger('shield_lite_train', log_file=os.path.join(log_dir, 'train.log'))
        logger.info("=" * 60)
        logger.info("SHIELD-Lite Training")
        logger.info("=" * 60)
        logger.info(f"Config: {args.config}")
        logger.info(f"Distributed training: {use_distributed}, World size: {world_size}")
    else:
        logger = None
    
    # Device
    if logger:
        logger.info(f"Using device: {device}, Rank: {rank}")
    
    # Components
    if logger:
        logger.info("Building dataloaders...")
    train_loader = build_dataloader(config, split='train', rank=rank, world_size=world_size)
    val_loader = build_dataloader(config, split='val', rank=rank, world_size=world_size)
    
    if logger:
        logger.info(f"Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
        logger.info(f"Val: {len(val_loader.dataset)} samples, {len(val_loader)} batches")
    
    if logger:
        logger.info("Building model...")
    model, ablation_config = build_model(config)
    model = model.to(device)
    
    # 分布式包装
    if use_distributed:
        model = DDP(model, device_ids=[device.index] if device.index is not None else None)
    
    if logger:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Loss config logging
    shield_cfg = config.get('shield_lite', {})
    loss_cfg = config.get('loss', {})
    if logger:
        logger.info(f"SHIELD-Lite Config:")
        logger.info(f"  - use_hazard_scorer: {shield_cfg.get('use_hazard_scorer', True)}")
        logger.info(f"  - use_small_hazard_prior: {shield_cfg.get('use_small_hazard_prior', True)}")
        logger.info(f"  - prior_fusion_mode: {shield_cfg.get('prior_fusion_mode', 'add')}")
        logger.info(f"  - lambda_coarse: {loss_cfg.get('lambda_coarse', 0.5)}")
        logger.info(f"  - lambda_hazard: {loss_cfg.get('lambda_hazard', 0.5)}")
    
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    criterion = build_loss(config)
    candidate_extractor = build_candidate_extractor(config)
    
    # Resume
    start_epoch = 0
    best_metric = 0.0
    best_metrics_dict = {}
    if args.resume:
        if logger:
            logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        # DDP 模型加载需要处理 module 前缀
        state_dict = checkpoint['model_state_dict']
        if use_distributed:
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            model.module.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_metric = checkpoint.get('best_metric', 0.0)
        if logger:
            logger.info(f"Resumed from epoch {start_epoch}, best metric: {best_metric:.4f}")
    
    # Training loop
    num_epochs = config['train'].get('num_epochs', 30)
    save_interval = checkpoint_cfg.get('save_interval', 5)
    val_interval = config['train'].get('val_interval', 1)
    max_val_batches = config['train'].get('max_val_batches', None)
    
    if logger:
        logger.info("Starting training...")
    
    for epoch in range(start_epoch, num_epochs):
        # 分布式训练时设置 epoch
        if use_distributed:
            train_loader.sampler.set_epoch(epoch)
        
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, logger, epoch, config,
            candidate_extractor=candidate_extractor, vis_dir=vis_dir, debug=args.debug,
            rank=rank
        )
        
        # Validate
        if (epoch - start_epoch) % val_interval == 0 or epoch == num_epochs - 1:
            val_metrics, val_loss = validate_epoch(
                model, val_loader, criterion, device, logger, epoch, config,
                candidate_extractor=candidate_extractor, vis_dir=vis_dir, max_batches=max_val_batches,
                rank=rank
            )
        else:
            val_metrics = {'final_ap': 0.0, 'final_auroc': 0.0, 'final_fpr95': 1.0}
            val_loss = {'total': 0.0}
        
        # LR schedule
        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            if logger:
                logger.info(f"Learning rate: {current_lr:.6f}")
        
        # Save (只有主进程保存)
        if not use_distributed or rank == 0:
            if (epoch - start_epoch) % val_interval == 0 or epoch == num_epochs - 1:
                # 使用 AP 作为 best checkpoint 的主要指标
                is_best = val_metrics.get('final_ap', 0) > best_metric
                if is_best:
                    best_metric = val_metrics.get('final_ap', 0)
                    best_metrics_dict = val_metrics.copy()
                    if logger:
                        logger.info(f"New best AP: {best_metric:.4f}, AUROC: {val_metrics.get('final_auroc', 0):.4f}, FPR95: {val_metrics.get('final_fpr95', 0):.4f}")
                
                if is_best or epoch == num_epochs - 1:
                    save_path = os.path.join(save_dir, f'epoch_{epoch}.pth')
                    # 提取原始模型state_dict
                    if use_distributed:
                        model_state = model.module.state_dict()
                    else:
                        model_state = model.state_dict()
                    save_checkpoint(model_state, optimizer, epoch, val_metrics, val_loss, save_path, is_best=is_best)
    
    if logger:
        logger.info("=" * 60)
        logger.info("Training finished!")
        logger.info(f"Best AP: {best_metric:.4f}")
        logger.info(f"Best Metrics: {best_metrics_dict}")
        logger.info(f"Best checkpoint: {os.path.join(save_dir, 'best.pth')}")
        logger.info(f"Latest checkpoint: {os.path.join(save_dir, 'latest.pth')}")
    
    # 清理分布式训练
    if use_distributed:
        dist.destroy_process_group()
    
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    criterion = build_loss(config)
    candidate_extractor = build_candidate_extractor(config)
    
    # Resume
    start_epoch = 0
    best_metric = 0.0
    best_metrics_dict = {}
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_metric = checkpoint.get('best_metric', 0.0)
        logger.info(f"Resumed from epoch {start_epoch}, best metric: {best_metric:.4f}")
    
    # Training loop
    num_epochs = config['train'].get('num_epochs', 30)
    save_interval = checkpoint_cfg.get('save_interval', 5)
    val_interval = config['train'].get('val_interval', 1)
    max_val_batches = config['train'].get('max_val_batches', None)
    
    logger.info("Starting training...")
    for epoch in range(start_epoch, num_epochs):
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, logger, epoch, config,
            candidate_extractor=candidate_extractor, vis_dir=vis_dir, debug=args.debug
        )
        
        # Validate
        if (epoch - start_epoch) % val_interval == 0 or epoch == num_epochs - 1:
            val_metrics, val_loss = validate_epoch(
                model, val_loader, criterion, device, logger, epoch, config,
                candidate_extractor=candidate_extractor, vis_dir=vis_dir, max_batches=max_val_batches
            )
        else:
            val_metrics = {'auroc': 0.0}
            val_loss = {'total': 0.0}
        
        # LR schedule
        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Learning rate: {current_lr:.6f}")
        
        # Save
        if (epoch - start_epoch) % val_interval == 0 or epoch == num_epochs - 1:
            # 使用 AP 作为 best checkpoint 的主要指标
            is_best = val_metrics.get('final_ap', 0) > best_metric
            if is_best:
                best_metric = val_metrics.get('final_ap', 0)
                best_metrics_dict = val_metrics.copy()
                logger.info(f"New best AP: {best_metric:.4f}, AUROC: {val_metrics.get('final_auroc', 0):.4f}, FPR95: {val_metrics.get('final_fpr95', 0):.4f}")
            
            if (epoch + 1) % save_interval == 0 or is_best:
                save_path = os.path.join(save_dir, f'epoch_{epoch}.pth')
                save_checkpoint(model, optimizer, epoch, val_metrics, val_loss, save_path, is_best=is_best)
    
    logger.info("=" * 60)
    logger.info("Training finished!")
    logger.info(f"Best AP: {best_metric:.4f}")
    logger.info(f"Best Metrics: {best_metrics_dict}")
    logger.info(f"Best checkpoint: {os.path.join(save_dir, 'best.pth')}")
    logger.info(f"Latest checkpoint: {os.path.join(save_dir, 'latest.pth')}")


if __name__ == '__main__':
    main()
