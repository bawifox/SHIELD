#!/usr/bin/env python3
"""
SHIELD-Lite Sanity Check Script
验证完整 SHIELD-Lite 模块是否正常工作
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config
from utils.logger import setup_logger
from utils.candidate_extractor import CandidateExtractor
from models.anomaly_segmentation import AnomalySegmentationModelWithPrior, AnomalySegmentationOutput
from datasets import AnomalySegmentationDataset, build_train_transforms
from losses.hazard_losses import AnomalySegmentationTotalLoss


def parse_args():
    parser = argparse.ArgumentParser(description="SHIELD-Lite Sanity Check")
    parser.add_argument("--config", type=str, default="configs/shield_lite.yaml")
    parser.add_argument("--resume", type=str, default="checkpoints/baseline_training/best.pth")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_batches", type=int, default=10, help="Number of batches to test")
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


def build_dataloader(config, split='train', max_batches=None):
    """构建数据加载器"""
    anomaly_cfg = config.get('anomaly_dataset', {})
    data_root = anomaly_cfg.get('data_root', 'train_set')

    if split == 'train':
        transform = build_train_transforms(config) if config.get('transforms', {}).get('train') else None
    else:
        transform = build_train_transforms(config)

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

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )

    return loader


def build_model(config, device):
    """构建 SHIELD-Lite 模型"""
    model_cfg = config.get('model', {})
    anomaly_cfg = config.get('anomaly_model', {})
    shield_cfg = config.get('shield_lite', {})

    pretrained_path = anomaly_cfg.get('pretrained_backbone_path')
    if not pretrained_path and 'checkpoint' in config:
        save_dir = config['checkpoint'].get('save_dir', 'checkpoints/cityscapes')
        pretrained_path = f"{save_dir}/backbone_weights.pth"

    model = AnomalySegmentationModelWithPrior(
        in_channels=model_cfg.get('in_channels', 3),
        decoder_dim=anomaly_cfg.get('decoder_dim', 256),
        pretrained_backbone_path=pretrained_path,
        
        # Hazard Scorer
        use_hazard_scorer=True,
        hazard_beta=shield_cfg.get('hazard_beta', 10.0),
        base_threshold=shield_cfg.get('base_threshold', 0.01),
        
        # Small Hazard Prior
        use_small_hazard_prior=True,
        prior_fusion_mode=shield_cfg.get('prior_fusion_mode', 'add'),
        
        enable_training_details=True
    )
    
    return model.to(device)


def load_checkpoint(model, checkpoint_path, device):
    """加载 baseline checkpoint"""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 检查 checkpoint 结构
    print(f"Checkpoint keys: {checkpoint.keys()}")
    
    # 加载模型权重
    state_dict = checkpoint['model_state_dict']
    
    # 移除 prefix (如果有)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v
    
    # 尝试加载
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")
    
    return model, checkpoint.get('epoch', 0)


def build_candidate_extractor(config):
    """构建候选区域提取器"""
    shield_cfg = config.get('shield_lite', {})
    
    return CandidateExtractor(
        threshold_high=shield_cfg.get('candidate_threshold_high', 0.5),
        threshold_small=shield_cfg.get('candidate_threshold_small', 0.3),
        tau_small=shield_cfg.get('candidate_tau_small', 500),
        local_response_thresh=shield_cfg.get('candidate_local_response_thresh', 0.6),
        N_max=shield_cfg.get('max_candidates', 10),
        min_area=shield_cfg.get('min_candidate_area', 50)
    )


def extract_candidates_batch(coarse_prob, extractor, device):
    """从 coarse probability map 批量提取候选区域"""
    B = coarse_prob.shape[0]
    N_max = extractor.N_max
    H, W = coarse_prob.shape[2], coarse_prob.shape[3]
    
    candidate_masks = torch.zeros(B, N_max, H, W, device=device)
    area_ratios = torch.zeros(B, N_max, device=device)
    num_candidates = []
    num_small_candidates = []
    
    for b in range(B):
        prob_map = coarse_prob[b, 0].detach().cpu().numpy()
        candidates = extractor(prob_map)
        
        n_cands = min(len(candidates), N_max)
        num_candidates.append(n_cands)
        
        # 统计 small candidates
        n_small = sum(1 for c in candidates if c.area < extractor.tau_small)
        num_small_candidates.append(n_small)
        
        for i, cand in enumerate(candidates[:n_cands]):
            candidate_masks[b, i] = torch.from_numpy(cand.mask).float().to(device)
            area_ratios[b, i] = cand.area_ratio
    
    return candidate_masks, area_ratios, num_candidates, num_small_candidates


def visualize_detailed(outputs, targets, batch_idx, vis_dir, prefix='sanity'):
    """详细可视化"""
    B = outputs.coarse_logits.shape[0]
    H, W = outputs.coarse_logits.shape[2], outputs.coarse_logits.shape[3]
    
    for b in range(min(B, 2)):  # 只可视化前2张
        fig = plt.figure(figsize=(20, 12))
        
        # 1. 原图 (从 batch 获取，这里简化用 coarse map)
        coarse_prob = torch.sigmoid(outputs.coarse_logits[b, 0]).detach().cpu().numpy()
        
        # 2. GT anomaly mask
        gt_mask = targets['anomaly_mask'][b, 0].detach().cpu().numpy()
        
        # 3. Coarse map
        ax1 = fig.add_subplot(2, 4, 1)
        ax1.imshow(coarse_prob, cmap='jet')
        ax1.set_title('Coarse Map')
        ax1.axis('off')
        
        # 4. GT Mask
        ax2 = fig.add_subplot(2, 4, 2)
        ax2.imshow(gt_mask, cmap='gray')
        ax2.set_title('GT Anomaly Mask')
        ax2.axis('off')
        
        # 5. Candidate regions
        ax3 = fig.add_subplot(2, 4, 3)
        if outputs.candidate_masks is not None:
            cand_sum = outputs.candidate_masks[b].sum(0).detach().cpu().numpy()
            ax3.imshow(cand_sum, cmap='jet')
            ax3.set_title('All Candidates')
        else:
            ax3.text(0.5, 0.5, 'No Candidates', ha='center', va='center')
            ax3.set_title('All Candidates')
        ax3.axis('off')
        
        # 6. Small hazard prior
        ax4 = fig.add_subplot(2, 4, 4)
        if outputs.small_hazard_prior is not None:
            prior = outputs.small_hazard_prior[b, 0].detach().cpu().numpy()
            im = ax4.imshow(prior, cmap='jet')
            ax4.set_title('Small Hazard Prior')
            plt.colorbar(im, ax=ax4, fraction=0.046)
        else:
            ax4.text(0.5, 0.5, 'No Prior', ha='center', va='center')
            ax4.set_title('Small Hazard Prior')
        ax4.axis('off')
        
        # 7. Final prediction
        ax5 = fig.add_subplot(2, 4, 5)
        final_prob = torch.sigmoid(outputs.final_logits[b, 0]).detach().cpu().numpy()
        im = ax5.imshow(final_prob, cmap='jet')
        ax5.set_title('Final Prediction')
        ax5.axis('off')
        plt.colorbar(im, ax=ax5, fraction=0.046)
        
        # 8. Hazard scores bar plot
        ax6 = fig.add_subplot(2, 4, 6)
        if outputs.hazard_scores is not None:
            scores = outputs.hazard_scores[b].detach().cpu().numpy()
            valid_scores = scores[scores > 0]
            if len(valid_scores) > 0:
                ax6.bar(range(len(valid_scores)), valid_scores)
            else:
                ax6.text(0.5, 0.5, 'No Valid Scores', ha='center', va='center')
            
            delta_t_str = f"Δt={outputs.delta_t[b].item():.4f}" if outputs.delta_t is not None else "Δt=N/A"
            ax6.set_title(f'Hazard Scores ({delta_t_str})')
            ax6.set_xlabel('Candidate')
            ax6.set_ylabel('Score')
        else:
            ax6.text(0.5, 0.5, 'No Scores', ha='center', va='center')
            ax6.set_title('Hazard Scores')
        ax6.axis('on')
        
        # 9. Overlay: GT + Prior
        ax7 = fig.add_subplot(2, 4, 7)
        if outputs.small_hazard_prior is not None:
            prior_binary = (outputs.small_hazard_prior[b, 0].detach().cpu().numpy() > 0.1).astype(float)
            overlay = gt_mask * 0.5 + prior_binary * 0.5
            ax7.imshow(overlay, cmap='jet')
            ax7.set_title('GT (gray) + Prior (color)')
        else:
            ax7.imshow(gt_mask, cmap='gray')
            ax7.set_title('GT Anomaly Mask')
        ax7.axis('off')
        
        # 10. Error map
        ax8 = fig.add_subplot(2, 4, 8)
        error = np.abs(final_prob - gt_mask)
        ax8.imshow(error, cmap='hot')
        ax8.set_title('Error Map')
        ax8.axis('off')
        
        plt.tight_layout()
        
        save_path = os.path.join(vis_dir, f'{prefix}_batch{batch_idx}_img{b}.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"Saved visualization to {save_path}")


def run_sanity_check(model, dataloader, criterion, optimizer, device, logger, config, candidate_extractor, vis_dir, num_batches=10):
    """运行 sanity check"""
    model.train()
    
    # 统计变量
    stats = {
        # A. Candidate extraction
        'num_candidates_per_image': [],
        'num_small_candidates_per_image': [],
        'empty_candidate_count': 0,
        
        # B. Hazard scoring
        'hazard_scores_all': [],
        'hazard_scores_pos': [],  # small GT region overlap
        'hazard_scores_neg': [],  # no overlap
        
        # C. Delta_t
        'delta_t_all': [],
        
        # D. Small-hazard prior
        'prior_mean': [],
        'prior_max': [],
        'prior_empty_count': 0,
        
        # E. Losses
        'loss_total': [],
        'loss_coarse': [],
        'loss_final': [],
        'loss_hazard': [],
    }
    
    logger.info("=" * 60)
    logger.info("Starting Sanity Check...")
    logger.info("=" * 60)
    
    pbar = tqdm(dataloader, desc="Sanity Check")
    for batch_idx, batch in enumerate(pbar):
        if batch_idx >= num_batches:
            break
            
        images = batch['image'].to(device)
        anomaly_masks = batch['anomaly_mask'].to(device)
        
        targets = {
            'anomaly_mask': anomaly_masks,
            'small_anomaly_mask': anomaly_masks
        }
        
        B = images.shape[0]
        
        optimizer.zero_grad()
        
        # ===== 1. 第一次 forward: 获取 coarse map =====
        features = model.backbone(images)
        coarse_logits, _ = model.decoder(features, target_size=(images.shape[2], images.shape[3]), small_hazard_prior=None)
        coarse_prob = torch.sigmoid(coarse_logits)
        
        # ===== 2. 提取候选区域 =====
        candidate_masks, area_ratios, num_cands, num_small_cands = extract_candidates_batch(
            coarse_prob, candidate_extractor, device
        )
        
        # 记录候选区域统计
        stats['num_candidates_per_image'].extend(num_cands)
        stats['num_small_candidates_per_image'].extend(num_small_cands)
        stats['empty_candidate_count'] += sum(1 for n in num_cands if n == 0)
        
        # ===== 3. 完整 forward: hazard scoring + prior =====
        outputs = model.forward_with_hazard_prior(
            images,
            candidate_masks=candidate_masks,
            area_ratios=area_ratios,
            text_similarities=None
        )
        
        # 保存 candidate_masks 用于 loss 计算
        outputs.candidate_masks = candidate_masks
        
        # ===== 4. Loss =====
        loss_dict = criterion(outputs, targets)
        total_loss = loss_dict['total_loss']
        
        # ===== 5. Backward (但不使用梯度，避免影响训练) =====
        # total_loss.backward()
        # optimizer.step()
        
        # ===== 收集统计信息 =====
        # A. Hazard scores
        if outputs.hazard_scores is not None:
            scores = outputs.hazard_scores.cpu().detach().numpy()
            stats['hazard_scores_all'].extend(scores.flatten().tolist())
            
            # 分类: 与 GT overlap 的为 positive
            # 简化: 使用 area_ratios > 0.01 作为 small candidate 的代理
            small_mask = (area_ratios.cpu().numpy() > 0.0001) & (area_ratios.cpu().numpy() < 0.01)
            if small_mask.any():
                small_scores = scores[small_mask]
                stats['hazard_scores_pos'].extend(small_scores.tolist())
            else:
                neg_scores = scores[~small_mask]
                stats['hazard_scores_neg'].extend(neg_scores.tolist())
        
        # B. Delta_t
        if outputs.delta_t is not None:
            delta_t = outputs.delta_t.cpu().detach().numpy()
            stats['delta_t_all'].extend(delta_t.tolist())
        
        # C. Prior
        if outputs.small_hazard_prior is not None:
            prior = outputs.small_hazard_prior.cpu().detach().numpy()
            prior_mean = prior.mean()
            prior_max = prior.max()
            stats['prior_mean'].append(prior_mean)
            stats['prior_max'].append(prior_max)
            if prior_max < 1e-6:
                stats['prior_empty_count'] += 1
        
        # D. Losses
        stats['loss_total'].append(total_loss.item())
        stats['loss_coarse'].append(loss_dict['coarse_loss'].item())
        stats['loss_final'].append(loss_dict['final_loss'].item())
        stats['loss_hazard'].append(loss_dict['hazard_loss'].item())
        
        # ===== 6. 可视化 =====
        if batch_idx < 3:  # 只可视化前3个batch
            visualize_detailed(outputs, targets, batch_idx, vis_dir, prefix='sanity')
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f"{total_loss.item():.4f}",
            'cands': f"{np.mean(num_cands):.1f}"
        })
    
    return stats


def analyze_and_report(stats, logger):
    """分析并报告统计结果"""
    logger.info("\n" + "=" * 60)
    logger.info("SANITY CHECK RESULTS")
    logger.info("=" * 60)
    
    # A. Candidate extraction
    logger.info("\n[A] Candidate Extraction:")
    avg_cands = np.mean(stats['num_candidates_per_image']) if stats['num_candidates_per_image'] else 0
    avg_small_cands = np.mean(stats['num_small_candidates_per_image']) if stats['num_small_candidates_per_image'] else 0
    empty_ratio = stats['empty_candidate_count'] / max(len(stats['num_candidates_per_image']), 1)
    
    logger.info(f"  - Avg candidates per image: {avg_cands:.2f}")
    logger.info(f"  - Avg small candidates per image: {avg_small_cands:.2f}")
    logger.info(f"  - Empty candidate ratio: {empty_ratio*100:.1f}%")
    
    if avg_cands < 0.5:
        logger.warning("  ⚠️ WARNING: Very few candidates extracted!")
    
    # B. Hazard scoring
    logger.info("\n[B] Hazard Scoring:")
    if stats['hazard_scores_all']:
        all_scores = np.array(stats['hazard_scores_all'])
        logger.info(f"  - Mean: {all_scores.mean():.4f}")
        logger.info(f"  - Std: {all_scores.std():.4f}")
        logger.info(f"  - Min: {all_scores.min():.4f}")
        logger.info(f"  - Max: {all_scores.max():.4f}")
        
        if stats['hazard_scores_pos'] and stats['hazard_scores_neg']:
            pos_mean = np.mean(stats['hazard_scores_pos'])
            neg_mean = np.mean(stats['hazard_scores_neg'])
            logger.info(f"  - Small GT regions (pos) mean: {pos_mean:.4f}")
            logger.info(f"  - Non-overlap (neg) mean: {neg_mean:.4f}")
            if pos_mean <= neg_mean:
                logger.warning("  ⚠️ WARNING: Positive scores not higher than negative!")
        else:
            logger.info("  - Not enough positive/negative samples to compare")
    else:
        logger.warning("  ⚠️ WARNING: No hazard scores recorded!")
    
    # C. Delta_t
    logger.info("\n[C] Delta_t (Threshold Offset):")
    if stats['delta_t_all']:
        delta_t = np.array(stats['delta_t_all'])
        logger.info(f"  - Mean: {delta_t.mean():.6f}")
        logger.info(f"  - Std: {delta_t.std():.6f}")
        logger.info(f"  - Min: {delta_t.min():.6f}")
        logger.info(f"  - Max: {delta_t.max():.6f}")
        logger.info(f"  - Range: {delta_t.max() - delta_t.min():.6f}")
        
        if delta_t.std() < 1e-6:
            logger.warning("  ⚠️ WARNING: Delta_t is nearly constant!")
        if abs(delta_t.mean()) > 1.0:
            logger.warning("  ⚠️ WARNING: Delta_t might be diverging!")
    else:
        logger.warning("  ⚠️ WARNING: No Delta_t recorded!")
    
    # D. Small-hazard prior
    logger.info("\n[D] Small-Hazard Prior:")
    if stats['prior_mean']:
        logger.info(f"  - Mean: {np.mean(stats['prior_mean']):.6f}")
        logger.info(f"  - Max: {np.mean(stats['prior_max']):.6f}")
        logger.info(f"  - Empty prior count: {stats['prior_empty_count']}")
        
        if stats['prior_empty_count'] > 0:
            logger.warning(f"  ⚠️ WARNING: {stats['prior_empty_count']} samples have empty prior!")
    else:
        logger.warning("  ⚠️ WARNING: No prior statistics recorded!")
    
    # E. Losses
    logger.info("\n[E] Losses:")
    logger.info(f"  - Total loss: {np.mean(stats['loss_total']):.4f}")
    logger.info(f"  - Coarse loss: {np.mean(stats['loss_coarse']):.4f}")
    logger.info(f"  - Final loss: {np.mean(stats['loss_final']):.4f}")
    logger.info(f"  - Hazard loss: {np.mean(stats['loss_hazard']):.4f}")
    
    # 总结
    logger.info("\n" + "=" * 60)
    
    issues = []
    if avg_cands < 0.5:
        issues.append("Too few candidates extracted")
    if stats['empty_candidate_count'] > len(stats['num_candidates_per_image']) * 0.5:
        issues.append("Too many empty candidates")
    if stats['hazard_scores_all'] and np.std(stats['hazard_scores_all']) < 0.01:
        issues.append("Hazard scores have no variance")
    if stats['delta_t_all'] and np.std(stats['delta_t_all']) < 1e-6:
        issues.append("Delta_t is constant")
    if stats['prior_empty_count'] > 0:
        issues.append("Prior is empty for some samples")
    
    if issues:
        logger.info("ISSUES FOUND:")
        for issue in issues:
            logger.info(f"  - {issue}")
        logger.info("\n⚠️  Please fix these issues before正式训练!")
        return False
    else:
        logger.info("✓ All checks passed!")
        logger.info("\nSanity check completed successfully. Ready for formal training.")
        return True


def main():
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config)
    config['device'] = args.device
    
    # 创建目录
    log_dir = "logs/shield_lite_sanity"
    vis_dir = "visualizations/shield_lite_sanity"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    logger = setup_logger('shield_lite_sanity', log_file=os.path.join(log_dir, 'sanity.log'))
    
    logger.info(f"Config: {args.config}")
    logger.info(f"Checkpoint: {args.resume}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Num batches: {args.num_batches}")
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 构建数据加载器
    logger.info("Building dataloader...")
    dataloader = build_dataloader(config, split='train', max_batches=args.num_batches)
    logger.info(f"Dataset size: {len(dataloader.dataset)}")
    logger.info(f"Batches: {len(dataloader)}")
    
    # 构建模型
    logger.info("Building model...")
    model = build_model(config, device)
    
    # 加载 baseline checkpoint
    if args.resume:
        model, start_epoch = load_checkpoint(model, args.resume, device)
        logger.info(f"Loaded from epoch {start_epoch}")
    
    # 构建组件
    candidate_extractor = build_candidate_extractor(config)
    
    criterion = AnomalySegmentationTotalLoss(
        bce_weight=config.get('loss', {}).get('bce_weight', 0.5),
        dice_weight=config.get('loss', {}).get('dice_weight', 0.5),
        lambda_coarse=config.get('loss', {}).get('lambda_coarse', 0.5),
        lambda_hazard=config.get('loss', {}).get('lambda_hazard', 0.5),
        hazard_overlap_threshold=config.get('shield_lite', {}).get('hazard_overlap_threshold', 0.3),
        hazard_loss_type=config.get('loss', {}).get('hazard_loss_type', 'bce'),
        hazard_pos_weight=config.get('loss', {}).get('hazard_pos_weight', 1.0),
        hazard_neg_weight=config.get('loss', {}).get('hazard_neg_weight', 0.5)
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)
    
    # 运行 sanity check
    stats = run_sanity_check(
        model, dataloader, criterion, optimizer, device, logger,
        config, candidate_extractor, vis_dir, num_batches=args.num_batches
    )
    
    # 分析并报告
    success = analyze_and_report(stats, logger)
    
    # 输出目录信息
    logger.info("\n" + "=" * 60)
    logger.info("OUTPUT DIRECTORIES:")
    logger.info(f"  - Log directory: {log_dir}")
    logger.info(f"  - Visualization directory: {vis_dir}")
    logger.info("=" * 60)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
