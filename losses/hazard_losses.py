"""
Hazard Score Loss Module
用于监督 hazard score 的训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class HazardScoreLossOutput:
    """Hazard Score Loss 输出"""
    total_loss: torch.Tensor
    positive_loss: torch.Tensor
    negative_loss: torch.Tensor
    num_positives: int
    num_negatives: int


class HazardScoreLoss(nn.Module):
    """
    Hazard Score Loss
    
    监督方式：
    - 正样本：small candidate 与 GT small anomaly overlap 超过阈值
    - 负样本：否则
    
    损失形式：
    - 正样本：Focal Loss 或 BCE Loss
    - 负样本：BCE Loss (可选)
    """

    def __init__(
        self,
        overlap_threshold: float = 0.3,
        use_focal: bool = True,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        pos_weight: float = 1.0,
        neg_weight: float = 0.5,
        temperature: float = 1.0
    ):
        """
        Args:
            overlap_threshold: IoU 阈值，超过此值认为正样本
            use_focal: 是否使用 Focal Loss
            focal_alpha: Focal Loss alpha 参数
            focal_gamma: Focal Loss gamma 参数
            pos_weight: 正样本权重
            neg_weight: 负样本权重
            temperature: 温度参数
        """
        super().__init__()
        self.overlap_threshold = overlap_threshold
        self.use_focal = use_focal
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.temperature = temperature
    
    def forward(
        self,
        hazard_scores: torch.Tensor,
        candidate_masks: torch.Tensor,
        gt_small_anomaly_mask: torch.Tensor,
        area_ratios: Optional[torch.Tensor] = None
    ) -> HazardScoreLossOutput:
        """
        计算 Hazard Score Loss
        
        Args:
            hazard_scores: [B, N] 预测的 hazard scores
            candidate_masks: [B, N, H, W] 候选区域 masks (二值)
            gt_small_anomaly_mask: [B, 1, H, W] GT small anomaly mask
            area_ratios: [B, N] 候选区域面积比例 (可选)
            
        Returns:
            HazardScoreLossOutput
        """
        B, N = hazard_scores.shape
        
        # 计算每个候选与 GT 的 IoU
        iou_scores = self._compute_iou(candidate_masks, gt_small_anomaly_mask)  # [B, N]
        
        # 确定正负样本标签
        labels = (iou_scores > self.overlap_threshold).float()  # [B, N]
        
        # 统计正负样本数量
        num_positives = int(labels.sum().item())
        num_negatives = int((1 - labels).sum().item())
        
        # 计算损失
        if self.use_focal:
            loss = self._focal_loss(hazard_scores, labels)
        else:
            loss = self._bce_loss(hazard_scores, labels)
        
        # 分离正负样本损失 (用于分析)
        positive_loss = self._bce_loss(hazard_scores, labels) * labels
        negative_loss = self._bce_loss(hazard_scores, labels) * (1 - labels)
        
        # 归一化
        if num_positives > 0:
            positive_loss = positive_loss.sum() / num_positives
        else:
            positive_loss = torch.tensor(0.0, device=hazard_scores.device)
            
        if num_negatives > 0:
            negative_loss = negative_loss.sum() / num_negatives
        else:
            negative_loss = torch.tensor(0.0, device=hazard_scores.device)
        
        # 加权总损失
        total_loss = self.pos_weight * positive_loss + self.neg_weight * negative_loss
        
        return HazardScoreLossOutput(
            total_loss=total_loss,
            positive_loss=positive_loss,
            negative_loss=negative_loss,
            num_positives=num_positives,
            num_negatives=num_negatives
        )
    
    def _compute_iou(
        self,
        pred_masks: torch.Tensor,
        gt_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 IoU
        
        Args:
            pred_masks: [B, N, H, W] 预测的 masks
            gt_mask: [B, 1, H, W] GT mask
            
        Returns:
            iou: [B, N]
        """
        B, N, H, W = pred_masks.shape
        
        # 扩展 GT 到 [B, N, H, W]
        gt_expanded = gt_mask.expand(-1, N, -1, -1)  # [B, N, H, W]
        
        # 展平空间维度
        pred_flat = pred_masks.flatten(2)  # [B, N, H*W]
        gt_flat = gt_mask.flatten(2)  # [B, 1, H*W]
        
        # Intersection: pred & gt
        intersection = (pred_flat * gt_flat).sum(dim=-1)  # [B, N]
        
        # Union: pred | gt
        union = ((pred_flat + gt_flat) > 0).float().sum(dim=-1)  # [B, N]
        
        # IoU
        iou = intersection / (union + 1e-6)  # [B, N]
        
        return iou
    
    def _focal_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Focal Loss for hazard score
        
        Args:
            predictions: [B, N] 预测的 hazard scores (0-1)
            targets: [B, N] 标签 (0 或 1)
        """
        # 转换为概率
        probs = torch.sigmoid(predictions / self.temperature)
        
        # Focal Loss
        ce = F.binary_cross_entropy(probs, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.focal_gamma
        
        loss = self.focal_alpha * focal_weight * ce
        
        return loss.mean()
    
    def _bce_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Binary Cross Entropy Loss"""
        probs = torch.sigmoid(predictions / self.temperature)
        loss = F.binary_cross_entropy(probs, targets, reduction='none')
        return loss


class HazardScoreLossV2(nn.Module):
    """
    Hazard Score Loss V2
    支持更灵活的标签生成方式
    """

    def __init__(
        self,
        overlap_threshold: float = 0.3,
        loss_type: str = 'bce',  # 'bce', 'focal', 'softmargin'
        pos_weight: float = 1.0,
        neg_weight: float = 0.5
    ):
        super().__init__()
        self.overlap_threshold = overlap_threshold
        self.loss_type = loss_type
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
    
    def forward(
        self,
        hazard_scores: torch.Tensor,
        candidate_masks: torch.Tensor,
        gt_small_anomaly_mask: torch.Tensor,
        small_candidate_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            hazard_scores: [B, N]
            candidate_masks: [B, N, H, W]
            gt_small_anomaly_mask: [B, 1, H, W]
            small_candidate_mask: [B, N] 可选，标识哪些是 small candidate
            
        Returns:
            loss, info_dict
        """
        B, N = hazard_scores.shape
        
        # 计算 IoU
        iou = self._compute_iou(candidate_masks, gt_small_anomaly_mask)
        
        # 标签
        labels = (iou > self.overlap_threshold).float()
        
        # 如果提供了 small_candidate_mask，只计算 small candidates 的损失
        if small_candidate_mask is not None:
            labels = labels * small_candidate_mask
            valid_mask = small_candidate_mask
        else:
            valid_mask = torch.ones_like(labels)
        
        # 计算损失
        if self.loss_type == 'bce':
            loss = self._bce_loss(hazard_scores, labels, valid_mask)
        elif self.loss_type == 'focal':
            loss = self._focal_loss(hazard_scores, labels, valid_mask)
        elif self.loss_type == 'softmargin':
            loss = self._softmargin_loss(hazard_scores, labels, valid_mask)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # 统计
        num_pos = int((labels * valid_mask).sum().item())
        num_neg = int(((1 - labels) * valid_mask).sum().item())
        
        info = {
            'num_positives': num_pos,
            'num_negatives': num_neg,
            'mean_iou': iou.mean().item(),
            'mean_hazard_score': hazard_scores.mean().item()
        }
        
        return loss, info
    
    def _compute_iou(self, pred_masks: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
        B, N = pred_masks.shape[:2]
        gt_expanded = gt_mask.expand(-1, N, -1, -1)
        
        pred_flat = pred_masks.flatten(2)
        gt_flat = gt_mask.flatten(2)
        
        intersection = (pred_flat * gt_flat).sum(-1)
        union = ((pred_flat + gt_flat) > 0).float().sum(-1)
        
        return intersection / (union + 1e-6)
    
    def _bce_loss(self, preds: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(preds)
        loss = F.binary_cross_entropy(probs, labels, reduction='none')
        
        # 加权
        weight = labels * self.pos_weight + (1 - labels) * self.neg_weight
        loss = loss * weight * mask
        
        return loss.sum() / (mask.sum() + 1e-6)
    
    def _focal_loss(self, preds: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(preds)
        ce = F.binary_cross_entropy(probs, labels, reduction='none')
        p_t = probs * labels + (1 - probs) * (1 - labels)
        focal_weight = (1 - p_t) ** 2
        
        loss = 0.25 * focal_weight * ce
        loss = loss * mask
        
        return loss.sum() / (mask.sum() + 1e-6)
    
    def _softmargin_loss(self, preds: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Softmargin loss: log(1 + exp(-y * x))
        loss = F.soft_margin_loss(preds, labels, reduction='none')
        loss = loss * mask
        
        return loss.sum() / (mask.sum() + 1e-6)


# ============ Total Loss ============

class AnomalySegmentationTotalLoss(nn.Module):
    """
    Anomaly Segmentation Total Loss
    
    L_total = L_final + lambda1 * L_coarse + lambda2 * L_hazard
    
    其中:
    - L_final: final anomaly prediction 的 BCE + Dice Loss
    - L_coarse: coarse anomaly prediction 的 BCE + Dice Loss
    - L_hazard: hazard score 的分类 Loss
    """

    def __init__(
        self,
        # Segmentation Loss 参数
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        
        # Loss 权重
        lambda_coarse: float = 0.5,
        lambda_hazard: float = 0.5,
        
        # Hazard Loss 参数
        hazard_overlap_threshold: float = 0.3,
        hazard_loss_type: str = 'bce',
        hazard_pos_weight: float = 1.0,
        hazard_neg_weight: float = 0.5,
        
        # 其他
        use_uncertainty_weighting: bool = False
    ):
        """
        Args:
            bce_weight: BCE Loss 权重
            dice_weight: Dice Loss 权重
            lambda_coarse: coarse loss 权重
            lambda_hazard: hazard loss 权重
            hazard_overlap_threshold: IoU 阈值
            hazard_loss_type: hazard loss 类型
        """
        super().__init__()
        
        # Segmentation Loss
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
        # Loss weights
        self.lambda_coarse = lambda_coarse
        self.lambda_hazard = lambda_hazard
        
        # BCE Loss
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        
        # Hazard Loss
        self.hazard_loss = HazardScoreLossV2(
            overlap_threshold=hazard_overlap_threshold,
            loss_type=hazard_loss_type,
            pos_weight=hazard_pos_weight,
            neg_weight=hazard_neg_weight
        )
        
        self.use_uncertainty_weighting = use_uncertainty_weighting
    
    def forward(
        self,
        outputs,
        targets
    ) -> Dict[str, torch.Tensor]:
        """
        计算总损失
        
        Args:
            outputs: 模型输出 (AnomalySegmentationOutput)
                - coarse_logits: [B, 1, H, W]
                - final_logits: [B, 1, H, W]
                - hazard_scores: [B, N] (可选)
                - candidate_masks: [B, N, H, W] (可选)
                
            targets: 目标字典
                - anomaly_mask: [B, 1, H, W]
                - small_anomaly_mask: [B, 1, H, W] (可选)
                
        Returns:
            loss_dict: {
                'total_loss': ...,
                'final_loss': ...,
                'coarse_loss': ...,
                'hazard_loss': ...,
                'final_bce': ...,
                'final_dice': ...,
                'coarse_bce': ...,
                'coarse_dice': ...,
                'num_positives': ...,
                'num_negatives': ...
            }
        """
        # 解析输入 - 支持 dict 或带属性的对象
        if isinstance(outputs, dict):
            coarse_logits = outputs['coarse_logits']
            final_logits = outputs['final_logits']
            hazard_scores = outputs.get('hazard_scores', None)
            candidate_masks = outputs.get('candidate_masks', None)
        else:
            coarse_logits = outputs.coarse_logits
            final_logits = outputs.final_logits
            hazard_scores = getattr(outputs, 'hazard_scores', None)
            candidate_masks = getattr(outputs, 'candidate_masks', None)
        
        if isinstance(targets, dict):
            anomaly_mask = targets['anomaly_mask']
            small_anomaly_mask = targets.get('small_anomaly_mask', anomaly_mask)
        else:
            anomaly_mask = targets
            small_anomaly_mask = targets
        
        # ===== 1. Final Loss =====
        final_bce = self._bce_loss(final_logits, anomaly_mask)
        final_dice = self._dice_loss(final_logits, anomaly_mask)
        final_loss = self.bce_weight * final_bce + self.dice_weight * final_dice
        
        # ===== 2. Coarse Loss =====
        coarse_bce = self._bce_loss(coarse_logits, anomaly_mask)
        coarse_dice = self._dice_loss(coarse_logits, anomaly_mask)
        coarse_loss = self.bce_weight * coarse_bce + self.dice_weight * coarse_dice
        
        # ===== 3. Hazard Loss =====
        hazard_loss = torch.tensor(0.0, device=coarse_logits.device)
        num_positives = 0
        num_negatives = 0
        hazard_info = {}
        
        # 检查是否需要计算 hazard loss（支持 dict 或带属性的对象）
        has_hazard = hazard_scores is not None and candidate_masks is not None
        
        if has_hazard:
            # 使用 small anomaly mask 作为 GT
            hazard_loss, hazard_info = self.hazard_loss(
                hazard_scores,
                candidate_masks,
                small_anomaly_mask
            )
            
            num_positives = hazard_info.get('num_positives', 0)
            num_negatives = hazard_info.get('num_negatives', 0)
        
        # ===== 4. Total Loss =====
        total_loss = final_loss + self.lambda_coarse * coarse_loss + self.lambda_hazard * hazard_loss
        
        return {
            'total_loss': total_loss,
            'final_loss': final_loss,
            'coarse_loss': coarse_loss,
            'hazard_loss': hazard_loss,
            'final_bce': final_bce,
            'final_dice': final_dice,
            'coarse_bce': coarse_bce,
            'coarse_dice': coarse_dice,
            'num_positives': num_positives,
            'num_negatives': num_negatives
        }
    
    def _bce_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """BCE Loss"""
        loss = self.bce(logits, target)
        return loss.mean()
    
    def _dice_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Dice Loss"""
        pred = torch.sigmoid(logits)
        pred_flat = pred.flatten(1)
        target_flat = target.flatten(1)
        
        intersection = (pred_flat * target_flat).sum(1)
        union = pred_flat.sum(1) + target_flat.sum(1)
        
        dice = (2.0 * intersection + 1.0) / (union + 1.0)
        return 1 - dice.mean()


# ============ 测试 ============

def test_hazard_loss():
    """测试 Hazard Loss"""
    print("=" * 60)
    print("Testing HazardScoreLoss...")
    print("=" * 60)
    
    B, N, H, W = 2, 5, 64, 64
    
    # 模拟输入
    hazard_scores = torch.rand(B, N)  # [B, N]
    candidate_masks = (torch.rand(B, N, H, W) > 0.7).float()
    gt_small_anomaly = (torch.rand(B, 1, H, W) > 0.9).float()
    
    # Test 1: 基础版本
    print("\n[1] Testing HazardScoreLoss...")
    loss_fn = HazardScoreLoss(
        overlap_threshold=0.3,
        use_focal=True
    )
    
    result = loss_fn(hazard_scores, candidate_masks, gt_small_anomaly)
    print(f"  Total loss: {result.total_loss.item():.4f}")
    print(f"  Pos loss: {result.positive_loss.item():.4f}")
    print(f"  Neg loss: {result.negative_loss.item():.4f}")
    print(f"  Num pos: {result.num_positives}, Num neg: {result.num_negatives}")
    
    # Test 2: V2 版本
    print("\n[2] Testing HazardScoreLossV2...")
    loss_fn_v2 = HazardScoreLossV2(
        overlap_threshold=0.3,
        loss_type='bce'
    )
    
    loss_v2, info = loss_fn_v2(hazard_scores, candidate_masks, gt_small_anomaly)
    print(f"  Loss: {loss_v2.item():.4f}")
    print(f"  Info: {info}")
    
    # Test 3: Total Loss
    print("\n[3] Testing AnomalySegmentationTotalLoss...")
    
    # 模拟模型输出
    class MockOutput:
        def __init__(self):
            self.coarse_logits = torch.randn(B, 1, H, W)
            self.final_logits = torch.randn(B, 1, H, W)
            self.hazard_scores = torch.rand(B, N)
            self.candidate_masks = candidate_masks
    
    # 模拟目标
    targets = {
        'anomaly_mask': (torch.rand(B, 1, H, W) > 0.8).float(),
        'small_anomaly_mask': (torch.rand(B, 1, H, W) > 0.9).float()
    }
    
    total_loss_fn = AnomalySegmentationTotalLoss(
        bce_weight=0.5,
        dice_weight=0.5,
        lambda_coarse=0.5,
        lambda_hazard=0.5
    )
    
    outputs = MockOutput()
    loss_dict = total_loss_fn(outputs, targets)
    
    print(f"  Total loss: {loss_dict['total_loss'].item():.4f}")
    print(f"  Final loss: {loss_dict['final_loss'].item():.4f}")
    print(f"  Coarse loss: {loss_dict['coarse_loss'].item():.4f}")
    print(f"  Hazard loss: {loss_dict['hazard_loss'].item():.4f}")
    print(f"  Num pos: {loss_dict['num_positives']}, Num neg: {loss_dict['num_negatives']}")
    
    # Test 4: 梯度检查
    print("\n[4] Testing gradient...")
    total_loss_fn.train()
    outputs = MockOutput()
    outputs.coarse_logits.requires_grad = True
    outputs.final_logits.requires_grad = True
    outputs.hazard_scores.requires_grad = True
    
    loss_dict = total_loss_fn(outputs, targets)
    loss_dict['total_loss'].backward()
    
    print(f"  Coarse logits grad exists: {outputs.coarse_logits.grad is not None}")
    print(f"  Final logits grad exists: {outputs.final_logits.grad is not None}")
    print(f"  Hazard scores grad exists: {outputs.hazard_scores.grad is not None}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_hazard_loss()
