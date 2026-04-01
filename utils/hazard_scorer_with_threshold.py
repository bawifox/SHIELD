"""
Hazard Scoring with Adaptive Threshold Integration
整合 hazard scorer 和自适应阈值偏移
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

from utils.hazard_scorer import HazardScorer, HazardScorerSimple
from utils.adaptive_threshold import AdaptiveThresholdOffset, AdaptiveThresholdOffsetV2, AdaptiveThresholdOffsetSoftGating


@dataclass
class HazardScoringResult:
    """Hazard scoring 结果"""
    hazard_scores: torch.Tensor      # [N,] 原始 hazard scores
    soft_gates: torch.Tensor        # [N,] soft gates (0~1)
    thresholds: torch.Tensor       # [1,] 使用的阈值 t_adapt
    Delta_t: torch.Tensor           # [1,] 阈值偏移量
    small_indices: torch.Tensor     # [M,] 小目标的原始索引
    is_small: torch.Tensor          # [N,] 是否为小目标 (基于阈值)


class HazardScorerWithAdaptiveThreshold(nn.Module):
    """
    Hazard Scorer with Adaptive Threshold
    
    整合 Hazard Scorer 和 Adaptive Threshold Offset
    
    训练模式:
        - 使用 soft gate: w_i = sigmoid(beta * (h_i - t_adapt))
        
    推理模式:
        - 可切换到硬阈值: w_i = 1 if h_i > t_adapt else 0
    
    Ablation 支持:
        - 固定阈值模式
        - 自适应阈值模式
    """
    
    def __init__(
        self,
        # Hazard Scorer 参数
        visual_channels: int = 256,
        text_channels: int = 512,
        hidden_dim: int = 256,
        num_prompts: int = 2,
        output_size: int = 7,
        use_spatial: bool = True,
        
        # Adaptive Threshold 参数
        global_feature_channels: int = 256,
        threshold_hidden_dim: int = 64,
        alpha: float = 0.02,
        
        # 门控参数
        beta: float = 10.0,
        
        # 阈值参数
        base_threshold: float = 0.01,
        
        # 模式
        use_adaptive_threshold: bool = True,
        use_soft_gate: bool = True
    ):
        """
        Args:
            visual_channels: 视觉特征通道数
            text_channels: 文本特征通道数
            hidden_dim: 隐藏层维度
            num_prompts: prompts 数量
            output_size: 视觉特征空间尺寸
            use_spatial: 是否使用空间视觉特征
            
            global_feature_channels: 全局特征通道数
            threshold_hidden_dim: 阈值预测隐藏层维度
            alpha: 阈值偏移范围
            
            beta: soft gate 温度参数
            base_threshold: 基础阈值 t2
            
            use_adaptive_threshold: 是否使用自适应阈值
            use_soft_gate: 是否使用 soft gate
        """
        super().__init__()
        
        # 子模块
        self.hazard_scorer = HazardScorer(
            visual_channels=visual_channels,
            text_channels=text_channels,
            hidden_dim=hidden_dim,
            num_prompts=num_prompts,
            output_size=output_size,
            use_spatial=use_spatial
        )
        
        self.threshold_offset = AdaptiveThresholdOffset(
            feature_channels=global_feature_channels,
            hidden_dim=threshold_hidden_dim,
            alpha=alpha
        )
        
        # 参数
        self.beta = beta
        self.base_threshold = base_threshold
        self.use_adaptive_threshold = use_adaptive_threshold
        self.use_soft_gate = use_soft_gate
        
        # 注册 buffer 用于推理模式切换
        self.register_buffer('_use_hard_threshold', torch.tensor(False))
    
    def set_training_mode(self, use_soft_gate: bool = True):
        """设置训练模式"""
        self.use_soft_gate = use_soft_gate
        self._use_hard_threshold = torch.tensor(not use_soft_gate)
    
    def set_inference_mode(self, use_hard_threshold: bool = True):
        """设置推理模式"""
        self._use_hard_threshold = torch.tensor(use_hard_threshold)
        self.use_soft_gate = not use_hard_threshold
    
    def forward(
        self,
        visual_features: torch.Tensor,
        area_ratios: torch.Tensor,
        text_similarities: Optional[torch.Tensor] = None,
        global_features: Optional[torch.Tensor] = None,
        return_details: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算 hazard score 和 soft gate
        
        Args:
            visual_features: [N, C, H, W] 视觉特征
            area_ratios: [N,] 面积比例
            text_similarities: [N, num_prompts] 文本相似度
            global_features: [B, C] 或 [B, C, 1, 1] 全局特征 (用于阈值偏移)
            return_details: 是否返回详细信息
            
        Returns:
            如果 return_details=False:
                hazard_scores: [N,] 原始分数
                soft_gates: [N,] 门控值 (0~1)
            
            如果 return_details=True:
                HazardScoringResult 对象
        """
        # Step 1: 计算阈值
        t_adapt, Delta_t = self._compute_threshold(global_features, num_candidates=visual_features.shape[0])
        
        # Step 2: 计算 hazard scores
        hazard_scores = self.hazard_scorer(visual_features, area_ratios, text_similarities)
        
        # Step 3: 计算 soft gate - 确保维度匹配
        # t_adapt 可能是 [1,] 或标量，需要广播到 [N,]
        if isinstance(t_adapt, torch.Tensor) and t_adapt.numel() == 1:
            t_adapt = t_adapt.item()
        
        if self.use_soft_gate or self.training:
            # Soft gate: w = sigmoid(beta * (h - t_adapt))
            soft_gates = torch.sigmoid(self.beta * (hazard_scores - t_adapt))
        else:
            # Hard gate: w = 1 if h > t_adapt else 0
            soft_gates = (hazard_scores > t_adapt).float()
        
        if return_details:
            # 判断是否为小目标
            is_small = area_ratios < t_adapt
            
            # 确保返回 tensor
            threshold_tensor = torch.tensor(t_adapt) if isinstance(t_adapt, float) else t_adapt
            delta_tensor = torch.tensor(Delta_t) if isinstance(Delta_t, float) else Delta_t
            
            return HazardScoringResult(
                hazard_scores=hazard_scores,
                soft_gates=soft_gates,
                thresholds=threshold_tensor,
                Delta_t=delta_tensor,
                small_indices=torch.where(is_small)[0],
                is_small=is_small
            )
        
        return hazard_scores, soft_gates
    
    def _compute_threshold(
        self,
        global_features: Optional[torch.Tensor],
        num_candidates: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算自适应阈值
        
        Args:
            global_features: [B, C] 或 [B, C, 1, 1] 全局特征
            num_candidates: 候选数量 N，用于扩展阈值
            
        Returns:
            t_adapt: 阈值 [N,] 或 [1,]
            Delta_t: 偏移量 [N,] 或 [1,]
        """
        if self.use_adaptive_threshold and global_features is not None:
            Delta_t = self.threshold_offset(global_features)  # [B,] or [1,]
            
            # 处理 batch 情况：全局特征是 per-image，候选是 per-image
            # 需要将阈值扩展到每个候选
            if Delta_t.dim() == 0 or Delta_t.shape[0] == 1:
                # 标量或单值，广播到所有候选
                t_adapt = self.base_threshold + Delta_t
            else:
                # 多值情况，需要根据 num_candidates 扩展
                if num_candidates is not None and Delta_t.shape[0] < num_candidates:
                    # 假设每个图像的阈值相同，扩展到所有候选
                    t_adapt = self.base_threshold + Delta_t[0] if Delta_t.shape[0] > 1 else self.base_threshold + Delta_t
                else:
                    t_adapt = self.base_threshold + Delta_t
        else:
            # 固定阈值
            device = 'cpu'
            if global_features is not None:
                device = global_features.device
            
            t_adapt = torch.tensor(self.base_threshold, device=device)
            Delta_t = torch.tensor(0.0, device=device)
        
        return t_adapt, Delta_t
    
    def forward_only_small(
        self,
        visual_features: torch.Tensor,
        area_ratios: torch.Tensor,
        text_similarities: Optional[torch.Tensor] = None,
        global_features: Optional[torch.Tensor] = None,
        area_threshold: float = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        只对 small candidates 计算 hazard score
        
        Args:
            visual_features: [N, C, H, W]
            area_ratios: [N,]
            text_similarities: [N, num_prompts]
            global_features: [B, C]
            area_threshold: 自定义面积阈值 (默认使用 base_threshold)
            
        Returns:
            small_hazard_scores: [M,] 小目标的 hazard scores
            small_gates: [M,] 小目标的 soft gates
            small_indices: [M,] 小目标的原始索引
        """
        # 计算阈值
        t_adapt, Delta_t = self._compute_threshold(global_features, num_candidates=visual_features.shape[0])
        
        # 确保 threshold 是标量
        if isinstance(t_adapt, torch.Tensor):
            threshold = t_adapt.item() if t_adapt.numel() == 1 else t_adapt[0].item()
        else:
            threshold = t_adapt
        
        if area_threshold is not None:
            threshold = area_threshold
        
        # 找出 small candidates
        small_mask = area_ratios < threshold
        small_indices = torch.where(small_mask)[0]
        
        if len(small_indices) == 0:
            # 没有小目标
            return (
                torch.empty(0, device=visual_features.device),
                torch.empty(0, device=visual_features.device),
                small_indices
            )
        
        # 提取小目标特征
        if visual_features.dim() == 4:
            vis_small = visual_features[small_indices]
        else:
            vis_small = visual_features[small_indices]
        
        area_small = area_ratios[small_indices]
        text_small = text_similarities[small_indices] if text_similarities is not None else None
        
        # 计算 hazard scores
        hazard_small = self.hazard_scorer(vis_small, area_small, text_small)
        
        # 计算 soft gate
        # 确保 threshold 是标量
        if isinstance(threshold, torch.Tensor):
            thresh_val = threshold.item() if threshold.numel() == 1 else threshold[0].item()
        else:
            thresh_val = threshold
            
        if self.use_soft_gate or self.training:
            small_gates = torch.sigmoid(self.beta * (hazard_small - thresh_val))
        else:
            small_gates = (hazard_small > thresh_val).float()
        
        return hazard_small, small_gates, small_indices


class HazardScorerAblation(nn.Module):
    """
    支持 Ablation 实验的 Hazard Scorer
    
    支持的变体:
        1. Fixed Threshold (基线)
        2. Adaptive Threshold
        3. Adaptive Threshold + Soft Gate
    """
    
    VARIANTS = ['fixed', 'adaptive', 'adaptive_soft']
    
    def __init__(
        self,
        variant: str = 'adaptive_soft',
        
        # Hazard Scorer 参数
        visual_channels: int = 256,
        text_channels: int = 512,
        hidden_dim: int = 256,
        num_prompts: int = 2,
        output_size: int = 7,
        
        # Threshold 参数
        base_threshold: float = 0.01,
        alpha: float = 0.02,
        threshold_hidden_dim: int = 64,
        
        # Gate 参数
        beta: float = 10.0
    ):
        """
        Args:
            variant: 'fixed', 'adaptive', 'adaptive_soft'
        """
        super().__init__()
        assert variant in self.VARIANTS, f"Unknown variant: {variant}"
        
        self.variant = variant
        self.base_threshold = base_threshold
        self.beta = beta
        
        # Hazard Scorer
        self.hazard_scorer = HazardScorer(
            visual_channels=visual_channels,
            text_channels=text_channels,
            hidden_dim=hidden_dim,
            num_prompts=num_prompts,
            output_size=output_size,
            use_spatial=True
        )
        
        # Adaptive Threshold (仅 adaptive 变体使用)
        if variant in ['adaptive', 'adaptive_soft']:
            self.threshold_offset = AdaptiveThresholdOffset(
                feature_channels=visual_channels,
                hidden_dim=threshold_hidden_dim,
                alpha=alpha
            )
    
    def forward(
        self,
        visual_features: torch.Tensor,
        area_ratios: torch.Tensor,
        text_similarities: Optional[torch.Tensor] = None,
        global_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        返回包含所有信息的字典，方便 ablation 分析
        """
        # 计算阈值
        if self.variant == 'fixed':
            threshold = self.base_threshold
            Delta_t = torch.tensor(0.0, device=visual_features.device)
        else:
            if global_features is not None:
                Delta_t = self.threshold_offset(global_features)
                # 处理维度问题
                if isinstance(Delta_t, torch.Tensor) and Delta_t.numel() == 1:
                    threshold = self.base_threshold + Delta_t.item()
                elif isinstance(Delta_t, torch.Tensor):
                    threshold = self.base_threshold + Delta_t[0].item()
                else:
                    threshold = self.base_threshold + Delta_t
            else:
                threshold = self.base_threshold
                Delta_t = torch.tensor(0.0, device=visual_features.device)
        
        # Hazard scores
        hazard_scores = self.hazard_scorer(
            visual_features, area_ratios, text_similarities
        )
        
        # 计算 gates
        if self.variant == 'adaptive_soft':
            # Soft gate
            gates = torch.sigmoid(self.beta * (hazard_scores - threshold))
        else:
            # Hard gate
            gates = (hazard_scores > threshold).float()
        
        # Small mask
        is_small = area_ratios < threshold
        
        return {
            'hazard_scores': hazard_scores,
            'gates': gates,
            'threshold': threshold,
            'Delta_t': Delta_t,
            'is_small': is_small,
            'variant': self.variant
        }
    
    def set_variant(self, variant: str):
        """切换变体"""
        assert variant in self.VARIANTS
        self.variant = variant


# ============ 便捷函数 ============

def create_hazard_scorer(
    mode: str = 'adaptive_soft',
    **kwargs
) -> HazardScorerWithAdaptiveThreshold:
    """
    创建 Hazard Scorer 的便捷函数
    
    Args:
        mode: 'fixed', 'adaptive', 'soft', 'hard'
        **kwargs: 其他参数
    """
    if mode == 'fixed':
        return HazardScorerWithAdaptiveThreshold(
            use_adaptive_threshold=False,
            use_soft_gate=False,
            **kwargs
        )
    elif mode == 'adaptive':
        return HazardScorerWithAdaptiveThreshold(
            use_adaptive_threshold=True,
            use_soft_gate=False,
            **kwargs
        )
    elif mode == 'soft':
        return HazardScorerWithAdaptiveThreshold(
            use_adaptive_threshold=True,
            use_soft_gate=True,
            **kwargs
        )
    elif mode == 'hard':
        return HazardScorerWithAdaptiveThreshold(
            use_adaptive_threshold=True,
            use_soft_gate=False,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ============ 测试 ============

def test_hazard_scorer_with_adaptive_threshold():
    """测试集成模块"""
    print("=" * 60)
    print("Testing HazardScorerWithAdaptiveThreshold...")
    print("=" * 60)
    
    B = 2  # Batch size
    N = 10  # 候选数量
    C = 256
    
    # 模拟输入
    visual_features = torch.randn(N, C, 7, 7)
    area_ratios = torch.rand(N) * 0.05
    text_similarities = torch.rand(N, 2)
    global_features = torch.randn(B, C)
    
    # Test 1: Adaptive + Soft Gate
    print("\n[1] Testing Adaptive + Soft Gate...")
    model = HazardScorerWithAdaptiveThreshold(
        visual_channels=C,
        text_channels=512,
        hidden_dim=256,
        num_prompts=2,
        base_threshold=0.01,
        alpha=0.02,
        beta=10.0,
        use_adaptive_threshold=True,
        use_soft_gate=True
    )
    
    hazard_scores, soft_gates = model(
        visual_features, area_ratios, text_similarities, global_features
    )
    print(f"  Hazard scores: {hazard_scores.shape}")
    print(f"  Soft gates: {soft_gates.shape}, range: [{soft_gates.min():.3f}, {soft_gates.max():.3f}]")
    
    # Test 2: 返回详细信息
    print("\n[2] Testing with return_details=True...")
    result = model(
        visual_features, area_ratios, text_similarities, global_features,
        return_details=True
    )
    print(f"  Threshold: {result.thresholds.item():.4f}")
    # Handle Delta_t which may have multiple elements
    delta_val = result.Delta_t.item() if result.Delta_t.numel() == 1 else result.Delta_t[0].item()
    print(f"  Delta_t: {delta_val:.6f}")
    print(f"  Small indices: {result.small_indices.tolist()}")
    
    # Test 3: 固定阈值模式
    print("\n[3] Testing Fixed Threshold...")
    model_fixed = HazardScorerWithAdaptiveThreshold(
        visual_channels=C,
        use_adaptive_threshold=False,
        use_soft_gate=True,
        base_threshold=0.01
    )
    
    hazard_fixed, gates_fixed = model_fixed(
        visual_features, area_ratios, text_similarities, global_features
    )
    print(f"  Gates (fixed): {gates_fixed[:3].tolist()}")
    
    # Test 4: 推理模式切换
    print("\n[4] Testing inference mode switch...")
    model.set_inference_mode(use_hard_threshold=True)
    hazard_hard, gates_hard = model(
        visual_features, area_ratios, text_similarities, global_features
    )
    print(f"  Gates (hard): {gates_hard[:3].tolist()}")
    
    # Test 5: Ablation 变体
    print("\n[5] Testing Ablation Variants...")
    for variant in ['fixed', 'adaptive', 'adaptive_soft']:
        model_ablation = HazardScorerAblation(
            variant=variant,
            visual_channels=C,
            base_threshold=0.01,
            alpha=0.02
        )
        result = model_ablation(
            visual_features, area_ratios, text_similarities, global_features
        )
        # Handle both tensor and float
        thresh_val = result['threshold'].item() if isinstance(result['threshold'], torch.Tensor) else result['threshold']
        gates_mean = result['gates'].mean().item()
        print(f"  {variant}: threshold={thresh_val:.4f}, gates_mean={gates_mean:.3f}")
    
    # Test 6: 梯度检查
    print("\n[6] Testing gradient...")
    model.train()
    visual_features.requires_grad = True
    hazard_scores, soft_gates = model(
        visual_features, area_ratios, text_similarities, global_features
    )
    loss = soft_gates.sum()
    loss.backward()
    print(f"  Gradient exists: {visual_features.grad is not None}")
    print(f"  Gradient norm: {visual_features.grad.norm().item():.4f}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_hazard_scorer_with_adaptive_threshold()
