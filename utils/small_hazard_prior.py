"""
Small-Hazard Prior Generation Module
根据 small candidate masks 和 hazard weights 生成 prior map
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SmallHazardPriorOutput:
    """Small hazard prior 输出"""
    prior_map: torch.Tensor      # [B, 1, H, W] 或 [B, H, W]
    num_candidates: int          # 候选区域数量


class SmallHazardPriorGenerator(nn.Module):
    """
    Small-Hazard Prior 生成器
    
    输入:
        - small candidate masks: 二值掩码 [B, N, H, W] 或 list of [N_i, H, W]
        - soft gates / hazard weights: [B, N] 或 [N,]
        
    输出:
        - small_hazard_prior: [B, 1, H, W] 与 coarse anomaly map 尺寸一致
        
    特点:
        - 多区域重叠时取 max
        - 支持 batch 处理
        - 支持不同尺寸的输入 mask
    """
    
    def __init__(
        self,
        output_size: Optional[Tuple[int, int]] = None,
        use_softmax: bool = False,
        fill_value: float = 1.0
    ):
        """
        Args:
            output_size: 输出尺寸 (H, W)，如果为 None 则使用输入 mask 尺寸
            use_softmax: 是否对 weights 做 softmax
            fill_value: 填充值（当 mask 为 1 时的权重乘数）
        """
        super().__init__()
        self.output_size = output_size
        self.use_softmax = use_softmax
        self.fill_value = fill_value
    
    def forward(
        self,
        masks: torch.Tensor,
        weights: torch.Tensor,
        target_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        生成 small-hazard prior map
        
        Args:
            masks: 候选区域掩码
                - 格式 1: [B, N, H, W] 3D mask
                - 格式 2: [B, N, H, W] 2D mask (二值)
            weights: 权重 [B, N] 或 [N,]
            target_size: 目标尺寸 (H, W)，用于 resize
            
        Returns:
            prior_map: [B, 1, H', W'] 或 [B, H', W']
        """
        # 获取尺寸信息
        if masks.dim() == 4:
            B, N, H, W = masks.shape
        elif masks.dim() == 3:
            # [N, H, W] - 单样本
            N, H, W = masks.shape
            B = 1
            masks = masks.unsqueeze(0)  # [1, N, H, W]
        else:
            raise ValueError(f"Unexpected masks dim: {masks.dim()}")
        
        # 处理 weights 维度
        if weights.dim() == 1:
            weights = weights.unsqueeze(0)  # [1, N]
        
        # 确保 batch 维度匹配
        assert weights.shape[0] == B, f"Weights batch {weights.shape[0]} != Masks batch {B}"
        assert weights.shape[1] == N, f"Weights num {weights.shape[1]} != Masks num {N}"
        
        # Softmax 归一化 (可选)
        if self.use_softmax:
            weights = F.softmax(weights, dim=-1)
        
        # 扩展 weights 到 2D mask 空间
        # weights: [B, N] -> [B, N, 1, 1]
        weights_expanded = weights.view(B, N, 1, 1) * self.fill_value
        
        # 加权求和: [B, N, H, W] * [B, N, 1, 1] -> [B, N, H, W]
        weighted_masks = masks.float() * weights_expanded
        
        # 多区域取 max: [B, N, H, W] -> [B, 1, H, W]
        prior_map, _ = torch.max(weighted_masks, dim=1, keepdim=True)
        
        # 限制到 [0, 1]
        prior_map = torch.clamp(prior_map, 0.0, 1.0)
        
        # Resize 到目标尺寸
        target_h, target_w = target_size or self.output_size or (H, W)
        if (target_h, target_w) != (H, W):
            prior_map = F.interpolate(
                prior_map, 
                size=(target_h, target_w), 
                mode='bilinear', 
                align_corners=False
            )
        
        return prior_map
    
    def forward_list(
        self,
        mask_list: List[torch.Tensor],
        weights_list: List[torch.Tensor],
        target_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        处理 list 格式的输入 (支持不同数量的候选)
        
        Args:
            mask_list: List of [N_i, H, W] masks, len = B
            weights_list: List of [N_i,] weights, len = B
            target_size: 目标尺寸
            
        Returns:
            prior_map: [B, 1, H', W']
        """
        B = len(mask_list)
        
        # 获取最大尺寸
        H, W = mask_list[0].shape[-2:]
        
        # 处理每个样本
        batch_priors = []
        for i in range(B):
            masks_i = mask_list[i]  # [N_i, H, W]
            weights_i = weights_list[i]  # [N_i,]
            
            N_i = masks_i.shape[0]
            
            # 扩展 weights
            if self.use_softmax:
                weights_i = F.softmax(weights_i, dim=-1)
            weights_i = weights_i.view(N_i, 1, 1) * self.fill_value
            
            # 加权求和
            weighted = masks_i.float() * weights_i  # [N_i, H, W]
            
            # Max pooling
            prior_i, _ = torch.max(weighted, dim=0, keepdim=True)  # [1, H, W]
            prior_i = torch.clamp(prior_i, 0.0, 1.0)
            
            batch_priors.append(prior_i)
        
        # 堆叠
        prior_map = torch.stack(batch_priors, dim=0)  # [B, 1, H, W]
        
        # Resize
        target_h, target_w = target_size or self.output_size or (H, W)
        if (target_h, target_w) != (H, W):
            prior_map = F.interpolate(
                prior_map,
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False
            )
        
        return prior_map


class SmallHazardPriorGeneratorV2(nn.Module):
    """
    Small-Hazard Prior 生成器 V2
    支持更多种归一化方式和融合策略
    """
    
    FUSION_MODES = ['max', 'sum', 'weighted_sum']
    
    def __init__(
        self,
        output_size: Optional[Tuple[int, int]] = None,
        fusion_mode: str = 'max',
        normalize_weights: bool = True,
        temperature: float = 1.0
    ):
        """
        Args:
            output_size: 输出尺寸
            fusion_mode: 融合模式 ('max', 'sum', 'weighted_sum')
            normalize_weights: 是否归一化 weights
            temperature: softmax 温度
        """
        super().__init__()
        assert fusion_mode in self.FUSION_MODES
        self.output_size = output_size
        self.fusion_mode = fusion_mode
        self.normalize_weights = normalize_weights
        self.temperature = temperature
    
    def forward(
        self,
        masks: torch.Tensor,
        weights: torch.Tensor,
        target_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        Args:
            masks: [B, N, H, W] 或 [N, H, W]
            weights: [B, N] 或 [N,]
            target_size: (H, W)
            
        Returns:
            prior_map: [B, 1, H, W]
        """
        # 处理输入维度
        if masks.dim() == 3:
            masks = masks.unsqueeze(0)
            weights = weights.unsqueeze(0) if weights.dim() == 1 else weights
            single_sample = True
        else:
            single_sample = False
        
        B, N, H, W = masks.shape
        
        # 归一化 weights
        if self.normalize_weights:
            weights = F.softmax(weights / self.temperature, dim=-1)
        
        # 扩展 weights
        weights = weights.view(B, N, 1, 1)
        
        # 加权
        weighted = masks.float() * weights  # [B, N, H, W]
        
        # 融合
        if self.fusion_mode == 'max':
            prior_map, _ = torch.max(weighted, dim=1, keepdim=True)
        elif self.fusion_mode == 'sum':
            prior_map = weighted.sum(dim=1, keepdim=True)
        elif self.fusion_mode == 'weighted_sum':
            prior_map = weighted.sum(dim=1, keepdim=True)
            # 归一化
            mask_count = (masks > 0).float().sum(dim=1, keepdim=True).clamp(min=1.0)
            prior_map = prior_map / mask_count
        
        # 限制范围
        prior_map = torch.clamp(prior_map, 0.0, 1.0)
        
        # Resize
        target_h, target_w = target_size or self.output_size or (H, W)
        if (target_h, target_w) != (H, W):
            prior_map = F.interpolate(
                prior_map, size=(target_h, target_w),
                mode='bilinear', align_corners=False
            )
        
        # 如果输入是单样本，返回 [1, H, W] 而不是 [1, 1, H, W]
        if single_sample:
            prior_map = prior_map.squeeze(1)
        
        return prior_map


class SmallHazardPriorGeneratorSoftGating(nn.Module):
    """
    Small-Hazard Prior 生成器 - 软门控版本
    直接使用 soft gate 作为权重
    """
    
    def __init__(
        self,
        output_size: Optional[Tuple[int, int]] = None,
        use_sigmoid: bool = True
    ):
        super().__init__()
        self.output_size = output_size
        self.use_sigmoid = use_sigmoid
    
    def forward(
        self,
        masks: torch.Tensor,
        soft_gates: torch.Tensor,
        target_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        Args:
            masks: [B, N, H, W]
            soft_gates: [B, N]
            target_size: (H, W)
            
        Returns:
            prior_map: [B, 1, H, W]
        """
        # 处理维度
        if masks.dim() == 3:
            masks = masks.unsqueeze(0)
            soft_gates = soft_gates.unsqueeze(0)
        
        B, N, H, W = masks.shape
        
        # 应用 sigmoid (可选)
        if self.use_sigmoid:
            weights = torch.sigmoid(soft_gates)
        else:
            weights = soft_gates
        
        # 扩展并加权
        weights = weights.view(B, N, 1, 1)
        weighted = masks.float() * weights
        
        # Max fusion
        prior_map, _ = torch.max(weighted, dim=1, keepdim=True)
        prior_map = torch.clamp(prior_map, 0.0, 1.0)
        
        # Resize
        target_h, target_w = target_size or self.output_size or (H, W)
        if (target_h, target_w) != (H, W):
            prior_map = F.interpolate(
                prior_map, size=(target_h, target_w),
                mode='bilinear', align_corners=False
            )
        
        return prior_map


# ============ 便捷函数 ============

def create_small_hazard_prior(
    mode: str = 'max',
    **kwargs
) -> SmallHazardPriorGenerator:
    """创建 prior 生成器"""
    if mode == 'max':
        return SmallHazardPriorGenerator(**kwargs)
    elif mode == 'v2':
        return SmallHazardPriorGeneratorV2(**kwargs)
    elif mode == 'soft':
        return SmallHazardPriorGeneratorSoftGating(**kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ============ 测试 ============

def test_small_hazard_prior():
    """测试 Small Hazard Prior 生成器"""
    print("=" * 60)
    print("Testing SmallHazardPriorGenerator...")
    print("=" * 60)
    
    B, N = 2, 5
    H, W = 64, 64
    target_h, target_w = 128, 256
    
    # 模拟输入
    masks = torch.randint(0, 2, (B, N, H, W)).float()  # [B, N, H, W]
    weights = torch.rand(B, N)  # [B, N]
    
    # Test 1: 基础版本
    print("\n[1] Testing Basic Version...")
    generator = SmallHazardPriorGenerator(
        output_size=(target_h, target_w),
        use_softmax=True
    )
    
    prior = generator(masks, weights, target_size=(target_h, target_w))
    print(f"  Input: masks {masks.shape}, weights {weights.shape}")
    print(f"  Output: {prior.shape}")
    print(f"  Range: [{prior.min().item():.3f}, {prior.max().item():.3f}]")
    
    # Test 2: List 输入
    print("\n[2] Testing List Input...")
    generator_list = SmallHazardPriorGenerator(
        output_size=(target_h, target_w),
        use_softmax=True
    )
    
    mask_list = [torch.randint(0, 2, (N, H, W)).float() for _ in range(B)]
    weights_list = [torch.rand(N) for _ in range(B)]
    
    prior_list = generator_list.forward_list(
        mask_list, weights_list, 
        target_size=(target_h, target_w)
    )
    print(f"  Output: {prior_list.shape}")
    
    # Test 3: V2 版本
    print("\n[3] Testing V2 Version...")
    generator_v2 = SmallHazardPriorGeneratorV2(
        output_size=(target_h, target_w),
        fusion_mode='max',
        normalize_weights=True
    )
    
    prior_v2 = generator_v2(masks, weights, target_size=(target_h, target_w))
    print(f"  Output: {prior_v2.shape}")
    
    # Test 4: 软门控版本
    print("\n[4] Testing Soft Gating Version...")
    generator_soft = SmallHazardPriorGeneratorSoftGating(
        output_size=(target_h, target_w),
        use_sigmoid=True
    )
    
    soft_gates = torch.rand(B, N)
    prior_soft = generator_soft(masks, soft_gates, target_size=(target_h, target_w))
    print(f"  Output: {prior_soft.shape}")
    
    # Test 5: 重叠区域测试
    print("\n[5] Testing Overlapping Regions...")
    # 创建两个重叠的 mask
    overlap_masks = torch.zeros(1, 2, H, W)
    overlap_masks[0, 0, 10:30, 10:30] = 1.0  # 区域 1
    overlap_masks[0, 1, 20:40, 20:40] = 1.0  # 区域 2 (与区域 1 重叠)
    
    # 不同权重
    weights_overlap = torch.tensor([[0.3, 0.8]])  # 区域 2 权重更高
    
    prior_overlap = generator(overlap_masks, weights_overlap)
    print(f"  Overlap region shape: {prior_overlap.shape}")
    print(f"  Max value (should be 0.8): {prior_overlap.max().item():.3f}")
    
    # Test 6: 梯度检查
    print("\n[6] Testing Gradient...")
    generator.train()
    masks.requires_grad = True
    weights.requires_grad = True
    
    prior = generator(masks, weights)
    loss = prior.sum()
    loss.backward()
    
    print(f"  Masks gradient exists: {masks.grad is not None}")
    print(f"  Weights gradient exists: {weights.grad is not None}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_small_hazard_prior()
