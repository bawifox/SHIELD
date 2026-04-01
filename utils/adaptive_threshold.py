"""
Image-Adaptive Threshold Offset Module
用于动态调整 small candidate 的筛选阈值
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AdaptiveThresholdOffset(nn.Module):
    """
    Image-Adaptive Threshold Offset
    
    根据整张图的全局视觉特征，预测一个阈值偏移量 Delta_t
    用来修正 small candidate 的筛选阈值: t_adapt = t2 + Delta_t
    
    输入:
        - global visual feature: 全局视觉特征 (来自 backbone 最深层 GAP)
        
    输出:
        - Delta_t: 阈值偏移量，范围 [-alpha, alpha]
    
    结构:
        - Global Feature Processing: FC layers
        - Range Control: tanh * alpha
    """
    
    def __init__(
        self,
        feature_channels: int = 256,
        hidden_dim: int = 64,
        alpha: float = 0.02,
        use_batch_norm: bool = False
    ):
        """
        Args:
            feature_channels: 输入特征通道数
            hidden_dim: 隐藏层维度
            alpha: Delta_t 的最大绝对值 (范围 [-alpha, alpha])
            use_batch_norm: 是否使用 BatchNorm
        """
        super().__init__()
        self.feature_channels = feature_channels
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        
        # MLP: GAP feature -> hidden -> 1
        if use_batch_norm:
            self.mlp = nn.Sequential(
                nn.Linear(feature_channels, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(feature_channels, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
    
    def forward(self, global_features: torch.Tensor) -> torch.Tensor:
        """
        计算阈值偏移量
        
        Args:
            global_features: [B, C] 或 [B, C, 1, 1] 全局视觉特征
            
        Returns:
            Delta_t: [B,] 阈值偏移量，范围 [-alpha, alpha]
        """
        # 统一形状: [B, C]
        if global_features.dim() == 4:
            # [B, C, 1, 1] -> [B, C]
            global_features = global_features.squeeze(-1).squeeze(-1)
        elif global_features.dim() == 2:
            pass  # [B, C]
        
        # MLP forward
        delta = self.mlp(global_features)  # [B, 1]
        
        # Tanh 控制范围: [-alpha, alpha]
        delta = torch.tanh(delta) * self.alpha
        
        # 展平为 [B,]
        delta = delta.squeeze(-1)
        
        return delta
    
    def forward_with_threshold(
        self,
        global_features: torch.Tensor,
        base_threshold: float = 0.01
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        计算调整后的阈值
        
        Args:
            global_features: [B, C] 全局视觉特征
            base_threshold: 基础阈值 t2
            
        Returns:
            t_adapt: [B,] 调整后的阈值
            Delta_t: [B,] 偏移量
        """
        Delta_t = self.forward(global_features)
        t_adapt = base_threshold + Delta_t
        
        return t_adapt, Delta_t
    
    def get_alpha(self) -> float:
        """获取 alpha 值"""
        return self.alpha
    
    def set_alpha(self, alpha: float):
        """设置 alpha 值"""
        self.alpha = alpha


class AdaptiveThresholdOffsetV2(nn.Module):
    """
    Image-Adaptive Threshold Offset V2
    支持更多输入选项和更灵活的结构
    """
    
    def __init__(
        self,
        feature_channels: int = 256,
        hidden_dim: int = 32,
        alpha: float = 0.02,
        num_layers: int = 2,
        use_layer_norm: bool = False
    ):
        """
        Args:
            feature_channels: 输入特征通道数
            hidden_dim: 隐藏层维度
            alpha: Delta_t 的最大绝对值
            num_layers: MLP 层数 (2 或 3)
            use_layer_norm: 是否使用 LayerNorm
        """
        super().__init__()
        self.feature_channels = feature_channels
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        
        # 构建 MLP
        layers = []
        in_dim = feature_channels
        
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else 1
            layers.append(nn.Linear(in_dim, out_dim))
            
            if i < num_layers - 1:
                layers.append(nn.ReLU())
                if use_layer_norm:
                    layers.append(nn.LayerNorm(out_dim))
                layers.append(nn.Dropout(0.1))
            
            in_dim = out_dim
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, global_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            global_features: [B, C] 或 [B, C, 1, 1]
            
        Returns:
            Delta_t: [B,]
        """
        if global_features.dim() == 4:
            global_features = global_features.squeeze(-1).squeeze(-1)
        
        delta = self.mlp(global_features)
        delta = torch.tanh(delta) * self.alpha
        
        return delta.squeeze(-1)


class AdaptiveThresholdOffsetSoftGating(nn.Module):
    """
    Image-Adaptive Threshold Offset with Soft Gating
    支持软门控形式，可学习门控权重
    """
    
    def __init__(
        self,
        feature_channels: int = 256,
        hidden_dim: int = 64,
        alpha: float = 0.02,
        gating_init: float = 0.0
    ):
        """
        Args:
            feature_channels: 输入特征通道数
            hidden_dim: 隐藏层维度
            alpha: Delta_t 的最大绝对值
            gating_init: 门控初始值 (sigmoid(gating_init) 为初始门控值)
        """
        super().__init__()
        self.alpha = alpha
        
        # 特征处理
        self.feature_proj = nn.Linear(feature_channels, hidden_dim)
        
        # 偏移量预测
        self.offset_mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 软门控权重 (可学习)
        self.gate_weight = nn.Parameter(torch.tensor(gating_init))
    
    def forward(
        self, 
        global_features: torch.Tensor,
        return_gate: bool = False
    ) -> torch.Tensor:
        """
        Args:
            global_features: [B, C]
            return_gate: 是否返回门控值
            
        Returns:
            Delta_t: [B,] 或 (Delta_t, gate) 如果 return_gate=True
        """
        if global_features.dim() == 4:
            global_features = global_features.squeeze(-1).squeeze(-1)
        
        # 特征投影
        h = self.feature_proj(global_features)  # [B, hidden]
        
        # 预测原始偏移
        raw_offset = self.offset_mlp(h).squeeze(-1)  # [B,]
        
        # 软门控: gate * raw_offset
        gate = torch.sigmoid(self.gate_weight)
        Delta_t = gate * torch.tanh(raw_offset) * self.alpha
        
        if return_gate:
            return Delta_t, gate
        return Delta_t
    
    def get_gate_value(self) -> float:
        """获取当前门控值"""
        return torch.sigmoid(self.gate_weight).item()


# ============ 测试 ============

def test_adaptive_threshold_offset():
    """测试 Adaptive Threshold Offset"""
    print("=" * 60)
    print("Testing AdaptiveThresholdOffset...")
    print("=" * 60)
    
    B = 4
    C = 256
    
    # 测试输入: [B, C] 和 [B, C, 1, 1]
    global_features = torch.randn(B, C)
    global_features_4d = global_features.unsqueeze(-1).unsqueeze(-1)
    
    # Test 1: 基础版本
    print("\n[1] Testing AdaptiveThresholdOffset...")
    offset_module = AdaptiveThresholdOffset(
        feature_channels=C,
        hidden_dim=64,
        alpha=0.02
    )
    
    Delta_t = offset_module(global_features)
    print(f"  Input: {global_features.shape}")
    print(f"  Delta_t: {Delta_t.shape}, values: {Delta_t.tolist()}")
    print(f"  Range: [{Delta_t.min().item():.4f}, {Delta_t.max().item():.4f}]")
    
    t_adapt, delta = offset_module.forward_with_threshold(global_features, base_threshold=0.01)
    print(f"  t_adapt: {t_adapt.tolist()}")
    
    # Test 2: V2 版本
    print("\n[2] Testing AdaptiveThresholdOffsetV2...")
    offset_v2 = AdaptiveThresholdOffsetV2(
        feature_channels=C,
        hidden_dim=32,
        alpha=0.02,
        num_layers=2
    )
    
    Delta_t_v2 = offset_v2(global_features)
    print(f"  Delta_t: {Delta_t_v2.shape}, values: {Delta_t_v2.tolist()}")
    
    # Test 3: 软门控版本
    print("\n[3] Testing AdaptiveThresholdOffsetSoftGating...")
    offset_soft = AdaptiveThresholdOffsetSoftGating(
        feature_channels=C,
        hidden_dim=64,
        alpha=0.02
    )
    
    Delta_t_soft, gate = offset_soft(global_features, return_gate=True)
    print(f"  Delta_t: {Delta_t_soft.shape}, values: {Delta_t_soft.tolist()}")
    print(f"  Gate: {gate.item():.4f}")
    
    # Test 4: 梯度检查
    print("\n[4] Testing gradient...")
    global_features.requires_grad = True
    Delta_t = offset_module(global_features)
    loss = Delta_t.sum()
    loss.backward()
    print(f"  Gradient exists: {global_features.grad is not None}")
    print(f"  Gradient norm: {global_features.grad.norm().item():.4f}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_adaptive_threshold_offset()
