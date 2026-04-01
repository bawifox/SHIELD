"""
Hazard Scorer Module
对 small candidate regions 进行 hazard scoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class HazardScore:
    """Hazard score 结果"""
    scores: torch.Tensor      # [N,] 每个候选的危险分数
    indices: torch.Tensor     # [N,] 对应的候选索引


class HazardScorer(nn.Module):
    """
    Hazard Scorer - 对小目标候选区域进行危险度评分
    
    输入:
        - region visual feature: 区域视觉特征 [N, C, H, W] 或 [N, C]
        - area ratio: 面积比例 [N,]
        - text similarity: 与 text prompts 的相似度 [N, T] (T=num_prompts)
    
    输出:
        - hazard score: [N,] 每个小目标的危险分数
    
    结构:
        - Visual MLP: 处理视觉特征
        - Scale MLP: 将 area ratio 转为 scale embedding
        - Fusion: 拼接所有特征，MLP 输出危险分数
    """
    
    def __init__(
        self,
        visual_channels: int = 256,
        text_channels: int = 512,
        hidden_dim: int = 256,
        num_prompts: int = 2,
        output_size: int = 7,
        use_spatial: bool = True
    ):
        """
        Args:
            visual_channels: 视觉特征通道数
            text_channels: 文本特征通道数
            hidden_dim: 隐藏层维度
            num_prompts: text prompts 数量
            output_size: 视觉特征的空间尺寸
            use_spatial: 是否使用空间视觉特征
        """
        super().__init__()
        self.visual_channels = visual_channels
        self.text_channels = text_channels
        self.hidden_dim = hidden_dim
        self.num_prompts = num_prompts
        self.output_size = output_size
        self.use_spatial = use_spatial
        
        # ===== Visual MLP =====
        if use_spatial:
            # 空间视觉特征: [N, C, H, W] -> flatten -> MLP
            visual_input_dim = visual_channels * output_size * output_size
            self.visual_mlp = nn.Sequential(
                nn.Flatten(),
                nn.Linear(visual_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
        else:
            # 池化后的视觉特征: [N, C]
            self.visual_mlp = nn.Sequential(
                nn.Linear(visual_channels, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
        
        # ===== Scale MLP (area ratio -> scale embedding) =====
        self.scale_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU()
        )
        
        # ===== Text Similarity Processing =====
        # 对 text similarity 做简单处理
        self.text_mlp = nn.Sequential(
            nn.Linear(num_prompts, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU()
        )
        
        # ===== Fusion MLP =====
        # 拼接: visual + scale + text -> hazard score
        fusion_input_dim = hidden_dim + hidden_dim + hidden_dim  # 3 * hidden_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(
        self,
        visual_features: torch.Tensor,
        area_ratios: torch.Tensor,
        text_similarities: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算 hazard score
        
        Args:
            visual_features: [N, C, H, W] 或 [N, C] 视觉特征
            area_ratios: [N,] 面积比例
            text_similarities: [N, num_prompts] 与 text prompts 的相似度
            
        Returns:
            hazard_scores: [N,] 危险分数
        """
        N = visual_features.shape[0]
        
        # ===== Visual Feature Processing =====
        if self.use_spatial and visual_features.dim() == 4:
            # 空间特征 [N, C, H, W]
            visual_emb = self.visual_mlp(visual_features)  # [N, hidden_dim]
        else:
            # 池化特征 [N, C]
            if visual_features.dim() == 3:
                # [N, C, 1, 1] -> [N, C]
                visual_features = visual_features.squeeze(-1).squeeze(-1)
            visual_emb = self.visual_mlp(visual_features)  # [N, hidden_dim]
        
        # ===== Scale Processing =====
        area_ratios = area_ratios.view(-1, 1)  # [N, 1]
        scale_emb = self.scale_mlp(area_ratios)  # [N, hidden_dim]
        
        # ===== Text Similarity Processing =====
        if text_similarities is not None:
            text_emb = self.text_mlp(text_similarities)  # [N, hidden_dim]
        else:
            # 如果没有 text similarity，用零填充
            text_emb = torch.zeros(N, self.hidden_dim, device=visual_features.device)
        
        # ===== Fusion =====
        fused = torch.cat([visual_emb, scale_emb, text_emb], dim=-1)  # [N, 3*hidden_dim]
        hazard_scores = self.fusion_mlp(fused)  # [N, 1]
        hazard_scores = hazard_scores.squeeze(-1)  # [N,]
        
        return hazard_scores
    
    def forward_with_filter(
        self,
        visual_features: torch.Tensor,
        area_ratios: torch.Tensor,
        text_similarities: Optional[torch.Tensor] = None,
        area_threshold: float = 0.01
    ) -> Tuple[HazardScore, List[int]]:
        """
        先过滤 small candidates，再计算 hazard score
        
        Args:
            visual_features: [N, C, H, W] 或 [N, C]
            area_ratios: [N,]
            text_similarities: [N, num_prompts]
            area_threshold: 面积比例阈值，小于此值认为是 small candidate
            
        Returns:
            hazard_score: HazardScore 对象
            small_indices: 小目标的原始索引
        """
        # 找出 small candidates
        small_mask = area_ratios < area_threshold
        small_indices = torch.where(small_mask)[0].tolist()
        
        if len(small_indices) == 0:
            # 没有小目标，返回零分
            return HazardScore(
                scores=torch.zeros(len(area_ratios), device=area_ratios.device),
                indices=area_ratios.new_tensor([], dtype=torch.long)
            ), []
        
        # 提取小目标的特征
        if visual_features.dim() == 4:
            visual_small = visual_features[small_indices]  # [M, C, H, W]
        else:
            visual_small = visual_features[small_indices]
        
        area_small = area_ratios[small_indices]  # [M,]
        
        text_small = None
        if text_similarities is not None:
            text_small = text_similarities[small_indices]  # [M, num_prompts]
        
        # 计算 hazard score
        scores_small = self.forward(visual_small, area_small, text_small)  # [M,]
        
        # 填充完整结果
        full_scores = torch.zeros(len(area_ratios), device=area_ratios.device)
        full_scores[small_indices] = scores_small
        
        return HazardScore(
            scores=full_scores,
            indices=small_indices
        ), small_indices


class HazardScorerSimple(nn.Module):
    """
    简化的 Hazard Scorer
    使用更轻量的结构
    """
    
    def __init__(
        self,
        visual_channels: int = 256,
        text_channels: int = 512,
        hidden_dim: int = 128,
        num_prompts: int = 2
    ):
        super().__init__()
        self.visual_channels = visual_channels
        self.text_channels = text_channels
        self.hidden_dim = hidden_dim
        self.num_prompts = num_prompts
        
        # 视觉特征先池化
        self.visual_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.visual_fc = nn.Linear(visual_channels, hidden_dim)
        
        # Scale: MLP with output hidden_dim
        self.scale_fc = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Text MLP
        self.text_fc = nn.Sequential(
            nn.Linear(num_prompts, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Output: 3 * hidden_dim -> 1
        self.output_fc = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(
        self,
        visual_features: torch.Tensor,
        area_ratios: torch.Tensor,
        text_similarities: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            visual_features: [N, C, H, W]
            area_ratios: [N,]
            text_similarities: [N, num_prompts]
            
        Returns:
            hazard_scores: [N,]
        """
        N = visual_features.shape[0]
        
        # Visual: [N, C, H, W] -> [N, C] -> [N, hidden_dim]
        vis = self.visual_pool(visual_features).squeeze(-1).squeeze(-1)
        vis = F.relu(self.visual_fc(vis))
        
        # Scale: [N, 1] -> [N, hidden_dim]
        scale = self.scale_fc(area_ratios.unsqueeze(-1))
        
        # Text: [N, num_prompts] -> [N, hidden_dim]
        if text_similarities is not None:
            text = F.relu(self.text_fc(text_similarities))
        else:
            text = torch.zeros(N, self.hidden_dim, device=visual_features.device)
        
        # Concat & output: vis(hidden_dim) + scale(hidden_dim) + text(hidden_dim) = 3*hidden_dim
        fused = torch.cat([vis, scale, text], dim=-1)  # [N, 3*hidden_dim]
        
        scores = self.output_fc(fused).squeeze(-1)
        
        return scores


# ============ 测试 ============

def test_hazard_scorer():
    """测试 Hazard Scorer"""
    print("=" * 60)
    print("Testing HazardScorer...")
    print("=" * 60)
    
    N = 10  # 候选数量
    C = 256
    num_prompts = 2
    
    # 模拟输入
    visual_features = torch.randn(N, C, 7, 7)
    area_ratios = torch.rand(N) * 0.05  # 小目标
    text_similarities = torch.rand(N, num_prompts)
    
    # 测试完整版
    print("\n[1] Testing HazardScorer (full version)...")
    scorer = HazardScorer(
        visual_channels=C,
        text_channels=512,
        hidden_dim=256,
        num_prompts=num_prompts,
        output_size=7,
        use_spatial=True
    )
    
    scores = scorer(visual_features=visual_features, area_ratios=area_ratios, text_similarities=text_similarities)
    print(f"  Input: visual {visual_features.shape}, area {area_ratios.shape}, text {text_similarities.shape}")
    print(f"  Output: {scores.shape}, values: {scores[:3].tolist()}")
    
    # 测试带过滤的版本
    print("\n[2] Testing HazardScorer with filter...")
    hazard_score, small_indices = scorer.forward_with_filter(
        visual_features, area_ratios, text_similarities, area_threshold=0.01
    )
    print(f"  Small candidates: {len(small_indices)}")
    print(f"  Scores shape: {hazard_score.scores.shape}")
    
    # 测试简化版
    print("\n[3] Testing HazardScorerSimple...")
    scorer_simple = HazardScorerSimple(
        visual_channels=C,
        text_channels=512,
        hidden_dim=128,
        num_prompts=num_prompts
    )
    
    scores_simple = scorer_simple(visual_features, area_ratios, text_similarities)
    print(f"  Output: {scores_simple.shape}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_hazard_scorer()
