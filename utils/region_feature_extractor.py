"""
Region Feature Extraction Module
从 backbone feature map 中提取候选区域的特征
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from dataclasses import dataclass

from candidate_extractor import CandidateRegion


@dataclass
class RegionFeature:
    """Region 特征数据结构"""
    embedding: torch.Tensor      # [C, output_size, output_size] 或 [C]
    area_ratio: float          # 面积占原图比例
    bbox: Tuple[int, int, int, int]  # 归一化 bbox (x1, y1, x2, y2) 范围 [0, 1]


class RegionFeatureExtractor(nn.Module):
    """
    从 backbone feature map 中提取候选区域的特征
    
    Args:
        feature_channels: backbone 特征通道数
        output_size: 输出特征图尺寸 (output_size, output_size)
        feature_level: 使用哪一层特征 (0=c1, 1=c2, 2=c3, 3=c4)
    """
    
    def __init__(
        self,
        feature_channels: int = 256,
        output_size: int = 7,
        feature_level: int = 1
    ):
        super().__init__()
        self.feature_channels = feature_channels
        self.output_size = output_size
        self.feature_level = feature_level
        
        # 投影层确保通道一致
        self.proj = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)
    
    def forward(
        self,
        backbone_features: List[torch.Tensor],
        candidates_batch: List[List[CandidateRegion]],
        image_size: Tuple[int, int]
    ) -> Tuple[List[List[RegionFeature]], List[List[torch.Tensor]]]:
        """
        提取候选区域的特征
        
        Args:
            backbone_features: [c1, c2, c3, c4] 多尺度特征，每项 [B, C, H, W]
            candidates_batch: 每张图像的候选列表，长度为 B，每个元素是候选列表
            image_size: 原始图像尺寸 (H, W)
            
        Returns:
            region_features_batch: 每张图像的区域特征列表
            area_ratios_batch: 每张图像的面积比例列表
        """
        # 选择指定层级的特征
        if self.feature_level >= len(backbone_features):
            raise ValueError(
                f"feature_level {self.feature_level} out of range, "
                f"only {len(backbone_features)} feature levels available"
            )
        
        feature_map = backbone_features[self.feature_level]  # [B, C, Hf, Wf]
        B, C, Hf, Wf = feature_map.shape
        
        # 原始图像尺寸
        H, W = image_size
        
        # 投影
        feature_map = self.proj(feature_map)
        
        region_features_batch = []
        area_ratios_batch = []
        
        for b in range(B):
            candidates = candidates_batch[b]
            region_features = []
            area_ratios = []
            
            # 当前图像的特征图
            feat_b = feature_map[b]  # [C, Hf, Wf]
            
            for cand in candidates:
                # 转换 bbox 到特征图坐标
                x1, y1, x2, y2 = cand.bbox
                
                # 归一化到特征图坐标
                x1_feat = x1 / W * Wf
                y1_feat = y1 / H * Hf
                x2_feat = x2 / W * Wf
                y2_feat = y2 / H * Hf
                
                # 边界裁剪
                x1_feat = max(0, min(int(x1_feat), Wf - 1))
                x2_feat = max(x1_feat + 1, min(int(x2_feat), Wf))
                y1_feat = max(0, min(int(y1_feat), Hf - 1))
                y2_feat = max(y1_feat + 1, min(int(y2_feat), Hf))
                
                # 提取区域特征
                region_feat = feat_b[:, y1_feat:y2_feat, x1_feat:x2_feat]  # [C, h, w]
                
                # 使用自适应池化输出固定尺寸
                if region_feat.shape[1] > 0 and region_feat.shape[2] > 0:
                    region_feat = F.adaptive_avg_pool2d(
                        region_feat.unsqueeze(0), 
                        (self.output_size, self.output_size)
                    ).squeeze(0)  # [C, output_size, output_size]
                else:
                    # 无效区域，输出零特征
                    region_feat = torch.zeros(
                        C, self.output_size, self.output_size,
                        device=feat_b.device, dtype=feat_b.dtype
                    )
                
                # 归一化 bbox
                bbox_norm = (x1 / W, y1 / H, x2 / W, y2 / H)
                
                region_features.append(RegionFeature(
                    embedding=region_feat,
                    area_ratio=cand.area_ratio,
                    bbox=bbox_norm
                ))
                area_ratios.append(torch.tensor(cand.area_ratio, dtype=torch.float32))
            
            region_features_batch.append(region_features)
            area_ratios_batch.append(area_ratios)
        
        return region_features_batch, area_ratios_batch


class RegionFeatureExtractorBatch(nn.Module):
    """
    批量区域特征提取器
    输入 bboxes 格式，更适合批量处理
    
    Args:
        feature_channels: 输入特征通道数
        output_size: 输出空间尺寸
        output_type: 'spatial' -> [B, N, C, output_size, output_size]
                     'pooled' -> [B, N, C]
    """
    
    def __init__(
        self,
        feature_channels: int = 256,
        output_size: int = 7,
        output_type: str = 'spatial'
    ):
        super().__init__()
        self.feature_channels = feature_channels
        self.output_size = output_size
        self.output_type = output_type
        
        # 投影层
        self.proj = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)
    
    def forward(
        self,
        feature_map: torch.Tensor,
        bboxes: torch.Tensor,
        image_size: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        提取区域特征
        
        Args:
            feature_map: [B, C, H, W] 特征图
            bboxes: [B, N, 4] 归一化 bbox (x1, y1, x2, y2)，范围 [0, 1]
            image_size: 原始图像尺寸 (H, W)
            
        Returns:
            region_features: [B, N, C, output_size, output_size] 或 [B, N, C]
            area_ratios: [B, N]
        """
        B, C, Hf, Wf = feature_map.shape
        N = bboxes.shape[1]
        
        H, W = image_size
        
        # 投影
        feature_map = self.proj(feature_map)
        
        # 转换 bbox 到特征图坐标
        bboxes_feat = bboxes.clone()
        bboxes_feat[..., 0] = bboxes[..., 0] / W * Wf  # x1
        bboxes_feat[..., 1] = bboxes[..., 1] / H * Hf  # y1
        bboxes_feat[..., 2] = bboxes[..., 2] / W * Wf  # x2
        bboxes_feat[..., 3] = bboxes[..., 3] / H * Hf  # y2
        
        # 初始化输出
        if self.output_type == 'spatial':
            region_features = torch.zeros(
                B, N, C, self.output_size, self.output_size,
                device=feature_map.device, dtype=feature_map.dtype
            )
        else:
            region_features = torch.zeros(
                B, N, C,
                device=feature_map.device, dtype=feature_map.dtype
            )
        
        area_ratios = torch.zeros(B, N, device=feature_map.device)
        
        for b in range(B):
            feat_b = feature_map[b]  # [C, Hf, Wf]
            bboxes_b = bboxes_feat[b]  # [N, 4]
            
            for n in range(N):
                x1, y1, x2, y2 = bboxes_b[n].tolist()
                
                # 边界裁剪
                x1 = max(0, min(int(x1), Wf - 1))
                y1 = max(0, min(int(y1), Hf - 1))
                x2 = max(x1 + 1, min(int(x2), Wf))
                y2 = max(y1 + 1, min(int(y2), Hf))
                
                if x2 > x1 and y2 > y1:
                    # 提取区域
                    region = feat_b[:, y1:y2, x1:x2]  # [C, h, w]
                    
                    # 池化到固定尺寸
                    feat = F.adaptive_avg_pool2d(
                        region.unsqueeze(0),
                        (self.output_size, self.output_size)
                    ).squeeze(0)  # [C, output_size, output_size]
                    
                    if self.output_type == 'spatial':
                        region_features[b, n] = feat
                    else:
                        region_features[b, n] = feat.mean(dim=(1, 2))  # [C]
                
                # 计算面积比例
                area = (bboxes[b, n, 2] - bboxes[b, n, 0]) * \
                       (bboxes[b, n, 3] - bboxes[b, n, 1])
                area_ratios[b, n] = area
        
        return region_features, area_ratios


def extract_region_features_simple(
    feature_map: torch.Tensor,
    candidates: List[CandidateRegion],
    output_size: int = 7
) -> Tuple[List[torch.Tensor], List[float]]:
    """
    简单的单图区域特征提取函数
    
    Args:
        feature_map: [C, H, W] 特征图
        candidates: 候选区域列表
        output_size: 输出尺寸
        
    Returns:
        embeddings: 特征列表，每个 [C, output_size, output_size]
        area_ratios: 面积比例列表
    """
    C, Hf, Wf = feature_map.shape
    
    embeddings = []
    area_ratios = []
    
    for cand in candidates:
        x1, y1, x2, y2 = cand.bbox
        
        # 映射到特征图坐标
        x1_f = int(x1 / Wf * Wf)
        y1_f = int(y1 / Hf * Hf)
        x2_f = int(x2 / W * Wf)
        y2_f = int(y2 / H * Hf)
        
        # 边界裁剪
        x1_f, x2_f = max(0, x1_f), min(Wf, x2_f)
        y1_f, y2_f = max(0, y1_f), min(Hf, y2_f)
        
        if x2_f <= x1_f or y2_f <= y1_f:
            region_feat = torch.zeros(C, output_size, output_size, dtype=feature_map.dtype)
        else:
            region = feature_map[:, y1_f:y2_f, x1_f:x2_f]
            region_feat = F.adaptive_avg_pool2d(
                region.unsqueeze(0),
                (output_size, output_size)
            ).squeeze(0)
        
        embeddings.append(region_feat)
        area_ratios.append(cand.area_ratio)
    
    return embeddings, area_ratios


# ============ 测试 ============

def test_region_feature_extractor():
    """测试 Region Feature Extractor"""
    import numpy as np
    
    print("=" * 60)
    print("Testing RegionFeatureExtractor...")
    print("=" * 60)
    
    # 模拟 backbone features
    B, C = 2, 256
    H, W = 512, 1024
    
    # 多尺度特征
    c1 = torch.randn(B, 64, H // 4, W // 4)
    c2 = torch.randn(B, 128, H // 8, W // 8)
    c3 = torch.randn(B, 320, H // 16, W // 16)
    c4 = torch.randn(B, 512, H // 32, W // 32)
    features = [c1, c2, c3, c4]
    
    # 模拟候选区域
    candidates_batch = []
    for b in range(B):
        candidates = [
            CandidateRegion(
                mask=np.random.rand(H, W) > 0.8,
                bbox=(100, 100, 200, 200),
                mean_score=0.7,
                max_score=0.9,
                area=10000,
                area_ratio=10000 / (H * W)
            ),
            CandidateRegion(
                mask=np.random.rand(H, W) > 0.9,
                bbox=(400, 300, 450, 350),
                mean_score=0.6,
                max_score=0.8,
                area=2500,
                area_ratio=2500 / (H * W)
            )
        ]
        candidates_batch.append(candidates)
    
    # 测试 V1
    extractor = RegionFeatureExtractor(
        feature_channels=128,
        output_size=7,
        feature_level=1
    )
    
    region_features_batch, area_ratios_batch = extractor(
        features, candidates_batch, (H, W)
    )
    
    print(f"\nBatch size: {B}")
    for b in range(B):
        print(f"  Image {b}: {len(region_features_batch[b])} candidates")
        for i, rf in enumerate(region_features_batch[b]):
            print(f"    Candidate {i}: embedding shape = {rf.embedding.shape}, "
                  f"area_ratio = {rf.area_ratio:.4f}")
    
    # 测试 V2 (Batch)
    print("\n" + "=" * 60)
    print("Testing RegionFeatureExtractorBatch...")
    print("=" * 60)
    
    extractor_v2 = RegionFeatureExtractorBatch(
        feature_channels=256,
        output_size=7,
        output_type='spatial'
    )
    
    # 准备 bboxes [B, N, 4]
    bboxes = torch.tensor([
        [[0.1, 0.1, 0.3, 0.3], [0.5, 0.4, 0.6, 0.5]],
        [[0.2, 0.2, 0.4, 0.4], [0.7, 0.6, 0.9, 0.8]]
    ], dtype=torch.float32)
    
    feat_map = torch.randn(B, 256, H // 8, W // 8)
    
    region_features, area_ratios = extractor_v2(feat_map, bboxes, (H, W))
    
    print(f"\nRegion features shape: {region_features.shape}")  # [B, N, C, 7, 7]
    print(f"Area ratios shape: {area_ratios.shape}")
    print(f"Area ratios: {area_ratios}")
    
    # 测试 pooled 模式
    extractor_pooled = RegionFeatureExtractorBatch(
        feature_channels=256,
        output_size=7,
        output_type='pooled'
    )
    
    region_features_pooled, _ = extractor_pooled(feat_map, bboxes, (H, W))
    print(f"Pooled features shape: {region_features_pooled.shape}")  # [B, N, C]
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_region_feature_extractor()
