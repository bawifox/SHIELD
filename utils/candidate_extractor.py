"""
Candidate Extractor Module
从 coarse anomaly probability map 中提取候选区域
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy import ndimage


@dataclass
class CandidateRegion:
    """候选区域数据结构"""
    mask: np.ndarray          # 二值 mask [H, W]
    bbox: Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)
    mean_score: float         # 平均响应分数
    max_score: float         # 最大响应分数
    area: int                # 面积 (像素数)
    area_ratio: float        # 面积占原图比例

    def to_dict(self) -> dict:
        return {
            'mask': self.mask,
            'bbox': self.bbox,
            'mean_score': self.mean_score,
            'max_score': self.max_score,
            'area': self.area,
            'area_ratio': self.area_ratio
        }


class CandidateExtractor:
    """
    从 coarse anomaly probability map 中提取候选区域

    Args:
        threshold_high: 高响应区域阈值
        threshold_small: 小区域阈值
        tau_small: 小区域面积阈值 (小于此值认为是小区域)
        local_response_thresh: 小区域局部响应阈值
        N_max: 最大候选数量
        min_area: 最小面积阈值
    """

    def __init__(
        self,
        threshold_high: float = 0.5,
        threshold_small: float = 0.3,
        tau_small: int = 500,
        local_response_thresh: float = 0.6,
        N_max: int = 10,
        min_area: int = 50
    ):
        self.threshold_high = threshold_high
        self.threshold_small = threshold_small
        self.tau_small = tau_small
        self.local_response_thresh = local_response_thresh
        self.N_max = N_max
        self.min_area = min_area

    def __call__(
        self,
        coarse_prob_map: np.ndarray,
        image_size: Optional[Tuple[int, int]] = None
    ) -> List[CandidateRegion]:
        """
        提取候选区域

        Args:
            coarse_prob_map: 粗粒度概率图 [H, W]，值范围 [0, 1]
            image_size: 原始图像尺寸 (H, W)，用于计算 area_ratio

        Returns:
            候选区域列表
        """
        if image_size is None:
            image_size = coarse_prob_map.shape

        H, W = coarse_prob_map.shape
        total_pixels = H * W

        # Step 1: 阈值化
        binary_map = self._threshold(coarse_prob_map)

        # Step 2: 连通域分析
        labeled_array, num_features = ndimage.label(binary_map)

        if num_features == 0:
            return []

        # Step 3: 对每个连通域提取特征
        candidates = []
        for region_id in range(1, num_features + 1):
            region_mask = (labeled_array == region_id)

            # 计算面积
            area = np.sum(region_mask)

            # 跳过太小或太大的区域
            if area < self.min_area:
                continue

            # 计算 bbox
            y_coords, x_coords = np.where(region_mask)
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()
            bbox = (int(x_min), int(y_min), int(x_max), int(y_max))

            # 计算分数
            region_scores = coarse_prob_map[region_mask]
            mean_score = float(np.mean(region_scores))
            max_score = float(np.max(region_scores))

            # 计算面积比例
            area_ratio = area / total_pixels

            # 分类筛选
            keep = self._should_keep(area, max_score, mean_score)

            if keep:
                candidates.append(CandidateRegion(
                    mask=region_mask.astype(np.uint8),
                    bbox=bbox,
                    mean_score=mean_score,
                    max_score=max_score,
                    area=int(area),
                    area_ratio=float(area_ratio)
                ))

        # Step 4: 按 max_score 排序，保留 top N_max
        candidates = self._select_top_n(candidates)

        return candidates

    def _threshold(self, prob_map: np.ndarray) -> np.ndarray:
        """
        双重阈值策略：
        - 高阈值区域直接保留
        - 低阈值中的小区域：如果局部响应强也保留
        """
        high_thresh = self.threshold_high
        low_thresh = self.threshold_small

        # 高阈值区域：直接保留
        high_mask = prob_map >= high_thresh

        # 低阈值区域：筛选小区域中的强响应
        low_mask = (prob_map >= low_thresh) & (prob_map < high_thresh)

        # 对低阈值区域进行连通域分析
        if np.any(low_mask):
            labeled_low, num_low = ndimage.label(low_mask)
            keep_mask = np.zeros_like(low_mask)

            for region_id in range(1, num_low + 1):
                region_mask = (labeled_low == region_id)
                region_area = np.sum(region_mask)

                # 小区域且局部响应强 -> 保留
                if region_area < self.tau_small:
                    region_max = np.max(prob_map[region_mask])
                    if region_max >= self.local_response_thresh:
                        keep_mask = keep_mask | region_mask

            low_mask = keep_mask

        # 合并
        binary_map = high_mask | low_mask
        return binary_map

    def _should_keep(self, area: int, max_score: float, mean_score: float) -> bool:
        """
        判断是否保留该区域

        保留条件：
        1. 高响应区域：max_score >= threshold_high
        2. 小区域：area < tau_small 且 max_score >= local_response_thresh
        """
        # 高响应区域
        if max_score >= self.threshold_high:
            return True

        # 小区域但局部响应强
        if area < self.tau_small and max_score >= self.local_response_thresh:
            return True

        return False

    def _select_top_n(self, candidates: List[CandidateRegion]) -> List[CandidateRegion]:
        """按 max_score 降序排序，保留前 N_max 个"""
        # 始终排序
        sorted_candidates = sorted(candidates, key=lambda x: x.max_score, reverse=True)

        if len(candidates) <= self.N_max:
            return sorted_candidates

        return sorted_candidates[:self.N_max]


class CandidateExtractorBatch:
    """
    批量处理多张图像的候选区域提取
    """

    def __init__(self, **kwargs):
        self.extractor = CandidateExtractor(**kwargs)

    def __call__(
        self,
        coarse_prob_maps: torch.Tensor,
        image_sizes: Optional[List[Tuple[int, int]]] = None
    ) -> List[List[CandidateRegion]]:
        """
        批量提取候选区域

        Args:
            coarse_prob_maps: [B, 1, H, W] 或 [B, H, W] 的 tensor
            image_sizes: 原始图像尺寸列表 [(H, W), ...]

        Returns:
            每张图像的候选区域列表
        """
        # 确保是 B, H, W 格式
        if coarse_prob_maps.dim() == 4:
            coarse_prob_maps = coarse_prob_maps.squeeze(1)
        elif coarse_prob_maps.dim() == 3:
            pass
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {coarse_prob_maps.dim()}D")

        B = coarse_prob_maps.shape[0]
        if image_sizes is None:
            image_sizes = [None] * B

        results = []
        for i in range(B):
            prob_map = coarse_prob_maps[i].cpu().numpy()
            candidates = self.extractor(prob_map, image_sizes[i])
            results.append(candidates)

        return results


def extract_candidates_numpy(
    prob_map: np.ndarray,
    threshold_high: float = 0.5,
    threshold_small: float = 0.3,
    tau_small: int = 500,
    local_response_thresh: float = 0.6,
    N_max: int = 10,
    min_area: int = 50
) -> List[CandidateRegion]:
    """
    便捷函数：从 numpy array 提取候选区域
    """
    extractor = CandidateExtractor(
        threshold_high=threshold_high,
        threshold_small=threshold_small,
        tau_small=tau_small,
        local_response_thresh=local_response_thresh,
        N_max=N_max,
        min_area=min_area
    )
    return extractor(prob_map)


def extract_candidates_tensor(
    prob_map: torch.Tensor,
    **kwargs
) -> List[CandidateRegion]:
    """
    便捷函数：从 torch tensor 提取候选区域
    """
    if prob_map.dim() == 4:
        prob_map = prob_map.squeeze(1)
    prob_np = prob_map.cpu().numpy()
    return extract_candidates_numpy(prob_np, **kwargs)


# ============ 单元测试 ============

def test_candidate_extractor():
    """测试 CandidateExtractor"""
    print("Running unit tests...")

    # 创建测试数据
    np.random.seed(42)

    # 测试1: 空图
    prob_map = np.zeros((256, 256))
    extractor = CandidateExtractor()
    candidates = extractor(prob_map)
    assert len(candidates) == 0, "Empty map should have no candidates"
    print("Test 1 (empty map): PASSED")

    # 测试2: 单个高响应区域
    prob_map = np.zeros((256, 256))
    prob_map[50:150, 50:150] = 0.8  # 高响应区域
    candidates = extractor(prob_map)
    assert len(candidates) == 1, f"Expected 1 candidate, got {len(candidates)}"
    assert candidates[0].area == 10000, f"Expected area 10000, got {candidates[0].area}"
    assert abs(candidates[0].mean_score - 0.8) < 0.01, f"Expected mean 0.8, got {candidates[0].mean_score}"
    print("Test 2 (single high response): PASSED")

    # 测试3: 多个区域，选择 top N
    prob_map = np.zeros((256, 256))
    # 区域1: 大高响应
    prob_map[20:80, 20:80] = 0.7
    # 区域2: 小高响应
    prob_map[150:160, 150:160] = 0.9
    # 区域3: 中等响应
    prob_map[100:130, 100:130] = 0.4

    extractor_small = CandidateExtractor(N_max=2)
    candidates = extractor_small(prob_map)
    assert len(candidates) == 2, f"Expected 2 candidates, got {len(candidates)}"
    assert candidates[0].max_score >= candidates[1].max_score, "Should be sorted by max_score"
    print("Test 3 (top N selection): PASSED")

    # 测试4: 小区域筛选
    prob_map = np.zeros((256, 256))
    # 小区域但高响应
    prob_map[10:15, 10:15] = 0.7  # 25 pixels, 满足 local_response_thresh
    # 小区域但低响应
    prob_map[20:25, 20:25] = 0.4

    extractor_small = CandidateExtractor(
        tau_small=100,
        local_response_thresh=0.6,
        min_area=10
    )
    candidates = extractor_small(prob_map)
    assert len(candidates) == 1, f"Expected 1 candidate, got {len(candidates)}"
    print("Test 4 (small region filtering): PASSED")

    # 测试5: Batch 处理
    prob_maps = torch.randn(4, 1, 256, 256)
    prob_maps = torch.sigmoid(prob_maps)

    batch_extractor = CandidateExtractorBatch(N_max=5)
    results = batch_extractor(prob_maps)
    assert len(results) == 4, f"Expected 4 results, got {len(results)}"
    print("Test 5 (batch processing): PASSED")

    # 测试6: 边界情况 - 小区域面积
    prob_map = np.zeros((256, 256))
    prob_map[100:101, 100:101] = 0.9  # 1 pixel

    extractor_big_min = CandidateExtractor(min_area=1)
    candidates = extractor_big_min(prob_map)
    assert len(candidates) == 1, f"Expected 1 candidate, got {len(candidates)}"
    print("Test 6 (min area): PASSED")

    # 测试7: 候选区域属性
    prob_map = np.zeros((256, 256))
    prob_map[50:150, 50:150] = 0.8

    candidates = extractor(prob_map, image_size=(512, 512))
    c = candidates[0]
    assert c.area == 10000
    assert abs(c.area_ratio - 10000 / (256 * 256)) < 0.01
    assert c.bbox == (50, 50, 149, 149)
    print("Test 7 (candidate attributes): PASSED")

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_candidate_extractor()
