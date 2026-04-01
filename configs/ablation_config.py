"""
Ablation Configuration Module
SHIELD-Lite 各组件的开关配置
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import yaml


@dataclass
class AblationConfig:
    """
    Ablation Study Configuration
    
    控制 SHIELD-Lite 各组件的启用/禁用:
    1. small_candidate_extraction: 小候选区域提取
    2. text_guided_hazard: 文本引导的危险评分
    3. adaptive_threshold: 自适应阈值偏移 Delta_t
    4. small_hazard_prior: 小目标先验 refinement
    """
    
    # ===== Core Components =====
    
    # 小候选区域提取 (从 coarse map 中提取)
    enable_small_candidate_extraction: bool = True
    
    # 文本引导的危险评分 (使用 CLIP 文本嵌入)
    enable_text_guided_hazard: bool = False
    
    # 自适应阈值偏移 Delta_t
    enable_adaptive_threshold: bool = True
    
    # 小目标先验 refinement
    enable_small_hazard_prior: bool = True
    
    # ===== Prior Fusion =====
    prior_fusion_mode: str = "add"  # "add", "concat", "attention"
    
    # ===== Hazard Scoring =====
    hazard_score_type: str = "area"  # "area", "visual", "text"
    
    # ===== Loss Weights =====
    lambda_coarse: float = 0.5
    lambda_hazard: float = 0.5
    
    # ===== Training =====
    use_hard_example_mining: bool = False
    
    def __post_init__(self):
        """验证配置合法性"""
        valid_fusion_modes = ["add", "concat", "attention", "multiply"]
        if self.prior_fusion_mode not in valid_fusion_modes:
            raise ValueError(f"Invalid prior_fusion_mode: {self.prior_fusion_mode}, must be one of {valid_fusion_modes}")
        
        valid_hazard_types = ["area", "visual", "text", "hybrid"]
        if self.hazard_score_type not in valid_hazard_types:
            raise ValueError(f"Invalid hazard_score_type: {self.hazard_score_type}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AblationConfig':
        """从字典创建配置"""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'AblationConfig':
        """从 YAML 文件加载配置"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {k: getattr(self, k) for k in self.__dataclass_fields__}
    
    def get_description(self) -> str:
        """获取配置描述"""
        enabled = []
        disabled = []
        
        if self.enable_small_candidate_extraction:
            enabled.append("SmallCandidate")
        else:
            disabled.append("SmallCandidate")
            
        if self.enable_text_guided_hazard:
            enabled.append("TextGuided")
        else:
            disabled.append("TextGuided")
            
        if self.enable_adaptive_threshold:
            enabled.append("AdaptiveThresh")
        else:
            disabled.append("AdaptiveThresh")
            
        if self.enable_small_hazard_prior:
            enabled.append(f"Prior({self.prior_fusion_mode})")
        else:
            disabled.append("Prior")
        
        desc = "+".join(enabled) if enabled else "Baseline"
        if disabled:
            desc += f" (-{'-'.join(disabled)})"
        return desc


# ===== Predefined Ablation Configs =====

ABLATION_CONFIGS = {
    # 1. Baseline: 仅有 coarse + final branch
    "baseline": AblationConfig(
        enable_small_candidate_extraction=False,
        enable_text_guided_hazard=False,
        enable_adaptive_threshold=False,
        enable_small_hazard_prior=False,
        lambda_coarse=0.0,
        lambda_hazard=0.0
    ),
    
    # 2. Only Coarse Branch
    "coarse_only": AblationConfig(
        enable_small_candidate_extraction=False,
        enable_text_guided_hazard=False,
        enable_adaptive_threshold=False,
        enable_small_hazard_prior=False,
        lambda_coarse=0.5,
        lambda_hazard=0.0
    ),
    
    # 3. + Small Candidate Extraction
    "with_candidates": AblationConfig(
        enable_small_candidate_extraction=True,
        enable_text_guided_hazard=False,
        enable_adaptive_threshold=False,
        enable_small_hazard_prior=False,
        lambda_coarse=0.5,
        lambda_hazard=0.0
    ),
    
    # 4. + Hazard Scoring (无文本)
    "with_hazard": AblationConfig(
        enable_small_candidate_extraction=True,
        enable_text_guided_hazard=False,
        enable_adaptive_threshold=False,
        enable_small_hazard_prior=False,
        hazard_score_type="area",
        lambda_coarse=0.5,
        lambda_hazard=0.3
    ),
    
    # 5. + Adaptive Threshold
    "with_adaptive_threshold": AblationConfig(
        enable_small_candidate_extraction=True,
        enable_text_guided_hazard=False,
        enable_adaptive_threshold=True,
        enable_small_hazard_prior=False,
        hazard_score_type="area",
        lambda_coarse=0.5,
        lambda_hazard=0.3
    ),
    
    # 6. + Small Hazard Prior (add fusion)
    "with_prior_add": AblationConfig(
        enable_small_candidate_extraction=True,
        enable_text_guided_hazard=False,
        enable_adaptive_threshold=True,
        enable_small_hazard_prior=True,
        prior_fusion_mode="add",
        hazard_score_type="area",
        lambda_coarse=0.5,
        lambda_hazard=0.5
    ),
    
    # 7. + Small Hazard Prior (attention fusion)
    "with_prior_attention": AblationConfig(
        enable_small_candidate_extraction=True,
        enable_text_guided_hazard=False,
        enable_adaptive_threshold=True,
        enable_small_hazard_prior=True,
        prior_fusion_mode="attention",
        hazard_score_type="area",
        lambda_coarse=0.5,
        lambda_hazard=0.5
    ),
    
    # 8. Full SHIELD-Lite (with text)
    "full_shield_lite": AblationConfig(
        enable_small_candidate_extraction=True,
        enable_text_guided_hazard=True,
        enable_adaptive_threshold=True,
        enable_small_hazard_prior=True,
        prior_fusion_mode="add",
        hazard_score_type="hybrid",
        lambda_coarse=0.5,
        lambda_hazard=0.5
    ),
    
    # 9. Full with concat fusion
    "full_concat": AblationConfig(
        enable_small_candidate_extraction=True,
        enable_text_guided_hazard=True,
        enable_adaptive_threshold=True,
        enable_small_hazard_prior=True,
        prior_fusion_mode="concat",
        hazard_score_type="hybrid",
        lambda_coarse=0.5,
        lambda_hazard=0.5
    ),
}


def get_ablation_config(name: str) -> AblationConfig:
    """获取预定义的 ablation 配置"""
    if name not in ABLATION_CONFIGS:
        raise ValueError(f"Unknown ablation config: {name}. Available: {list(ABLATION_CONFIGS.keys())}")
    return ABLATION_CONFIGS[name]


def list_ablation_configs() -> List[str]:
    """列出所有预定义的 ablation 配置"""
    return list(ABLATION_CONFIGS.keys())


# ===== YAML Templates =====

ABLATION_YAML_TEMPLATE = """# =============================================================================
# SHIELD-Lite Ablation Study Configuration Template
# =============================================================================
# 使用方式:
#   1. 选择预设: 设置 _ablation_name 为预设名称
#   2. 或手动配置: 设置各组件开关
# =============================================================================

# -----------------------------------------------------------------------------
# 预设配置 (选择其中一个)
# -----------------------------------------------------------------------------
_ablation_name: "full_shield_lite"  # 可选: baseline, coarse_only, with_candidates, with_hazard, 
                                     #       with_adaptive_threshold, with_prior_add, with_prior_attention,
                                     #       full_shield_lite, full_concat

# -----------------------------------------------------------------------------
# 手动配置 (覆盖预设)
# -----------------------------------------------------------------------------
ablation:
  # 小候选区域提取 (从 coarse map 中提取小目标候选)
  enable_small_candidate_extraction: true
  
  # 文本引导的危险评分 (使用 CLIP 文本嵌入)
  enable_text_guided_hazard: false
  
  # 自适应阈值偏移 Delta_t
  enable_adaptive_threshold: true
  
  # 小目标先验 refinement
  enable_small_hazard_prior: true
  
  # Prior 融合方式
  prior_fusion_mode: "add"  # add, concat, attention, multiply
  
  # Hazard score 计算方式
  hazard_score_type: "area"  # area, visual, text, hybrid

# -----------------------------------------------------------------------------
# Loss 权重
# -----------------------------------------------------------------------------
loss:
  lambda_coarse: 0.5
  lambda_hazard: 0.5
  
  # BCE/Dice 权重
  bce_weight: 0.5
  dice_weight: 0.5

# -----------------------------------------------------------------------------
# 候选区域提取参数
# -----------------------------------------------------------------------------
candidate:
  threshold_high: 0.5
  threshold_small: 0.3
  tau_small: 500
  local_response_thresh: 0.6
  max_candidates: 10
  min_area: 50

# -----------------------------------------------------------------------------
# Hazard Scorer 参数
# -----------------------------------------------------------------------------
hazard_scorer:
  hazard_beta: 10.0
  base_threshold: 0.01
  use_soft_gate: true

# =============================================================================
# 详细配置说明
# =============================================================================
#
# 各组件说明:
#   1. enable_small_candidate_extraction:
#      - 从 coarse anomaly probability map 中提取小目标候选区域
#      - 用于后续 hazard scoring 和 prior 生成
#      - 需要 candidate extractor
#
#   2. enable_text_guided_hazard:
#      - 使用 CLIP 文本嵌入引导 hazard score 计算
#      - 需要 text embeddings (normal vs anomaly prompts)
#      - 设为 false 时使用 visual/area features
#
#   3. enable_adaptive_threshold:
#      - 启用 Delta_t (阈值偏移) 的学习
#      - 根据图像内容动态调整二值化阈值
#      - 需要 hazard scorer 模块
#
#   4. enable_small_hazard_prior:
#      - 基于 hazard scores 生成 small-hazard prior
#      - 融入 final decoder 提升小目标检测
#      - 需要 prior_generator 模块
#
#   5. prior_fusion_mode:
#      - add: 简单相加 (prior + features)
#      - concat: 通道拼接 (prior | features)
#      - attention: 注意力机制融合
#      - multiply: 逐元素相乘
#
#   6. hazard_score_type:
#      - area: 仅使用候选区域面积比例
#      - visual: 使用视觉特征 (推荐)
#      - text: 使用文本嵌入相似度
#      - visual + text: 混合使用
#
# =============================================================================
"""


def generate_ablation_configs_yaml():
    """生成所有 ablation 配置的 YAML 文件"""
    configs = {}
    
    for name, config in ABLATION_CONFIGS.items():
        configs[name] = config.to_dict()
    
    return yaml.dump(configs, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    # 测试
    print("Available ablation configs:")
    for name in list_ablation_configs():
        config = get_ablation_config(name)
        print(f"  {name}: {config.get_description()}")
    
    print("\n" + "="*60)
    print("Full config example:")
    print("="*60)
    config = get_ablation_config("full_shield_lite")
    print(f"Name: full_shield_lite")
    print(f"Description: {config.get_description()}")
    print(f"Config:")
    for k, v in config.to_dict().items():
        print(f"  {k}: {v}")
