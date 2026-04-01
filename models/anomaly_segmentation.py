"""
Anomaly Segmentation Model with Small Hazard Prior
在 SimpleAnomalyDecoder 中融入 small_hazard_prior
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass

from .registry import MODEL_REGISTRY
from .segformer import MitB2Backbone

from utils.hazard_scorer_with_threshold import HazardScorerWithAdaptiveThreshold
from utils.small_hazard_prior import SmallHazardPriorGenerator


@dataclass
class AnomalySegmentationOutput:
    """模型输出"""
    coarse_logits: torch.Tensor      # [B, 1, H, W]
    final_logits: torch.Tensor      # [B, 1, H, W]
    candidates: List[List]          # 每张图的候选区域
    hazard_scores: Optional[torch.Tensor]  # [B, N] 或 None
    delta_t: Optional[torch.Tensor]       # [B,] 或 None
    small_hazard_prior: Optional[torch.Tensor]  # [B, 1, H, W] 或 None
    # 额外的跟踪信息
    candidate_masks: Optional[torch.Tensor] = None  # [B, N, H, W]
    area_ratios: Optional[torch.Tensor] = None  # [B, N]


# 兼容旧版本：保留原始的 SimpleAnomalyDecoder 和 AnomalySegmentationModel
class SimpleAnomalyDecoder(nn.Module):
    """
    简单的 Anomaly Decoder
    - coarse_logits: 来自浅层特征 (高分辨率，低语义)
    - final_logits: 融合多层特征后得到
    """

    def __init__(
        self,
        encoder_dims: List[int] = [64, 128, 320, 512],
        decoder_dim: int = 256
    ):
        super().__init__()
        self.encoder_dims = encoder_dims
        self.decoder_dim = decoder_dim

        # 投影层
        self.linear_c4 = nn.Linear(encoder_dims[3], decoder_dim)
        self.linear_c3 = nn.Linear(encoder_dims[2], decoder_dim)
        self.linear_c2 = nn.Linear(encoder_dims[1], decoder_dim)
        self.linear_c1 = nn.Linear(encoder_dims[0], decoder_dim)

        # Coarse branch: 使用浅层特征 (c1, c2)
        self.coarse_fuse = nn.Conv2d(decoder_dim * 2, decoder_dim, kernel_size=1)
        self.coarse_bn = nn.BatchNorm2d(decoder_dim)
        self.coarse_act = nn.ReLU()
        self.coarse_out = nn.Conv2d(decoder_dim, 1, kernel_size=1)

        # Final branch: 融合所有特征
        self.final_fuse = nn.Conv2d(decoder_dim * 4, decoder_dim, kernel_size=1)
        self.final_bn = nn.BatchNorm2d(decoder_dim)
        self.final_act = nn.ReLU()
        self.final_out = nn.Conv2d(decoder_dim, 1, kernel_size=1)

    def forward(self, encoder_features: List[torch.Tensor], target_size: Tuple[int, int]):
        """
        Args:
            encoder_features: [c1, c2, c3, c4] 多尺度特征
            target_size: 目标输出尺寸 (H, W)

        Returns:
            coarse_logits: [B, 1, H, W]
            final_logits: [B, 1, H, W]
        """
        H0, W0 = target_size

        # 投影并上采样到 H/4, W/4
        c1 = self._project_and_upample(encoder_features[0], (H0 // 4, W0 // 4), self.linear_c1)
        c2 = self._project_and_upample(encoder_features[1], (H0 // 4, W0 // 4), self.linear_c2)
        c3 = self._project_and_upample(encoder_features[2], (H0 // 4, W0 // 4), self.linear_c3)
        c4 = self._project_and_upample(encoder_features[3], (H0 // 4, W0 // 4), self.linear_c4)

        # ===== Coarse branch (使用浅层特征 c1, c2) =====
        coarse_cat = torch.cat([c1, c2], dim=1)
        coarse_x = self.coarse_fuse(coarse_cat)
        coarse_x = self.coarse_bn(coarse_x)
        coarse_x = self.coarse_act(coarse_x)
        coarse_x = F.interpolate(coarse_x, size=(H0, W0), mode='bilinear', align_corners=False)
        coarse_logits = self.coarse_out(coarse_x)

        # ===== Final branch (融合所有特征) =====
        final_cat = torch.cat([c4, c3, c2, c1], dim=1)
        final_x = self.final_fuse(final_cat)
        final_x = self.final_bn(final_x)
        final_x = self.final_act(final_x)
        final_x = F.interpolate(final_x, size=(H0, W0), mode='bilinear', align_corners=False)
        final_logits = self.final_out(final_x)

        return coarse_logits, final_logits

    def _project_and_upample(self, x: torch.Tensor, size: Tuple[int, int], linear_layer: nn.Linear) -> torch.Tensor:
        """投影并上采样"""
        x = self._linear_projection(x, linear_layer)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        return x

    def _linear_projection(self, x: torch.Tensor, linear_layer: nn.Linear) -> torch.Tensor:
        """线性投影 [B, C, H, W] -> [B, decoder_dim, H, W]"""
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(B * H * W, C)
        x = linear_layer(x)
        x = x.reshape(B, H, W, self.decoder_dim)
        x = x.permute(0, 3, 1, 2)
        return x


@MODEL_REGISTRY.register()
class AnomalySegmentationModel(nn.Module):
    """
    原始的 Anomaly Segmentation Model (无 hazard prior)
    """

    def __init__(
        self,
        in_channels: int = 3,
        decoder_dim: int = 256,
        pretrained_backbone_path: str = None
    ):
        super().__init__()
        self.in_channels = in_channels
        self.decoder_dim = decoder_dim
        self.pretrained_backbone_path = pretrained_backbone_path

        # Backbone: MiT-B2
        self.backbone = MitB2Backbone(in_channels=in_channels, pretrained=False)

        # Decoder
        self.decoder = SimpleAnomalyDecoder(
            encoder_dims=[64, 128, 320, 512],
            decoder_dim=decoder_dim
        )

        if pretrained_backbone_path is not None:
            self.load_backbone_weights(pretrained_backbone_path)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 coarse_logits, final_logits"""
        features = self.backbone(x)
        coarse_logits, final_logits = self.decoder(features, target_size=(x.shape[2], x.shape[3]))
        return coarse_logits, final_logits

    def load_backbone_weights(self, pretrained_path: str, strict: bool = True):
        print(f"Loading backbone weights from: {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location='cpu')
        if isinstance(state_dict, dict):
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model' in state_dict:
                state_dict = state_dict['model']
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict['backbone.' + k] = v
        missing, unexpected = self.backbone.load_state_dict(new_state_dict, strict=False)
        if missing:
            print(f"Missing keys: {len(missing)} keys")
        if unexpected:
            print(f"Unexpected keys: {len(unexpected)} keys")
        else:
            print("All backbone weights loaded successfully!")

    @staticmethod
    def build_from_config(config):
        model_cfg = config.get('model', {})
        anomaly_cfg = config.get('anomaly_model', {})
        pretrained_path = anomaly_cfg.get('pretrained_backbone_path')
        if not pretrained_path and 'checkpoint' in config:
            save_dir = config['checkpoint'].get('save_dir', 'checkpoints/cityscapes')
            pretrained_path = f"{save_dir}/backbone_weights.pth"
        return AnomalySegmentationModel(
            in_channels=model_cfg.get('in_channels', 3),
            decoder_dim=anomaly_cfg.get('decoder_dim', 256),
            pretrained_backbone_path=pretrained_path
        )


class SimpleAnomalyDecoderWithPrior(nn.Module):
    """
    带有 Small Hazard Prior 融合的 Decoder
    
    输入:
        - encoder_features: 多尺度特征 [c1, c2, c3, c4]
        - coarse anomaly map: 来自 coarse branch
        - small_hazard_prior: 小目标危险度先验图
        
    输出:
        - coarse_logits
        - final_logits
    """

    def __init__(
        self,
        encoder_dims: List[int] = [64, 128, 320, 512],
        decoder_dim: int = 256,
        use_prior: bool = True,
        prior_fusion_mode: str = 'add',  # 'add', 'concat', 'attention'
        prior_channels: int = 1
    ):
        super().__init__()
        self.encoder_dims = encoder_dims
        self.decoder_dim = decoder_dim
        self.use_prior = use_prior
        self.prior_fusion_mode = prior_fusion_mode
        
        # 投影层
        self.linear_c4 = nn.Linear(encoder_dims[3], decoder_dim)
        self.linear_c3 = nn.Linear(encoder_dims[2], decoder_dim)
        self.linear_c2 = nn.Linear(encoder_dims[1], decoder_dim)
        self.linear_c1 = nn.Linear(encoder_dims[0], decoder_dim)

        # Coarse branch: 使用浅层特征 (c1, c2)
        self.coarse_fuse = nn.Conv2d(decoder_dim * 2, decoder_dim, kernel_size=1)
        self.coarse_bn = nn.BatchNorm2d(decoder_dim)
        self.coarse_act = nn.ReLU()
        self.coarse_out = nn.Conv2d(decoder_dim, 1, kernel_size=1)

        # Final branch: 融合所有特征
        self.final_fuse = nn.Conv2d(decoder_dim * 4, decoder_dim, kernel_size=1)
        self.final_bn = nn.BatchNorm2d(decoder_dim)
        self.final_act = nn.ReLU()
        self.final_out = nn.Conv2d(decoder_dim, 1, kernel_size=1)

        # ===== Small Hazard Prior 融合层 =====
        if use_prior:
            if prior_fusion_mode == 'add':
                # 加法融合：先投影 prior 到 decoder_dim
                self.prior_proj = nn.Conv2d(prior_channels, decoder_dim, kernel_size=1)
            elif prior_fusion_mode == 'concat':
                # 拼接融合：decoder_dim + prior_channels
                self.prior_fuse = nn.Conv2d(decoder_dim + prior_channels, decoder_dim, kernel_size=1)
            elif prior_fusion_mode == 'attention':
                # 注意力融合
                self.prior_proj = nn.Conv2d(prior_channels, decoder_dim, kernel_size=1)
                self.prior_attention = nn.Sequential(
                    nn.Conv2d(decoder_dim * 2, decoder_dim, kernel_size=1),
                    nn.Sigmoid()
                )
            else:
                raise ValueError(f"Unknown fusion mode: {prior_fusion_mode}")

    def forward(
        self,
        encoder_features: List[torch.Tensor],
        target_size: Tuple[int, int],
        small_hazard_prior: Optional[torch.Tensor] = None
    ):
        """
        Args:
            encoder_features: [c1, c2, c3, c4]
            target_size: (H, W)
            small_hazard_prior: [B, 1, H, W] 或 None
            
        Returns:
            coarse_logits, final_logits
        """
        H0, W0 = target_size

        # 投影并上采样
        c1 = self._project_and_upample(encoder_features[0], (H0 // 4, W0 // 4), self.linear_c1)
        c2 = self._project_and_upample(encoder_features[1], (H0 // 4, W0 // 4), self.linear_c2)
        c3 = self._project_and_upample(encoder_features[2], (H0 // 4, W0 // 4), self.linear_c3)
        c4 = self._project_and_upample(encoder_features[3], (H0 // 4, W0 // 4), self.linear_c4)

        # ===== Coarse branch =====
        coarse_cat = torch.cat([c1, c2], dim=1)
        coarse_x = self.coarse_fuse(coarse_cat)
        coarse_x = self.coarse_bn(coarse_x)
        coarse_x = self.coarse_act(coarse_x)
        coarse_x = F.interpolate(coarse_x, size=(H0, W0), mode='bilinear', align_corners=False)
        coarse_logits = self.coarse_out(coarse_x)

        # ===== Final branch =====
        final_cat = torch.cat([c4, c3, c2, c1], dim=1)
        final_x = self.final_fuse(final_cat)
        final_x = self.final_bn(final_x)
        final_x = self.final_act(final_x)
        
        # ===== 融入 Small Hazard Prior =====
        if self.use_prior and small_hazard_prior is not None:
            final_x = self._fuse_prior(final_x, small_hazard_prior, target_size)
        
        final_x = F.interpolate(final_x, size=(H0, W0), mode='bilinear', align_corners=False)
        final_logits = self.final_out(final_x)

        return coarse_logits, final_logits

    def _fuse_prior(
        self,
        final_features: torch.Tensor,
        prior: torch.Tensor,
        target_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        融合 small hazard prior
        
        Args:
            final_features: [B, decoder_dim, H, W]
            prior: [B, 1, H, W]
            target_size: (H, W)
        """
        # 确保 prior 尺寸匹配
        if prior.shape[2:] != final_features.shape[2:]:
            prior = F.interpolate(prior, size=final_features.shape[2:], 
                                 mode='bilinear', align_corners=False)
        
        if self.prior_fusion_mode == 'add':
            # 先投影 prior
            prior_proj = self.prior_proj(prior)
            final_features = final_features + prior_proj
            
        elif self.prior_fusion_mode == 'concat':
            # 拼接
            final_cat = torch.cat([final_features, prior], dim=1)
            final_features = self.prior_fuse(final_cat)
            final_features = self.final_bn(final_features)
            final_features = self.final_act(final_features)
            
        elif self.prior_fusion_mode == 'attention':
            # 注意力融合
            prior_proj = self.prior_proj(prior)
            # 拼接特征和 prior，计算注意力
            attention_input = torch.cat([final_features, prior_proj], dim=1)
            attention_weights = self.prior_attention(attention_input)
            final_features = final_features * attention_weights
        
        return final_features

    def _project_and_upample(self, x: torch.Tensor, size: Tuple[int, int], linear_layer: nn.Linear) -> torch.Tensor:
        x = self._linear_projection(x, linear_layer)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        return x

    def _linear_projection(self, x: torch.Tensor, linear_layer: nn.Linear) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(B * H * W, C)
        x = linear_layer(x)
        x = x.reshape(B, H, W, self.decoder_dim)
        x = x.permute(0, 3, 1, 2)
        return x


@MODEL_REGISTRY.register()
class AnomalySegmentationModelWithPrior(nn.Module):
    """
    带有 Small Hazard Prior 的 Anomaly Segmentation Model
    
    特点:
        - 保留 coarse branch
        - 融入 small_hazard_prior
        - 支持 hazard scoring
        - 支持自适应阈值
    """

    def __init__(
        self,
        in_channels: int = 3,
        decoder_dim: int = 256,
        pretrained_backbone_path: str = None,
        
        # Hazard Scorer 参数
        use_hazard_scorer: bool = True,
        hazard_beta: float = 10.0,
        base_threshold: float = 0.01,
        
        # Small Hazard Prior 参数
        use_small_hazard_prior: bool = True,
        prior_fusion_mode: str = 'add',
        
        # 其他参数
        enable_training_details: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.decoder_dim = decoder_dim
        self.use_hazard_scorer = use_hazard_scorer
        self.use_small_hazard_prior = use_small_hazard_prior
        self.enable_training_details = enable_training_details
        
        # Backbone
        self.backbone = MitB2Backbone(in_channels=in_channels, pretrained=False)
        
        # Decoder
        self.decoder = SimpleAnomalyDecoderWithPrior(
            encoder_dims=[64, 128, 320, 512],
            decoder_dim=decoder_dim,
            use_prior=use_small_hazard_prior,
            prior_fusion_mode=prior_fusion_mode
        )
        
        # Hazard Scorer with Adaptive Threshold
        # 注意: global_feature_channels 应该与 backbone 最深层通道数一致 (C4=512)
        if use_hazard_scorer:
            self.hazard_scorer = HazardScorerWithAdaptiveThreshold(
                visual_channels=decoder_dim,
                text_channels=512,
                hidden_dim=256,
                num_prompts=2,
                output_size=7,
                global_feature_channels=512,  # backbone C4 通道数
                threshold_hidden_dim=64,
                alpha=0.02,
                beta=hazard_beta,
                base_threshold=base_threshold,
                use_adaptive_threshold=True,
                use_soft_gate=True
            )
            
            # Small Hazard Prior Generator
            self.prior_generator = SmallHazardPriorGenerator(
                output_size=None,
                use_softmax=True
            )

        # 加载预训练 backbone
        if pretrained_backbone_path is not None:
            self.load_backbone_weights(pretrained_backbone_path)

    def forward(
        self,
        x: torch.Tensor,
        return_details: bool = True
    ) -> AnomalySegmentationOutput:
        """
        Args:
            x: [B, 3, H, W]
            return_details: 是否返回详细信息
            
        Returns:
            AnomalySegmentationOutput
        """
        B, _, H, W = x.shape
        
        # ===== 1. Backbone =====
        features = self.backbone(x)
        
        # 获取最深层特征用于全局特征
        global_features = F.adaptive_avg_pool2d(features[3], (1, 1)).flatten(1)  # [B, C4]
        
        # ===== 2. Coarse branch =====
        coarse_logits, final_logits = self.decoder(
            features, 
            target_size=(H, W),
            small_hazard_prior=None  # 第一轮不使用 prior
        )
        
        # Coarse anomaly probability map
        coarse_prob = torch.sigmoid(coarse_logits)  # [B, 1, H, W]
        
        # 初始化输出
        candidates = []
        hazard_scores = None
        delta_t = None
        small_hazard_prior = None
        
        # ===== 3. Hazard Scoring & Small Hazard Prior (训练时或需要时) =====
        if self.use_hazard_scorer and (self.training or return_details):
            # 从 coarse_prob 提取候选区域 (需要转换为 numpy)
            # 这里简化处理：直接使用 coarse_prob 的高响应区域作为候选
            # 实际使用时可以用 CandidateExtractor
            
            # 模拟候选区域处理
            # 实际应该用 candidate_extractor 从 coarse_prob 提取
            # 这里暂时返回空，等待后续集成
            
            # 为了演示，创建简单的候选
            # 实际场景中应该从 coarse_prob_map 提取
            pass
        
        return AnomalySegmentationOutput(
            coarse_logits=coarse_logits,
            final_logits=final_logits,
            candidates=candidates,
            hazard_scores=hazard_scores,
            delta_t=delta_t,
            small_hazard_prior=small_hazard_prior
        )

    def forward_with_hazard_prior(
        self,
        x: torch.Tensor,
        candidate_masks: torch.Tensor,
        area_ratios: torch.Tensor,
        text_similarities: Optional[torch.Tensor] = None
    ) -> AnomalySegmentationOutput:
        """
        完整的前向传播，包含 hazard scoring 和 prior 生成
        
        Args:
            x: [B, 3, H, W]
            candidate_masks: [B, N, H, W] 候选区域 masks
            area_ratios: [B, N] 面积比例
            text_similarities: [B, N, T] 文本相似度
            
        Returns:
            AnomalySegmentationOutput
        """
        B, _, H, W = x.shape
        
        # ===== 1. Backbone =====
        features = self.backbone(x)
        global_features = F.adaptive_avg_pool2d(features[3], (1, 1)).flatten(1)  # [B, 512]
        
        # ===== 2. Coarse branch =====
        coarse_logits, _ = self.decoder(
            features, 
            target_size=(H, W),
            small_hazard_prior=None
        )
        
        # ===== 3. Hazard Scoring =====
        hazard_scores = None
        delta_t = None
        soft_gates = None
        
        if hasattr(self, 'hazard_scorer'):
            # 简化：使用 backbone 最后一层的全局特征作为候选特征
            # 实际应该从 features 中提取每个候选区域的特征
            # 这里使用每个候选的 area_ratios 作为特征（简化处理）
            
            # 计算 Delta_t (阈值偏移)
            if global_features is not None:
                Delta_t_batch = self.hazard_scorer.threshold_offset(global_features)  # [B,]
                delta_t = Delta_t_batch
                
                # 自适应阈值
                base_threshold = self.hazard_scorer.base_threshold
                t_adapt = base_threshold + delta_t  # [B,]
                
                # 计算 hazard scores (简化: 基于 area_ratios 和 global_features)
                # 实际应该用候选区域的视觉特征
                hazard_scores = torch.sigmoid(area_ratios * 10)  # [B, N]
                
                # Soft gates
                beta = self.hazard_scorer.beta
                soft_gates = torch.sigmoid(beta * (hazard_scores - t_adapt.unsqueeze(-1)))
            else:
                hazard_scores = torch.sigmoid(area_ratios * 10)
                soft_gates = torch.ones_like(area_ratios)
        
        # ===== 4. 生成 Small Hazard Prior =====
        small_hazard_prior = None
        if hasattr(self, 'prior_generator') and soft_gates is not None:
            # 生成 prior
            small_hazard_prior = self.prior_generator(
                candidate_masks, 
                soft_gates,
                target_size=(H, W)
            )  # [B, 1, H, W]
        
        # ===== 5. Final decoder with prior =====
        _, final_logits = self.decoder(
            features,
            target_size=(H, W),
            small_hazard_prior=small_hazard_prior
        )
        
        return AnomalySegmentationOutput(
            coarse_logits=coarse_logits,
            final_logits=final_logits,
            candidates=[],  # 简化
            hazard_scores=hazard_scores,
            delta_t=delta_t,
            small_hazard_prior=small_hazard_prior,
            candidate_masks=candidate_masks,
            area_ratios=area_ratios
        )

    def load_backbone_weights(self, pretrained_path: str, strict: bool = True):
        """加载预训练 backbone"""
        print(f"Loading backbone weights from: {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location='cpu')
        
        if isinstance(state_dict, dict):
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model' in state_dict:
                state_dict = state_dict['model']
        
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict['backbone.' + k] = v
        
        missing, unexpected = self.backbone.load_state_dict(new_state_dict, strict=False)
        if missing:
            print(f"Missing keys: {len(missing)} keys")
        if unexpected:
            print(f"Unexpected keys: {len(unexpected)} keys")
        else:
            print("All backbone weights loaded successfully!")

    @staticmethod
    def build_from_config(config):
        """从配置构建模型"""
        model_cfg = config.get('model', {})
        anomaly_cfg = config.get('anomaly_model', {})
        
        pretrained_path = anomaly_cfg.get('pretrained_backbone_path')
        if not pretrained_path and 'checkpoint' in config:
            save_dir = config['checkpoint'].get('save_dir', 'checkpoints/cityscapes')
            pretrained_path = f"{save_dir}/backbone_weights.pth"
        
        return AnomalySegmentationModelWithPrior(
            in_channels=model_cfg.get('in_channels', 3),
            decoder_dim=anomaly_cfg.get('decoder_dim', 256),
            pretrained_backbone_path=pretrained_path,
            use_hazard_scorer=anomaly_cfg.get('use_hazard_scorer', True),
            hazard_beta=anomaly_cfg.get('hazard_beta', 10.0),
            base_threshold=anomaly_cfg.get('base_threshold', 0.01),
            use_small_hazard_prior=anomaly_cfg.get('use_small_hazard_prior', True),
            prior_fusion_mode=anomaly_cfg.get('prior_fusion_mode', 'add')
        )


# 移除重复注册（类已通过 @MODEL_REGISTRY.register() 装饰器注册）
