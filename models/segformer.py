"""
SegFormer 模型实现
包含 MiT (Mix Transformer) Backbone 和 MLP Decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional

from .registry import MODEL_REGISTRY
from .base import BaseSegmentationModel


class OverlapPatchEmbed(nn.Module):
    """
    Overlapping Patch Embedding
    使用重叠的滑动窗口进行 patch 嵌入
    """

    def __init__(self, patch_size: int = 7, stride: int = 4, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class TransformerBlock(nn.Module):
    """
    Efficient Transformer Block
    包含 Multi-Head Self-Attention 和 MixFFN
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=attn_drop, bias=qkv_bias, batch_first=True
        )
        self.drop_path1 = nn.Identity() if drop_path <= 0. else nn.Dropout(drop_path)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )
        self.drop_path2 = nn.Identity() if drop_path <= 0. else nn.Dropout(drop_path)

    def forward(self, x, H, W):
        # Multi-Head Self Attention
        x = x + self.drop_path1(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])

        # MixFFN
        x = x + self.drop_path2(self.mlp(self.norm2(x)))

        return x


class MiT(nn.Module):
    """
    Mix Transformer (MiT) Backbone
    多个阶段的 Transformer 编码器
    """

    def __init__(
        self,
        in_chans: int = 3,
        embed_dims: List[int] = [64, 128, 256, 512],
        num_heads: List[int] = [1, 2, 4, 8],
        mlp_ratios: List[float] = [4, 4, 4, 4],
        qkv_bias: bool = True,
        depths: List[int] = [3, 4, 6, 3],
        sr_ratios: List[int] = [8, 4, 2, 1],
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1
    ):
        super().__init__()
        self.depths = depths

        # Patch Embedding layers
        self.patch_embed1 = OverlapPatchEmbed(
            patch_size=7, stride=4, in_chans=in_chans, embed_dim=embed_dims[0]
        )
        self.patch_embed2 = OverlapPatchEmbed(
            patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1]
        )
        self.patch_embed3 = OverlapPatchEmbed(
            patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2]
        )
        self.patch_embed4 = OverlapPatchEmbed(
            patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3]
        )

        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList([
            TransformerBlock(
                dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i]
            ) for i in range(depths[0])
        ])
        self.norm1 = nn.LayerNorm(embed_dims[0])

        self.block2 = nn.ModuleList([
            TransformerBlock(
                dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1],
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[depths[0] + i]
            ) for i in range(depths[1])
        ])
        self.norm2 = nn.LayerNorm(embed_dims[1])

        self.block3 = nn.ModuleList([
            TransformerBlock(
                dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[depths[0] + depths[1] + i]
            ) for i in range(depths[2])
        ])
        self.norm3 = nn.LayerNorm(embed_dims[2])

        self.block4 = nn.ModuleList([
            TransformerBlock(
                dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3],
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:3]) + i]
            ) for i in range(depths[3])
        ])
        self.norm4 = nn.LayerNorm(embed_dims[3])

    def forward(self, x):
        """返回多尺度特征 (B, C, H, W)"""
        B = x.shape[0]
        outs = []

        # Stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # Stage 2
        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # Stage 3
        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # Stage 4
        x, H, W = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs


class MLPDecoder(nn.Module):
    """
    轻量级 MLP Decoder for SegFormer
    将多尺度特征融合并上采样到原始分辨率
    """

    def __init__(
        self,
        encoder_dims: List[int],
        decoder_dim: int = 256,
        num_classes: int = 19
    ):
        super().__init__()
        self.encoder_dims = encoder_dims
        self.decoder_dim = decoder_dim
        self.num_classes = num_classes

        # 线性投影层，将 encoder 特征映射到相同维度
        self.linear_c4 = nn.Linear(encoder_dims[3], decoder_dim)
        self.linear_c3 = nn.Linear(encoder_dims[2], decoder_dim)
        self.linear_c2 = nn.Linear(encoder_dims[1], decoder_dim)
        self.linear_c1 = nn.Linear(encoder_dims[0], decoder_dim)

        # 特征融合和上采样
        self.linear_fuse = nn.Conv2d(decoder_dim * 4, decoder_dim, kernel_size=1)
        self.batch_norm = nn.BatchNorm2d(decoder_dim)
        self.activation = nn.ReLU()

        # 输出层
        self.linear_pred = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)

    def forward(self, encoder_features: List[torch.Tensor], target_size: Tuple[int, int]):
        """
        Args:
            encoder_features: 来自 backbone 的多尺度特征列表 [c1, c2, c3, c4]
            target_size: 目标输出尺寸 (H, W)

        Returns:
            分割 logits [B, num_classes, H, W]
        """
        B, _, H, W = encoder_features[0].shape

        # 收集各阶段特征
        c1, c2, c3, c4 = encoder_features

        # 投影到相同维度并上采样到相同尺寸
        # c1: H/4, W/4 -> 上采样到 H/4, W/4 (相对原始图像)
        # c2: H/8, W/8 -> 上采样到 H/4, W/4
        # c3: H/16, W/16 -> 上采样到 H/4, W/4
        # c4: H/32, W/32 -> 上采样到 H/4, W/4

        # 获取原始图像尺寸
        H0, W0 = target_size

        # 投影并上采样
        c1 = self.linear_c1(c1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # B, decoder_dim, H/4, W/4
        c2 = self.linear_c2(c2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # B, decoder_dim, H/8, W/8
        c3 = self.linear_c3(c3.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # B, decoder_dim, H/16, W/16
        c4 = self.linear_c4(c4.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # B, decoder_dim, H/32, W/32

        # 上采样到相同尺寸 (H/4, W/4)
        c1 = F.interpolate(c1, size=(H0 // 4, W0 // 4), mode='bilinear', align_corners=False)
        c2 = F.interpolate(c2, size=(H0 // 4, W0 // 4), mode='bilinear', align_corners=False)
        c3 = F.interpolate(c3, size=(H0 // 4, W0 // 4), mode='bilinear', align_corners=False)
        c4 = F.interpolate(c4, size=(H0 // 4, W0 // 4), mode='bilinear', align_corners=False)

        # 拼接所有特征
        x = torch.cat([c4, c3, c2, c1], dim=1)

        # 融合
        x = self.linear_fuse(x)
        x = self.batch_norm(x)
        x = self.activation(x)

        # 上采样到目标尺寸
        x = F.interpolate(x, size=(H0, W0), mode='bilinear', align_corners=False)

        # 输出分割 logits
        x = self.linear_pred(x)

        return x
        x = self.linear_fuse(x)
        x = self.batch_norm(x)
        x = self.activation(x)

        # 上采样到目标尺寸
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)

        # 输出分割 logits
        x = self.linear_pred(x)

        return x


@MODEL_REGISTRY.register()
class SegFormerB2(BaseSegmentationModel):
    """
    SegFormer-B2 模型
    - Backbone: MiT-B2
    - Decoder: 轻量 MLP Decoder
    """

    def __init__(
        self,
        num_classes: int = 19,
        in_channels: int = 3,
        pretrained: bool = False,
        decoder_dim: int = 768
    ):
        """
        Args:
            num_classes: 分割类别数
            in_channels: 输入通道数
            pretrained: 是否加载 ImageNet 预训练权重
            decoder_dim: Decoder 中间维度
        """
        super().__init__(num_classes, in_channels)

        # MiT-B2 配置
        # embed_dims: [64, 128, 320, 512]
        # depths: [3, 4, 6, 3]
        # num_heads: [1, 2, 5, 8]
        self.backbone = MiT(
            in_chans=in_channels,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            depths=[3, 4, 6, 3],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1
        )

        # MLP Decoder
        self.decoder = MLPDecoder(
            encoder_dims=[64, 128, 320, 512],
            decoder_dim=decoder_dim,
            num_classes=num_classes
        )

        self.pretrained = pretrained
        self.decoder_dim = decoder_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入图像 [B, 3, H, W]

        Returns:
            分割 logits [B, num_classes, H, W]
        """
        # 获取多尺度特征
        features = self.backbone(x)

        # MLP Decoder
        logits = self.decoder(features, target_size=(x.shape[2], x.shape[3]))

        return logits

    def get_params_groups(self) -> list:
        """获取参数组，用于不同的学习率设置"""
        return [
            {'params': self.backbone.parameters(), 'lr_mult': 1.0},
            {'params': self.decoder.parameters(), 'lr_mult': 1.0}
        ]

    def load_pretrained(self, pretrained_path: str, strict: bool = True):
        """加载预训练权重"""
        if pretrained_path.startswith('http'):
            # 从 URL 加载
            state_dict = torch.hub.load_state_dict_from_url(
                pretrained_path, map_location='cpu', check_hash=True
            )
        else:
            # 从本地文件加载
            state_dict = torch.load(pretrained_path, map_location='cpu')

        # 处理 checkpoint 格式
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        if 'model' in state_dict:
            state_dict = state_dict['model']

        # 加载权重
        self.load_state_dict(state_dict, strict=strict)

    @staticmethod
    def build_from_config(config):
        """从配置构建模型"""
        model_cfg = config['model']
        return SegFormerB2(
            num_classes=model_cfg['num_classes'],
            in_channels=model_cfg.get('in_channels', 3),
            pretrained=model_cfg.get('pretrained', False),
            decoder_dim=model_cfg.get('decoder_dim', 768)
        )


@MODEL_REGISTRY.register()
class MitB2Backbone(nn.Module):
    """
    MiT-B2 Backbone 独立模块
    可复用于 anomaly segmentation
    """

    def __init__(
        self,
        in_channels: int = 3,
        pretrained: bool = False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.pretrained = pretrained

        # MiT-B2 配置
        self.backbone = MiT(
            in_chans=in_channels,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            depths=[3, 4, 6, 3],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        返回多尺度特征

        Returns:
            [c1, c2, c3, c4] - 4 个尺度的特征
        """
        return self.backbone(x)

    def get_embedding_dims(self) -> List[int]:
        """返回各阶段的通道数"""
        return [64, 128, 320, 512]

    def load_pretrained(self, pretrained_path: str, strict: bool = True):
        """加载预训练权重"""
        if pretrained_path.startswith('http'):
            state_dict = torch.hub.load_state_dict_from_url(
                pretrained_path, map_location='cpu', check_hash=True
            )
        else:
            state_dict = torch.load(pretrained_path, map_location='cpu')

        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        if 'model' in state_dict:
            state_dict = state_dict['model']

        self.load_state_dict(state_dict, strict=strict)
