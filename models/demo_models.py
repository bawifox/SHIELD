"""
Demo 模型实现 - 简单的 U-Net 模型
用于演示项目结构，实际使用请替换为更复杂的模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import MODEL_REGISTRY
from .base import BaseSegmentationModel


class DoubleConv(nn.Module):
    """UNet 的基本卷积块: Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """下采样模块: MaxPool -> DoubleConv"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """上采样模块: Upsample -> Concat -> DoubleConv"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 处理尺寸不匹配
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


@MODEL_REGISTRY.register()
class UNet(BaseSegmentationModel):
    """
    简单的 U-Net 实现，用于语义分割。
    """

    def __init__(self, num_classes: int = 19, in_channels: int = 3, base_channels: int = 64):
        """
        初始化 UNet。

        Args:
            num_classes: 分割类别数
            in_channels: 输入通道数
            base_channels: 基础通道数
        """
        super().__init__(num_classes, in_channels)
        self.base_channels = base_channels

        # 编码器
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16)

        # 解码器
        self.up1 = Up(base_channels * 16, base_channels * 8)
        self.up2 = Up(base_channels * 8, base_channels * 4)
        self.up3 = Up(base_channels * 4, base_channels * 2)
        self.up4 = Up(base_channels * 2, base_channels)

        # 输出层
        self.outc = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        # 编码器
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # 解码器
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # 输出
        logits = self.outc(x)
        return logits

    @staticmethod
    def build_from_config(config):
        """从配置构建模型"""
        model_cfg = config['model']
        return UNet(
            num_classes=model_cfg['num_classes'],
            in_channels=model_cfg.get('in_channels', 3)
        )


@MODEL_REGISTRY.register()
class ResNet50UNet(BaseSegmentationModel):
    """
    带 ResNet50 编码器的 U-Net。
    实际实现可以加载 ImageNet 预训练权重。
    """

    def __init__(self, num_classes: int = 19, in_channels: int = 3, pretrained: bool = False):
        """
        初始化 ResNet50UNet。

        Args:
            num_classes: 分割类别数
            in_channels: 输入通道数
            pretrained: 是否使用 ImageNet 预训练
        """
        super().__init__(num_classes, in_channels)
        # 简化实现：使用基础 UNet
        # 实际可以使用 torchvision 的 resnet50 作为编码器
        self.model = UNet(num_classes, in_channels)

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def build_from_config(config):
        """从配置构建模型"""
        model_cfg = config['model']
        return ResNet50UNet(
            num_classes=model_cfg['num_classes'],
            in_channels=model_cfg.get('in_channels', 3)
        )
