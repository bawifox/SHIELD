"""
数据增强 transforms
支持训练时的数据增强操作
"""

import numpy as np
import torch
from PIL import Image, ImageEnhance
import random
from typing import Dict, Any, Tuple, Optional


class Compose:
    """组合多个 transform"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for t in self.transforms:
            data = t(data)
        return data


class Resize:
    """调整图像和标签大小"""

    def __init__(self, size: Tuple[int, int]):
        """
        Args:
            size: (h, w)
        """
        self.size = size

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        image = data['image']
        label = data.get('label')
        mask = data.get('mask')

        # 调整图像大小
        image = image.resize(self.size, Image.BILINEAR)

        if label is not None:
            label = label.resize(self.size, Image.NEAREST)
        
        if mask is not None:
            mask = mask.resize(self.size, Image.NEAREST)

        data['image'] = image
        data['label'] = label
        data['mask'] = mask
        return data


class RandomResize:
    """随机缩放"""

    def __init__(self, min_size: int = 512, max_size: int = 2048, ratio: Tuple[float, float] = (0.5, 2.0)):
        """
        Args:
            min_size: 最小尺寸
            max_size: 最大尺寸
            ratio: 缩放比例范围
        """
        self.min_size = min_size
        self.max_size = max_size
        self.ratio = ratio

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        image = data['image']
        label = data.get('label')
        mask = data.get('mask')

        # 随机选择缩放比例
        ratio = random.uniform(self.ratio[0], self.ratio[1])
        w, h = image.size
        new_h = int(h * ratio)
        new_w = int(w * ratio)

        # 限制尺寸范围
        new_h = max(self.min_size, min(new_h, self.max_size))
        new_w = max(self.min_size, min(new_w, self.max_size))

        # 调整图像大小
        image = image.resize((new_w, new_h), Image.BILINEAR)

        if label is not None:
            label = label.resize((new_w, new_h), Image.NEAREST)
        
        if mask is not None:
            mask = mask.resize((new_w, new_h), Image.NEAREST)

        data['image'] = image
        data['label'] = label
        data['mask'] = mask
        return data


class RandomCrop:
    """随机裁剪"""

    def __init__(self, crop_size: Tuple[int, int]):
        """
        Args:
            crop_size: (h, w)
        """
        self.crop_size = crop_size

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        image = data['image']
        label = data.get('label')
        mask = data.get('mask')

        w, h = image.size
        crop_h, crop_w = self.crop_size

        # 确保图像足够大
        if w < crop_w or h < crop_h:
            # 如果图像太小，先 resize
            scale = max(crop_w / w, crop_h / h) + 0.1
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = image.resize((new_w, new_h), Image.BILINEAR)
            if label is not None:
                label = label.resize((new_w, new_h), Image.NEAREST)
            if mask is not None:
                mask = mask.resize((new_w, new_h), Image.NEAREST)
            w, h = new_w, new_h

        # 随机裁剪位置
        if w == crop_w:
            left = 0
        else:
            left = random.randint(0, w - crop_w)

        if h == crop_h:
            top = 0
        else:
            top = random.randint(0, h - crop_h)

        right = left + crop_w
        top = top
        bottom = top + crop_h

        # 裁剪
        image = image.crop((left, top, right, bottom))

        if label is not None:
            label = label.crop((left, top, right, bottom))
        
        if mask is not None:
            mask = mask.crop((left, top, right, bottom))

        data['image'] = image
        data['label'] = label
        data['mask'] = mask
        return data


class RandomHorizontalFlip:
    """随机水平翻转"""

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() < self.prob:
            image = data['image']
            label = data.get('label')
            mask = data.get('mask')

            image = image.transpose(Image.FLIP_LEFT_RIGHT)

            if label is not None:
                label = label.transpose(Image.FLIP_LEFT_RIGHT)
            
            if mask is not None:
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

            data['image'] = image
            data['label'] = label
            data['mask'] = mask

        return data


class ColorJitter:
    """颜色抖动"""

    def __init__(self, brightness: float = 0.2, contrast: float = 0.2,
                 saturation: float = 0.2, hue: float = 0.1):
        from PIL import ImageEnhance
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        image = data['image']

        # 随机调整亮度
        if self.brightness > 0:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(factor)

        # 随机调整对比度
        if self.contrast > 0:
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(factor)

        # 随机调整饱和度
        if self.saturation > 0:
            factor = 1.0 + random.uniform(-self.saturation, self.saturation)
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(factor)

        data['image'] = image
        return data


class ToTensor:
    """转换为 Tensor"""

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        image = data['image']
        label = data.get('label')
        mask = data.get('mask')

        # 转换为 numpy array
        image = np.array(image).astype(np.float32) / 255.0

        # HWC -> CHW
        image = image.transpose(2, 0, 1)

        data['image'] = torch.from_numpy(image)

        if label is not None:
            label = np.array(label).astype(np.int64)
            data['label'] = torch.from_numpy(label)
        
        if mask is not None:
            mask = np.array(mask).astype(np.float32)
            mask = torch.from_numpy(mask).unsqueeze(0)  # 添加 channel 维度 [1, H, W]
            data['mask'] = mask

        return data


class Normalize:
    """归一化"""

    def __init__(self, mean: Tuple[float, float, float], std: Tuple[float, float, float]):
        """
        Args:
            mean: RGB 均值
            std: RGB 标准差
        """
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        image = data['image']

        if isinstance(image, torch.Tensor):
            image = (image - self.mean) / self.std
        else:
            image = torch.from_numpy(image)
            image = (image - self.mean) / self.std

        data['image'] = image
        return data


# ImageNet 归一化参数
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transforms(config: Dict[str, Any]):
    """
    构建训练数据增强 pipeline。

    Args:
        config: 配置字典

    Returns:
        Compose transform
    """
    # 从 transforms.train 读取配置，如果没有则使用默认值
    transform_cfg = config.get('transforms', {}).get('train', {})
    resize = tuple(transform_cfg.get('resize', [256, 512]))
    use_random_crop = transform_cfg.get('random_crop', False)
    use_flip = transform_cfg.get('random_flip', True)
    use_color_jitter = transform_cfg.get('color_jitter', True)

    transforms = []

    # Resize
    transforms.append(Resize(size=resize))

    if use_random_crop:
        transforms.append(RandomCrop(crop_size=resize))

    if use_flip:
        transforms.append(RandomHorizontalFlip(prob=0.5))

    if use_color_jitter:
        transforms.append(ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))

    transforms.extend([
        ToTensor(),
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return Compose(transforms)


def build_val_transforms(config: Dict[str, Any]):
    """
    构建验证数据增强 pipeline。

    Args:
        config: 配置字典

    Returns:
        Compose transform
    """
    # 从 transforms.val 读取配置，如果没有则使用默认值
    transform_cfg = config.get('transforms', {}).get('val', {})
    resize = tuple(transform_cfg.get('resize', [256, 512]))

    transforms = [
        Resize(size=resize),
        ToTensor(),
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]

    return Compose(transforms)
