"""
Cityscapes 数据集标签映射
Cityscapes 原始有 35 类（包含 19 类有效类别 + 背景类别），需要映射到训练用的 19 类
"""

# Cityscapes 原始标签 ID -> 有效类别 ID 映射
# 0-18 为有效类别，255 为 ignore index
# 参考: https://cityscapes-dataset.com/static/docs/downloads.html

# Cityscapes 35 类标签映射到 19 类
# -1 表示该类别被忽略
CITYSCAPES_LABEL_MAP = {
    # 道路相关 (road)
    7: 0,   # road
    
    # 建筑相关 (construction)
    11: 1,  # building
    12: 2,  # wall
    13: 3,  # fence
    
    # 物体相关 (object)
    26: 4,  # traffic sign
    27: 5,  # traffic light
    
    # 天空相关 (nature)
    23: 6,  # vegetation
    22: 7,  # terrain
    
    # 人类相关 (human)
    24: 8,  # person
    25: 9,  # rider
    
    # 车辆相关 (vehicle)
    17: 10, # car
    18: 11, # truck
    19: 12, # bus
    20: 13, # train
    21: 14, # motorcycle
    22: 15, # bicycle
    15: 16, # dynamic (视为 motorcycle)
    16: 17, # static (视为 bicycle)
}

# 完整的 35 类原始标签名称
CITYSCAPES_35_CLASSES = [
    'unlabeled', 'ego vehicle', 'rectification border', 'out of roi', 'static',
    'dynamic', 'ground', 'road', 'sidewalk', 'parking', 'rail track',
    'building', 'wall', 'fence', 'guard rail', 'bridge', 'tunnel',
    'pole', 'polegroup', 'traffic light', 'traffic sign', 'vegetation',
    'terrain', 'person', 'rider', 'car', 'truck', 'bus', 'caravan',
    'trailer', 'train', 'motorcycle', 'bicycle'
]

# 训练用的 19 类名称
CITYSCAPES_19_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign',
    'vegetation', 'terrain', 'person', 'rider', 'car', 'truck', 'bus', 'train',
    'motorcycle', 'bicycle'
]

# 19 类的颜色（用于可视化）
CITYSCAPES_PALETTE = [
    [128, 64, 128],    # road - 0
    [244, 35, 232],    # sidewalk - 1
    [70, 70, 70],      # building - 2
    [102, 102, 156],   # wall - 3
    [190, 153, 153],   # fence - 4
    [153, 153, 153],   # pole - 5
    [250, 170, 30],    # traffic light - 6
    [220, 220, 0],     # traffic sign - 7
    [107, 142, 35],    # vegetation - 8
    [152, 251, 152],   # terrain - 9
    [220, 20, 60],     # person - 10
    [255, 0, 0],       # rider - 11
    [0, 0, 142],       # car - 12
    [0, 0, 70],        # truck - 13
    [0, 60, 100],      # bus - 14
    [0, 80, 100],      # train - 15
    [0, 0, 230],       # motorcycle - 16
    [119, 11, 32],     # bicycle - 17
]


def remap_cityscapes_label(label: int) -> int:
    """
    将 Cityscapes 原始 35 类标签映射到 19 类训练标签。

    Args:
        label: Cityscapes 原始标签 ID (0-34)

    Returns:
        映射后的标签 ID (0-18)，无效类别返回 255
    """
    # 有效类别直接映射
    if label in CITYSCAPES_LABEL_MAP:
        return CITYSCAPES_LABEL_MAP[label]
    # 其他类别返回 ignore index
    return 255


def create_label_remap_array() -> list:
    """
    创建用于 numpy 数组快速映射的数组。

    Returns:
        大小为 256 的数组，可用于 np.take() 或索引
    """
    remap = [255] * 256
    for orig_label, new_label in CITYSCAPES_LABEL_MAP.items():
        remap[orig_label] = new_label
    return remap


# 预创建的映射数组，用于加速
CITYSCAPES_REMAP_ARRAY = create_label_remap_array()
