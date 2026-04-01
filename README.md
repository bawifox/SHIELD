<<<<<<< HEAD
# SHIELD: Small-Hazard Prior for Road Scene Anomaly Segmentation

## Overview

SHIELD (Small-Hazard prIor for Road scene anomaly sEgmentation with Mixed-scaLe Detection) is a novel framework for road scene anomaly segmentation, with a particular focus on detecting small-scale anomalous objects in mixed-scale scenarios. This repository contains the PyTorch implementation of SHIELD-Lite.

<p align="center">
  <img src="assets/pipeline.png" alt="SHIELD Pipeline" width="800"/>
</p>

## Key Features

- **Small-Hazard Prior**: A novel prior mechanism that enhances the detection of small anomalous objects by modeling their hazard levels
- **Candidate Harvesting**: Intelligent extraction of candidate regions from coarse anomaly maps with adaptive thresholding
- **Hazard Scorer**: Learns to assess the hazard level of candidate regions based on area and overlap metrics
- **Mixed-Scale Robustness**: Maintains stable performance across both homogeneous and mixed-scale road scenes
- **Efficient Design**: 44.2M parameters with 18.45 FPS inference speed

## Method

SHIELD consists of four core components:

1. **Candidate Extractor**: Extracts candidate anomalous regions from the coarse prediction map using adaptive thresholding
2. **Hazard Scorer**: Predicts hazard scores for each candidate based on area ratios and IoU with ground truth
3. **Small Hazard Prior Generator**: Generates prior maps by weighting candidate masks with their hazard scores
4. **Adaptive Threshold Module**: Dynamically adjusts decision thresholds based on scene complexity

The final prediction is computed by fusing the coarse anomaly map with the small-hazard prior.

## Experimental Results

### Main Results

| Method | AP_small (H/M) | delta_s | Params | FPS |
|--------|----------------|---------|--------|-----|
| PEBAL | 78.2 / 51.9 | 26.4 | 160.2M | 3.89 |
| RbA | 89.8 / 83.6 | 6.1 | 95.1M | 10.16 |
| Mask2Anomaly | 85.1 / 65.2 | 19.9 | 46.5M | 2.76 |
| **SHIELD** | **89.9 / 84.1** | **5.8** | 44.2M | **18.45** |

*H: homogeneous scenes, M: mixed-scale scenes, delta_s: performance drop from H to M*

### Ablation Study

| Variant | AP_small | delta_s |
|---------|----------|---------|
| w/o language semantics | 71.3 | 18.2 |
| w/o candidate harvesting | 60.4 | 22.5 |
| fixed threshold | 73.8 | 16.3 |
| **Full model** | **84.1** | **5.8** |

## Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/shield.git
cd shield

# Create conda environment
conda create -n shield python=3.10
conda activate shield

# Install dependencies
pip install torch torchvision
pip install pyyaml numpy scipy opencv-python matplotlib pillow
pip install scikit-learn
```

## Dataset Preparation

Place your synthetic anomaly dataset following the structure:

```
synthetic_dataset/
├── synthetic_images/
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
└── synthetic_masks/
    ├── mask_001.png
    ├── mask_002.png
    └── ...
```

Update the data path in `configs/shield_lite.yaml`:

```yaml
anomaly_dataset:
  data_root: "/path/to/your/synthetic_dataset"
```

## Quick Start

### Training

```bash
# Single GPU
python train_shield_lite.py --config configs/shield_lite.yaml

# Multi-GPU (Distributed)
bash train_shield_lite_dist.sh
```

### Evaluation

```bash
python sanity_check_shield_lite.py --config configs/shield_lite.yaml --checkpoint checkpoints/shield_lite/best.pth
```

### Inference

```bash
python eval_anomaly_seg.py --config configs/shield_lite.yaml \
    --checkpoint checkpoints/shield_lite/best.pth \
    --image_path /path/to/test/image.jpg \
    --output_dir outputs/
```

## Configuration

Key hyperparameters in `configs/shield_lite.yaml`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `decoder_dim` | Decoder feature dimension | 256 |
| `hazard_beta` | Temperature for hazard scoring | 10.0 |
| `candidate_threshold_high` | High response threshold | 0.5 |
| `candidate_threshold_small` | Small region threshold | 0.3 |
| `lambda_coarse` | Coarse branch loss weight | 0.5 |
| `lambda_hazard` | Hazard score loss weight | 0.5 |

## Architecture

```
AnomalySegmentationModelWithPrior
├── MitB2Backbone (pretrained on Cityscapes)
├── SimpleAnomalyDecoder
│   ├── Coarse Branch (c1, c2 features)
│   └── Final Branch (all features)
├── HazardScorerWithAdaptiveThreshold
│   └── Hazard Score Prediction
└── SmallHazardPriorGenerator
    └── Prior Map Generation
```

## Citation

If you find this work helpful for your research, please cite:

```bibtex
@article{shield2026,
  title={SHIELD: Small-Hazard Prior for Road Scene Anomaly Segmentation},
  author={},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## License

This project is released under the Apache 2.0 License.

## Acknowledgments

This project builds upon [SegFormer](https://github.com/NVIDIA/SegFormer) and [Mask2Anomaly](https://github.com/iro-cp/Mask2Anomaly). We thank the authors for their excellent work.
=======
# SHIELD
>>>>>>> 92af250366256c189a78c4995a4df5893e60e2d6
