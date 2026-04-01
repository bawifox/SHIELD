# SHIELD: Region-to-Pixel Hazard Reasoning

## Overview

<img width="1399" height="610" alt="db762174e236b9c8b4c92130cef0107f" src="https://github.com/user-attachments/assets/76bfea50-23cb-40e6-88e5-10ddf501287a" />

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
### Model Weights

The pre-trained weights for SHIELD are hosted on Google Drive. Please note that the backbone network was pre-trained on the **Cityscapes** dataset.

[📥 Download Weights from Google Drive](https://drive.google.com/file/d/1DylWRNCJuYPky0Zz8-v22mqBJpA_wcvg/view?usp=sharing)

Place the downloaded `.pth` files into the `checkpoints/cityscapes` directory before running the code.

### Preparation
After downloading, please place the `.pth` files into their respective directories to match the expected structure:
```text
SHIELD/
└── checkpoints/
    ├── cityscapes/
    │   └── backbone_weights.pth
    └── shield_lite/
        └── best.pth
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

## License

This project is released under the Apache 2.0 License.

## Acknowledgments

This project builds upon [SegFormer](https://github.com/NVIDIA/SegFormer) and [CityScape](https://github.com/mcordts/cityscapesScripts). We thank the authors for their excellent work.
```
@article{xie2021segformer,
  title={SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers},
  author={Xie, Enze and Wang, Wenhai and Yu, Zhiding and Anandkumar, Anima and Alvarez, Jose M and Luo, Ping},
  journal=NeurIPs,
  volume={34},
  year={2021}
}
@inproceedings{cordts2016cityscapes,
  author    = {Cordts, Marius and Omran, Mohamed and Ramos, Sebastian and Rehfeld, Timo and Enzweiler, Markus and Benenson, Rodrigo and Franke, Uwe and Roth, Stefan and Schiele, Bernt},
  title     = {The cityscapes dataset for semantic urban scene understanding},
  booktitle = CVPR,
  year      = {2016}
}
```
