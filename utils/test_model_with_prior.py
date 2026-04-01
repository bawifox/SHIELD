"""
Test for AnomalySegmentationModelWithPrior
"""

import sys
sys.path.insert(0, '/data/lyx/mm2026')

import torch
from models.anomaly_segmentation import AnomalySegmentationModel, AnomalySegmentationModelWithPrior, AnomalySegmentationOutput


def test_model():
    print("=" * 60)
    print("Testing AnomalySegmentationModelWithPrior...")
    print("=" * 60)
    
    B, C, H, W = 2, 3, 256, 512
    
    # Test 1: 原始模型
    print("\n[1] Testing AnomalySegmentationModel...")
    model_orig = AnomalySegmentationModel(
        in_channels=3,
        decoder_dim=256
    )
    
    x = torch.randn(B, C, H, W)
    coarse, final = model_orig(x)
    print(f"  Input: {x.shape}")
    print(f"  Coarse: {coarse.shape}, Final: {final.shape}")
    
    # Test 2: 带 Prior 的模型
    print("\n[2] Testing AnomalySegmentationModelWithPrior...")
    model_prior = AnomalySegmentationModelWithPrior(
        in_channels=3,
        decoder_dim=256,
        use_hazard_scorer=True,
        use_small_hazard_prior=True,
        prior_fusion_mode='add'
    )
    
    output = model_prior(x, return_details=True)
    print(f"  Input: {x.shape}")
    print(f"  Coarse logits: {output.coarse_logits.shape}")
    print(f"  Final logits: {output.final_logits.shape}")
    print(f"  Hazard scores: {output.hazard_scores}")
    print(f"  Delta t: {output.delta_t}")
    print(f"  Small hazard prior: {output.small_hazard_prior}")
    
    # Test 3: 带候选的完整前向
    print("\n[3] Testing forward_with_hazard_prior...")
    N = 5
    candidate_masks = torch.randint(0, 2, (B, N, H, W)).float()
    area_ratios = torch.rand(B, N) * 0.05
    text_similarities = torch.rand(B, N, 2)
    
    output_full = model_prior.forward_with_hazard_prior(
        x, candidate_masks, area_ratios, text_similarities
    )
    print(f"  Hazard scores: {output_full.hazard_scores.shape}")
    print(f"  Small hazard prior: {output_full.small_hazard_prior.shape}")
    print(f"  Final logits: {output_full.final_logits.shape}")
    
    # Test 4: 不同的 prior 融合模式
    print("\n[4] Testing different fusion modes...")
    for mode in ['add', 'concat', 'attention']:
        model = AnomalySegmentationModelWithPrior(
            in_channels=3,
            decoder_dim=256,
            prior_fusion_mode=mode
        )
        out = model(x, return_details=False)
        print(f"  Mode {mode}: final_logits shape = {out.final_logits.shape}")
    
    # Test 5: 梯度检查
    print("\n[5] Testing gradient...")
    model_prior.train()
    x.requires_grad = True
    output = model_prior(x, return_details=True)
    loss = output.final_logits.sum()
    loss.backward()
    print(f"  Gradient exists: {x.grad is not None}")
    print(f"  Gradient norm: {x.grad.norm().item():.4f}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_model()
