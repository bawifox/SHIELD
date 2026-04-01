python -m torch.distributed.launch --nproc_per_node=4 --master_port=29500 \
  train_shield_lite.py --config configs/shield_lite.yaml \
  --resume checkpoints/shield_lite/latest.pth