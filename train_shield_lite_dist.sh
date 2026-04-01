#!/bin/bash
# =============================================================================
# SHIELD-Lite Distributed Training Script
# 使用 4 卡: cuda:4, cuda:5, cuda:6, cuda:7
# =============================================================================

# GPU 设置
export CUDA_VISIBLE_DEVICES=4,5,6,7

# 训练参数
CONFIG="configs/shield_lite.yaml"
GPUS=4
BATCH_SIZE=8  # 每卡 batch size，总共 8*4=32
NUM_WORKERS=4
NUM_EPOCHS=100

# 输出目录
EXP_NAME="shield_lite_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/${EXP_NAME}"
CHECKPOINT_DIR="checkpoints/${EXP_NAME}"
VIS_DIR="visualizations/${EXP_NAME}"

# 创建目录
mkdir -p ${LOG_DIR}
mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${VIS_DIR}

# 训练命令
python -m torch.distributed.launch \
    --nproc_per_node=${GPUS} \
    --master_port=29500 \
    train_shield_lite.py \
    --config ${CONFIG} \
    --exp_name ${EXP_NAME} \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --num_epochs ${NUM_EPOCHS} \
    --log_dir ${LOG_DIR} \
    --checkpoint_dir ${CHECKPOINT_DIR} \
    --vis_dir ${VIS_DIR} \
    --distributed

echo "Training started!"
echo "Experiment: ${EXP_NAME}"
echo "Log dir: ${LOG_DIR}"
echo "Checkpoint dir: ${CHECKPOINT_DIR}"
