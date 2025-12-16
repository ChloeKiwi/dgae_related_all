#!/bin/bash
set -e

date=$(date +%Y%m%d-%H%M)
wandb='online'
# wandb='disabled'

# 实验名称
# exp_name="test_bert_type_mask_debug"
exp_name="test_bert_type_mask_20251212-22"

# 配置参数 (对应原脚本中的 experiment_configs)
# 原配置: ['community_128_1_ignore_pad_cosine_bert_mask_type']="community 128 1 0 cosine 1"
# config_name="community_128_1_ignore_pad_cosine_bert_mask_type"
config_name="community_128_1_ignore_pad_cosine_predict_eos_pad_cal_mask_loss"

dataset="community"
codebook_size=128
nc=1
gpu=0
mask_func="cosine"
iterations_rate=1

# 创建必要的目录
mkdir -p models_own/$exp_name/$config_name/${dataset}_prior

echo "开始实验配置: $config_name (数据集: $dataset, GPU: $gpu, Codebook Size: $codebook_size, NC: $nc, Mask Function: $mask_func, Iterations Rate: $iterations_rate)"

echo "[$config_name] train_prior 开始..."

# 运行命令 (train_prior)
# 注意：移除了后台运行符 '&'，以便支持 ipdb 交互调试
python main.py \
    --work_type train_prior \
    --gpu $gpu \
    --exp_name $exp_name \
    --run_name $config_name \
    --dataset $dataset \
    --wandb $wandb \
    --codebook_size $codebook_size \
    --nc $nc \
    --use_mask \
    --mask_func $mask_func \
    --iterations_rate $iterations_rate \
    2>&1 | tee "models_own/$exp_name/$config_name/${dataset}_prior/$date.log"

echo "[$config_name] train_prior 完成"

