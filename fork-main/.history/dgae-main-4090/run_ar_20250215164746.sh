#!/bin/bash
set -e

date=$(date +%Y%m%d)
wandb='online'
# wandb='disabled'

# exp_name='dgae_baseline_different_hyper'
# exp_name='dgae_baseline_different_hyper'
# exp_name='citeseer_r4'
# run_name='community_cb_compare'
# exp_name='community_test_prior'
# exp_name='community_decode'
# exp_name='community_index'
# exp_name='community_index'
# exp_name='grid_base'
# exp_name='ar_baseline'
exp_name='ar_recon_gen_collaps'

# 定义实验配置，现在包含数据集信息
declare -A experiment_configs=(
    # ['community_16_1']='community 16 1 2'
    # ['community_32_1']='community 32 1 2'
    # ['community_256_1']='community 256 1 2'
    ['community_512_1']='community 512 1 2'
)
for exp_config in "${!experiment_configs[@]}"; do
    read -r dataset _ _ _ <<< "${experiment_configs[$exp_config]}"
    mkdir -p models_own/$exp_name/$exp_config/${dataset}_autoencoder
    mkdir -p models_own/$exp_name/$exp_config/${dataset}_prior
    mkdir -p models_own/$exp_name/$exp_config/${dataset}_sample
done

# 修改运行实验的函数
run_experiment() {
    local exp_config=$1
    read -r dataset codebook_size nc gpu <<< "${experiment_configs[$exp_config]}"
    
    echo "开始实验配置: $exp_config (数据集: $dataset, GPU: $gpu, Codebook Size: $codebook_size, NC: $nc)"
   
    # # 训练自编码器
    echo "[$exp_config] train_autoencoder 开始..."
    python main.py --work_type train_autoencoder --gpu $gpu --exp_name $exp_name --run_name $exp_config --dataset $dataset --wandb $wandb --codebook_size $codebook_size --nc $nc 2>&1 | tee "models_own/$exp_name/$exp_config/${dataset}_autoencoder/$date.log"
    echo "[$exp_config] train_autoencoder 完成"
    
    echo "[$exp_config] train_prior 开始..."
    python main.py --work_type train_prior --gpu $gpu --exp_name $exp_name --run_name $exp_config --dataset $dataset --wandb $wandb --codebook_size $codebook_size --nc $nc 2>&1 | tee "models_own/$exp_name/$exp_config/${dataset}_prior/$date.log"
    echo "[$exp_config] train_prior 完成"

    # 采样
    echo "[$exp_config] sample 开始..."
    python main.py --work_type sample --gpu $gpu --exp_name $exp_name --run_name $exp_config --dataset $dataset --codebook_size $codebook_size --nc $nc 2>&1 | tee "models_own/$exp_name/$exp_config/${dataset}_sample/$date.log"
    echo "[$exp_config] sample 完成"
}

# 并行运行所有实验配置
declare -A experiment_pids
for exp_config in "${!experiment_configs[@]}"; do
    run_experiment "$exp_config" &
    experiment_pids[$exp_config]=$!
done

# # 等待任务完成并显示完成状态
# remaining_tasks=${#experiment_pids[@]}
# while [ $remaining_tasks -gt 0 ]; do
#     for exp_config in "${!experiment_pids[@]}"; do
#         if ! kill -0 ${experiment_pids[$exp_config]} 2>/dev/null; then
#             if wait ${experiment_pids[$exp_config]}; then
#                 echo "✅ 实验配置 $exp_config 处理完成"
#             else
#                 echo "❌ 实验配置 $exp_config 处理失败"
#             fi
#             unset experiment_pids[$exp_config]
#             ((remaining_tasks--))
#         fi
#     done
#     sleep 5
# done