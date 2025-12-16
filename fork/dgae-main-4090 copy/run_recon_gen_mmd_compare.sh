#!/bin/bash
set -e

copy_wandb_files() {
    local work_type=$1
    local exp_config=$2  # 添加 exp_config 参数
    local dataset="community"
    # 读取特定运行的 run ID
    local run_id=$(cat "./models_own/$exp_name/$exp_config/${dataset}_${work_type}/wandb_run_id")
    if [ -n "$run_id" ]; then
        # echo "复制 ${work_type} 的wandb文件 (Run ID: ${run_id})..."
        if [ -d "models_own/$exp_name/$exp_config/${dataset}_${work_type}" ]; then
            rm -rf "models_own/$exp_name/$exp_config/${dataset}_${work_type}"/*
        fi
        cp -r "./wandb/${run_id}/files"/* "./models_own/$exp_name/$exp_config/${dataset}_${work_type}"
        # rm "./models_own/$exp_name/$exp_config/${dataset}_${work_type}/wandb_run_id"  # 清理临时文件
    else
        echo "警告: 找不到wandb运行ID"
    fi
}

date=$(date +%Y%m%d)
wandb='online'

exp_name='dgae_baseline_different_hyper'
# run_name='community_cb_compare'

# 定义实验配置
declare -A experiment_configs=(
    # ["community_cb8_1"]="8 1 0"      # format: "codebook_size nc gpu"
    # ["community_cb16_1"]="16 1 1"
    # ["community_cb32_1"]="32 1 0"
    # ["community_cb64_1"]="64 1 1"
    ["community_cb128_1"]="128 1 0"
    # ["community_cb256_1"]="256 1 1"
    # ["community_cb512_1"]="512 1 2"
    # ["community_cb1024_1"]="1024 1 3"
    # ["community_cb8_2"]="8 2 0"
    # ["community_cb16_2"]="16 2 1"
    ["community_cb32_2"]="32 2 0"
    ["community_cb64_2"]="64 2 1"
    # # ["community_cb128_2"]="128 2 3"
    ["community_cb256_2"]="256 2 1"    
)

# 创建所有必要的目录
for exp_config in "${!experiment_configs[@]}"; do
    mkdir -p models_own/$exp_name/$exp_config/community_autoencoder
    mkdir -p models_own/$exp_name/$exp_config/community_prior
    mkdir -p models_own/$exp_name/$exp_config/community_sample
done

# 定义运行单个配置的所有步骤的函数
run_experiment() {
    local exp_config=$1
    read -r codebook_size nc gpu <<< "${experiment_configs[$exp_config]}"
    local dataset="community"
    
    echo "开始实验配置: $exp_config (GPU: $gpu, Codebook Size: $codebook_size, NC: $nc)"
    
    # 训练自编码器
    echo "[$exp_config] train_autoencoder 开始..."
    if python main.py --work_type train_autoencoder --gpu $gpu --exp_name $exp_name --run_name $exp_config \
        --dataset $dataset --wandb $wandb --codebook_size $codebook_size --nc $nc \
        2>&1 | tee "models_own/$exp_name/$exp_config/${dataset}_autoencoder/$date.log"; then
        copy_wandb_files autoencoder "$exp_config"
    else
        echo "[$exp_config] train_autoencoder 失败!"
        return 1
    fi

    # 训练先验模型
    echo "[$exp_config] train_prior 开始..."
    if python main.py --work_type train_prior --gpu $gpu --exp_name $exp_name --run_name $exp_config \
        --dataset $dataset --wandb $wandb --codebook_size $codebook_size --nc $nc \
        2>&1 | tee "models_own/$exp_name/$exp_config/${dataset}_prior/$date.log"; then
        copy_wandb_files prior "$exp_config"
    else
        echo "[$exp_config] train_prior 失败!"
        return 1
    fi

    # 采样
    echo "[$exp_config] sample 开始..."
    python main.py --work_type sample --gpu $gpu --exp_name $exp_name --run_name $exp_config \
        --dataset $dataset --codebook_size $codebook_size --nc $nc \
        2>&1 | tee "models_own/$exp_name/$exp_config/${dataset}_sample/$date.log"
    echo "[$exp_config] sample 完成!"
}

# 并行运行所有实验配置
declare -A experiment_pids
for exp_config in "${!experiment_configs[@]}"; do
    run_experiment "$exp_config" &
    experiment_pids[$exp_config]=$!
done

# 等待任务完成并显示完成状态
remaining_tasks=${#experiment_pids[@]}
while [ $remaining_tasks -gt 0 ]; do
    for exp_config in "${!experiment_pids[@]}"; do
        if ! kill -0 ${experiment_pids[$exp_config]} 2>/dev/null; then
            if wait ${experiment_pids[$exp_config]}; then
                echo "✅ 实验配置 $exp_config 处理完成"
            else
                echo "❌ 实验配置 $exp_config 处理失败"
            fi
            unset experiment_pids[$exp_config]
            ((remaining_tasks--))
        fi
    done
    sleep 5
done

# # 添加一个串行执行的选项
# PARALLEL=false  # 设置为 true 则并行执行，false 则串行执行

# if [ "$PARALLEL" = true ]; then
#     # 原有的并行执行代码
#     for exp_config in "${!experiment_configs[@]}"; do
#         run_experiment "$exp_config" &
#         experiment_pids[$exp_config]=$!
#     done
#     # ... 等待任务完成的代码 ...
# else
#     # 串行执行
#     for exp_config in "${!experiment_configs[@]}"; do
#         run_experiment "$exp_config"
#     done
# fi