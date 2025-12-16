#!/bin/bash
set -e

copy_wandb_files() {
    local work_type=$1
    # 读取特定运行的 run ID
    local run_id=$(cat "./models_own/$exp_name/$run_name/${dataset}_${work_type}/wandb_run_id")
    if [ -n "$run_id" ]; then
        # echo "复制 ${work_type} 的wandb文件 (Run ID: ${run_id})..."
        if [ -d "models_own/$exp_name/$run_name/${dataset}_${work_type}" ]; then
            rm -rf "models_own/$exp_name/$run_name/${dataset}_${work_type}"/*
        fi
        cp -r "./wandb/${run_id}/files"/* "./models_own/$exp_name/$run_name/${dataset}_${work_type}"
        # rm "./models_own/$exp_name/$run_name/${dataset}_${work_type}/wandb_run_id"  # 清理临时文件
    else
        echo "警告: 找不到wandb运行ID"
    fi
}

date=$(date +%Y%m%d)
wandb='online'

exp_name='dgae_baseline'
run_name='20250116_repro'

exp_name='mlpvae'
# run_name='community_before_post'
run_name='community_before_post_complex'
# exp_name='dgae_baseline_different_hyper'
# run_name='community_initsteps100'
# run_name='community_cb256'
# run_name='grid_cb64_initsteps100'
# run_name='grid_cb64_initsteps0'

# exp_name='dage_baseline_debug'
# run_name='node_feat'
# 定义数据集和对应的GPU
declare -A dataset_gpu=(
    # ["grid"]="0"
    # ["ego"]="0"
    # ["citeseer"]="3"
    ["community"]="2"
    # ['community_ego']=0
    # ["enzymes"]="3"
    # ["qm9"]="1"
    # ["zinc"]="1"
    # ["planar"]="0"
    # ["sbm"]="1"
    # ["protein"]="2"
)

# 定义一个数组来存储每个数据集的进程ID
declare -A dataset_pids

# 创建所有必要的目录
for dataset in "${!dataset_gpu[@]}"; do
    mkdir -p models_own/$exp_name/$run_name/${dataset}_autoencoder
    mkdir -p models_own/$exp_name/$run_name/${dataset}_prior
    mkdir -p models_own/$exp_name/$run_name/${dataset}_sample
done

# 定义运行单个数据集的所有步骤的函数
run_dataset() {
    local dataset=$1
    local gpu=$2
    
    echo "开始数据集: $dataset (GPU: $gpu)"
    
    # 训练自编码器
    # echo "[$dataset] train_autoencoder 开始..."
    # if python main.py --work_type train_autoencoder --gpu $gpu --exp_name $exp_name --run_name $run_name --dataset $dataset --wandb $wandb 2>&1| tee "models_own/$exp_name/$run_name/${dataset}_autoencoder/$date.log"; then
    #     copy_wandb_files autoencoder
    # else
    #     echo "[$dataset] train_autoencoder 失败!"
    #     return 1
    # fi

    # # 训练先验模型
    echo "[$dataset] train_prior 开始..."
    if python main.py --work_type train_prior --gpu $gpu --exp_name $exp_name --run_name $run_name --dataset $dataset --wandb $wandb 2>&1 | tee "models_own/$exp_name/$run_name/${dataset}_prior/$date.log"; then
        copy_wandb_files prior
    else
        echo "[$dataset] train_prior 失败!"
        return 1
    fi

    # 采样
    echo "[$dataset] sample 开始..."
    python main.py --work_type sample --gpu $gpu --exp_name $exp_name --run_name $run_name --dataset $dataset 2>&1 | tee "models_own/$exp_name/$run_name/${dataset}_sample/$date.log"
    echo "[$dataset] sample 完成!"
}

# 并行运行所有数据集
for dataset in "${!dataset_gpu[@]}"; do
    gpu=${dataset_gpu[$dataset]}
    run_dataset "$dataset" "$gpu" &
    dataset_pids[$dataset]=$!  # 保存每个数据集对应的进程ID
done

# 等待任务完成并显示完成状态
remaining_tasks=${#dataset_pids[@]}
while [ $remaining_tasks -gt 0 ]; do
    for dataset in "${!dataset_pids[@]}"; do
        if ! kill -0 ${dataset_pids[$dataset]} 2>/dev/null; then
            if wait ${dataset_pids[$dataset]}; then
                echo "✅ 数据集 $dataset 处理完成"
            else
                echo "❌ 数据集 $dataset 处理失败"
            fi
            unset dataset_pids[$dataset]
            ((remaining_tasks--))
        fi
    done
    sleep 5  # 避免过于频繁的检查
done