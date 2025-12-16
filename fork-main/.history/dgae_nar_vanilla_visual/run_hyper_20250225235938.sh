#!/bin/bash
set -e

copy_wandb_files() {
    local work_type=$1
    local exp_config=$2
    local dataset=$3  # 修改：添加 dataset 参数
    # 读取特定运行的 run ID
    local run_id=$(cat "./models_own/$exp_name/$exp_config/${dataset}_${work_type}/wandb_run_id")
    if [ -n "$run_id" ]; then
        if [ -d "models_own/$exp_name/$exp_config/${dataset}_${work_type}" ]; then
            rm -rf "models_own/$exp_name/$exp_config/${dataset}_${work_type}"/*
        fi
        cp -r "./wandb/${run_id}/files"/* "./models_own/$exp_name/$exp_config/${dataset}_${work_type}"
    else
        echo "警告: 找不到wandb运行ID"
    fi
}

date=$(date +%Y%m%d)
wandb='online' #! train ae和prior必须使用wandb
# wandb='disabled'

# exp_name='dgae_baseline_different_hyper'
# exp_name='dgae_baseline_different_hyper'
# exp_name='mlm_moc_baseline'
# exp_name='citeseer_r4'
# run_name='community_cb_compare'
# exp_name='community_mlm_base'
exp_name='nar_vanilla_visual'

# 定义实验配置，现在包含数据集信息
declare -A experiment_configs=(
    # community数据集的配置
    # ["community_cb128_1"]="community 128 1 0"    # 新格式: "dataset codebook_size nc gpu"
    # ["community_cb32_2"]="community 32 2 0"
    # ["community_cb64_2"]="community 64 2 1"
    # ["community_cb256_2"]="community 256 2 1"
    # ['community_16_2_greedy']="community 16 2 2"
    # ['community_16_2']="community 16 2 0"
    # ['community_16_2_ignore_pad']="community 16 2 2"
    # ['community_16_2_not_ignore_pad_linear']="community 16 2 0 linear"
    # ['community_16_2_not_ignore_pad_cosine']="community 16 2 1 cosine"
    # ['community_128_1_not_ignore_pad_linear']="community 128 1 0 linear"
    # ['community_128_1_not_ignore_pad_cosine']="community 128 1 2 cosine"
    ['community_64_1_not_ignore_pad_cosine']=""
    # ['community_256_1_not_ignore_pad_linear']="community 256 1 1 linear"
    # ['community_256_1_not_ignore_pad_cosine']="community 256 1 0 cosine"
    # ['community_16_2_not_ignore_pad_cosine']="community 16 2 1"
    # ['community_256_1']="community 256 1 2"
    # grid数据集的配置
    # ["grid_cb64_2"]="grid 64 2 1"
    # ["grid_cb128_2"]="grid 128 2 0"
    # ["grid_cb256_2"]="grid 256 2 0"
    # ['grid_cb512_2']="grid 512 2 1"
    # ["grid_cb1024_2"]="grid 1024 2 1"
    # # 可以添加其他数据集的配置
    # ["protein_cb64_2"]="protein 64 2 1"
    # ["protein_cb128_2"]="protein 128 2 0"
    # ["protein_cb1024_2"]='protein 1024 2 2'
    # ["protein_cb512_2"]='protein 512 2 2'
    # ['citeseer_cb8_2']="citeseer 8 2 1"
    # ['citeseer_cb16_2']="citeseer 16 2 1"
    # ['citeseer_cb16_2']="citeseer 16 2 1"
    # ['qm9_cb16_2']="qm9 16 2 1"
    # ['zinc_cb32_2']="zinc 32 2 3"

)

# 创建所有必要的目录
for exp_config in "${!experiment_configs[@]}"; do
    read -r dataset _ _ _ _ <<< "${experiment_configs[$exp_config]}"
    mkdir -p models_own/$exp_name/$exp_config/${dataset}_autoencoder
    mkdir -p models_own/$exp_name/$exp_config/${dataset}_prior
    mkdir -p models_own/$exp_name/$exp_config/${dataset}_sample
done

# 修改运行实验的函数
run_experiment() {
    local exp_config=$1
    read -r dataset codebook_size nc gpu mask_func <<< "${experiment_configs[$exp_config]}"
    
    echo "开始实验配置: $exp_config (数据集: $dataset, GPU: $gpu, Codebook Size: $codebook_size, NC: $nc, Mask Function: $mask_func)"
   
    # # # 训练自编码器
    # echo "[$exp_config] train_autoencoder 开始..."
    # if python main.py --work_type train_autoencoder --gpu $gpu --exp_name $exp_name --run_name $exp_config \
    #     --dataset $dataset --wandb $wandb --codebook_size $codebook_size --nc $nc \
    #     2>&1 | tee "models_own/$exp_name/$exp_config/${dataset}_autoencoder/$date.log"; then
    #     copy_wandb_files autoencoder "$exp_config" "$dataset"
    # else
    #     echo "[$exp_config] train_autoencoder 失败!"
    #     return 1
    # fi

    # 训练先验模型
    # echo "[$exp_config] train_prior 开始..."
    # if python main.py --work_type train_prior --gpu $gpu --exp_name $exp_name --run_name $exp_config --dataset $dataset --wandb $wandb --codebook_size $codebook_size --nc $nc --use_mask --mask_func $mask_func 2>&1 | tee "models_own/$exp_name/$exp_config/${dataset}_prior/$date.log"; then
    #     copy_wandb_files prior "$exp_config" "$dataset"
    # else
    #     echo "[$exp_config] train_prior 失败!"
    #     return 1
    # fi

    # 采样
    echo "[$exp_config] sample 开始..."
    python main.py --work_type sample --gpu $gpu --exp_name $exp_name --run_name $exp_config \
        --dataset $dataset --codebook_size $codebook_size --nc $nc --use_mask --mask_func $mask_func \
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