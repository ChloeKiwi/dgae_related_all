# # !/bin/bash

# # set -e 命令使得脚本在遇到错误时立即退出
# # 这行很有必要,因为它确保了如果脚本中的任何命令失败,整个脚本就会停止执行
# # 这可以防止在前面的步骤失败的情况下继续执行后续步骤
# # set -e

# date=$(date +%Y%m%d)

# dataset='community'
# # dataset='enzymes'
# # dataset='qm9'
# # dataset='zinc'
# # gpu=0
# # gpu=1
# # gpu=2
# gpu=3       
# # codebook_size=1024
# # nc=1
# wandb='online'

# # exp_name='mask_T10_1024_1'
# # exp_name='baseline-cb16_2-mlm' #运行bash前在这里设置exp_name
# # exp_name='baseline-cb32_2-mlm' #运行bash前在这里设置exp_name
# # exp_name='baseline-cb16_2' #运行bash前在这里设置exp_name
# # exp_name='baseline-cb256_1' #运行bash前在这里设置exp_name
# # exp_name='baseline-cb256_1-mlm' #运行bash前在这里设置exp_name
# # exp_name='baseline-cb256_1-mlm-recon_plot' #运行bash前在这里设置exp_name
# # exp_name='baseline-cb32_2-mlm-recon_plot' #运行bash前在这里设置exp_name
# # exp_name='16_3_test_collapse'
# # exp_name='baseline-cb256_1-mlm-reproduce'
# # exp_name='baseline-cb256_1-reproduce'
# # exp_name='baseline_enzymes_mlm' #32_2
# # exp_name='baseline_cb256_1_mlm-use_codebook_embedding' #community prior use codebook embedding, instead of tok_emb
# # exp_name='baseline_qm9_mlm'
# # exp_name='baseline_zinc_mlm'
# # exp_name='baseline_cb256_1_mlm_codebook_emb'
# # exp_name='baseline_cb256_1_mlm_repro'
# # exp_name='baseline_cb256_1_mlm_repro_more_layers_128_8_6' #transformer多几层参数
# # exp_name='baseline_cb256_1_mlm_repro_more_layers_256' #transformer多几层参数
# # exp_name='baseline_cb256_1_mlm_repro_weightying'
# exp_name='baseline_cb256_1_mlm_no_feature_aug'

# run_name=''


# mkdir -p models_own/$exp_name/$run_name/$dataset'_autoencoder'
# mkdir -p models_own/$exp_name/$run_name/$dataset'_prior'
# mkdir -p models_own/$exp_name/$run_name/$dataset'_sample'

# copy_wandb_files() {
#     local work_type=$1
#     # 读取特定运行的 run ID
#     local run_id=$(cat "./models_own/$exp_name/$run_name/${dataset}_${work_type}/wandb_run_id")
#     if [ -n "$run_id" ]; then
#         # echo "复制 ${work_type} 的wandb文件 (Run ID: ${run_id})..."
#         if [ -d "models_own/$exp_name/$run_name/${dataset}_${work_type}" ]; then
#             rm -rf "models_own/$exp_name/$run_name/${dataset}_${work_type}"/*
#         fi
#         cp -r "./wandb/${run_id}/files"/* "./models_own/$exp_name/$run_name/${dataset}_${work_type}"
#         # rm "./models_own/$exp_name/$run_name/${dataset}_${work_type}/wandb_run_id"  # 清理临时文件
#     else
#         echo "警告: 找不到wandb运行ID"
#     fi
# }

# echo "train_autoencoder start!!!"
# # if python main.py --work_type train_autoencoder --gpu $gpu --exp_name $exp_name --dataset $dataset --use_mask --wandb $wandb --codebook_size $codebook_size --nc $nc 2>&1| tee 'models_own/'$exp_name'/'$dataset'_autoencoder/'$date'.log'; then
# if python main.py --work_type train_autoencoder --gpu $gpu --exp_name $exp_name --run_name $run_name --use_mask --dataset $dataset --wandb $wandb 2>&1| tee "models_own/$exp_name/$run_name/$date_train_autoencoder.log"; then
#     copy_wandb_files autoencoder
# else
#     echo "train_autoencoder failed!!!"
#     exit 1
# fi

# echo "train_prior start!!!"
# # if python main.py --work_type train_prior --gpu $gpu --exp_name $exp_name --dataset $dataset --use_mask --wandb $wandb --codebook_size $codebook_size --nc $nc 2>&1 | tee 'models_own/'$exp_name'/'$dataset'_prior/'$date'.log'; then
# if python main.py --work_type train_prior --gpu $gpu --exp_name $exp_name --run_name $run_name --use_mask --dataset $dataset --wandb $wandb 2>&1 | tee "models_own/$exp_name/$run_name/$date_train_prior.log"; then
#     copy_wandb_files prior
# else
#     echo "train_prior failed!!!"
#     exit 1
# fi

# echo "sample start!!!"
# python main.py --work_type sample --gpu $gpu --exp_name $exp_name --run_name $run_name --use_mask --dataset $dataset 2>&1 | tee "models_own/$exp_name/$run_name/$date_sample.log"
# echo "sample end!!!"

#!/bin/bash
set -e

date=$(date +%Y%m%d)
wandb='online'
# exp_name='dgae_baseline_different_hyper'
# exp_name='dgae_baseline'
exp_name='dgae_mlm'
run_name='community_16'

# 定义数据集和对应的GPU
declare -A dataset_gpu=(
    ["community"]="3"
    # ["enzymes"]="2"
    # ["qm9"]="0"
    # ["zinc"]="1"
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
    echo "[$dataset] train_autoencoder 开始..."
    if python main.py --work_type train_autoencoder --gpu $gpu --exp_name $exp_name --run_name $run_name --dataset $dataset --use_mask --wandb $wandb 2>&1| tee "models_own/$exp_name/$run_name/${dataset}_autoencoder/$date.log"; then
        copy_wandb_files autoencoder
    else
        echo "[$dataset] train_autoencoder 失败!"
        return 1
    fi

    # 训练先验模型
    echo "[$dataset] train_prior 开始..."
    if python main.py --work_type train_prior --gpu $gpu --exp_name $exp_name --run_name $run_name --dataset $dataset --use_mask --wandb $wandb 2>&1 | tee "models_own/$exp_name/$run_name/${dataset}_prior/$date.log"; then
        copy_wandb_files prior
    else
        echo "[$dataset] train_prior 失败!"
        return 1
    fi

    # 采样
    echo "[$dataset] sample 开始..."
    python main.py --work_type sample --gpu $gpu --exp_name $exp_name --run_name $run_name --use_mask --dataset $dataset 2>&1 | tee "models_own/$exp_name/$run_name/${dataset}_sample/$date.log"
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
    sleep 1  # 避免过于频繁的检查
done