#!/bin/bash

# set -e 命令使得脚本在遇到错误时立即退出
# 这行很有必要,因为它确保了如果脚本中的任何命令失败,整个脚本就会停止执行
# 这可以防止在前面的步骤失败的情况下继续执行后续步骤
set -e

date=$(date +%Y%m%d)

dataset='community'
# dataset='enzymes'
# dataset='qm9'
# dataset='zinc'
# gpu=3
gpu=2
# codebook_size=1024
# nc=1
wandb='offline'

# exp_name='mask_T10_1024_1'
# exp_name='baseline-cb16_2-mlm' #运行bash前在这里设置exp_name
# exp_name='baseline-cb32_2-mlm' #运行bash前在这里设置exp_name
# exp_name='baseline-cb16_2' #运行bash前在这里设置exp_name
# exp_name='baseline-cb256_1' #运行bash前在这里设置exp_name
# exp_name='baseline-cb256_1-mlm' #运行bash前在这里设置exp_name
# exp_name='baseline-cb256_1-mlm-recon_plot' #运行bash前在这里设置exp_name
# exp_name='baseline-cb32_2-mlm-recon_plot' #运行bash前在这里设置exp_name
# exp_name='16_3_test_collapse'
# exp_name='baseline-cb256_1-mlm-reproduce-rewirte'
# exp_name='baseline-cb256_1-reproduce'
exp_name='baseline-cb256_1-mlm_length_test'

mkdir -p models_own/$exp_name/$dataset'_autoencoder'
mkdir -p models_own/$exp_name/$dataset'_prior'
mkdir -p models_own/$exp_name/$dataset'_sample'

copy_wandb_files() {
    local work_type=$1
    local target_dir="models_own/$exp_name/${dataset}_${work_type}"
    
    # 读取特定运行的 run ID
    local run_id=$(cat "$target_dir/wandb_run_id")
    if [ -n "$run_id" ]; then
        # 删除所有非.txt文件
        if [ -d "$target_dir" ]; then
            find "$target_dir" -type f ! -name "*.txt" -delete
        fi
        
        # 复制新文件
        cp -r "./wandb/${run_id}/files"/* "$target_dir"
    else
        echo "警告: 找不到wandb运行ID"
    fi
}

# echo "train_autoencoder start!!!"
# # if python main.py --work_type train_autoencoder --gpu $gpu --exp_name $exp_name --use_mask --dataset $dataset --wandb $wandb 2>&1| tee "models_own/$exp_name/${dataset}_autoencoder/$date.log"; then
# if python main.py --work_type train_autoencoder --gpu $gpu --exp_name $exp_name --use_mask --dataset $dataset --wandb $wandb 2>&1| tee "models_own/$exp_name/${dataset}_autoencoder/$date.log"; then
#     copy_wandb_files autoencoder
# else
#     echo "train_autoencoder failed!!!"
#     exit 1
# fi

echo "train_prior start!!!"
# if python main.py --work_type train_prior --gpu $gpu --exp_name $exp_name --dataset $dataset --use_mask --wandb $wandb --codebook_size $codebook_size --nc $nc 2>&1 | tee 'models_own/'$exp_name'/'$dataset'_prior/'$date'.log'; then
if python main.py --work_type train_prior --gpu $gpu --exp_name $exp_name --dataset $dataset --use_mask --wandb $wandb 2>&1 | tee "models_own/$exp_name/${dataset}_prior/$date.log"; then
    copy_wandb_files prior
else
    echo "train_prior failed!!!"
    exit 1
fi

echo "sample start!!!"
python main.py --work_type sample --gpu $gpu --exp_name $exp_name --dataset $dataset --use_mask 2>&1 | tee "models_own/$exp_name/${dataset}_sample/$date.log"
echo "sample end!!!"
