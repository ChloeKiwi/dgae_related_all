#!/bin/bash

# set -e 命令使得脚本在遇到错误时立即退出
# 这行很有必要,因为它确保了如果脚本中的任何命令失败,整个脚本就会停止执行
set -e

date=$(date +%Y%m%d)
# exp_name='baseline-cb16_2-not_sorted' #运行bash前在这里设置exp_name
# exp_name='baseline-cb16_2-mlm' #运行bash前在这里设置exp_name
# exp_name='baseline-cb256_1' #运行bash前在这里设置exp_name
exp_name='baseline-cb1024_1' #运行bash前在这里设置exp_name
# exp_name='baseline-cb256_1-reproduce'
# dataset='community'
dataset='enzymes'
gpu=0

mkdir -p models_own/$exp_name/$dataset'_autoencoder'
mkdir -p models_own/$exp_name/$dataset'_prior'
mkdir -p models_own/$exp_name/$dataset'_sample'

copy_wandb_files() {
    local work_type=$1
    # 读取特定运行的 run ID
    local run_id=$(cat "./models_own/$exp_name/${dataset}_${work_type}/wandb_run_id")
    if [ -n "$run_id" ]; then
        # echo "复制 ${work_type} 的wandb文件 (Run ID: ${run_id})..."
        if [ -d "models_own/$exp_name/${dataset}_${work_type}" ]; then
            rm -rf "models_own/$exp_name/${dataset}_${work_type}"/*
        fi
        cp -r "./wandb/${run_id}/files"/* "./models_own/$exp_name/${dataset}_${work_type}"
        # rm "./models_own/$exp_name/${dataset}_${work_type}/wandb_run_id"  # 清理临时文件
    else
        echo "警告: 找不到wandb运行ID"
    fi
}

# echo "train_autoencoder start!!!"
# if python main.py --work_type train_autoencoder --gpu $gpu --exp_name $exp_name --dataset $dataset 2>&1| tee 'models_own/'$exp_name'/'$dataset'_autoencoder/'$date'.log'; then
#     copy_wandb_files autoencoder
# else
#     echo "train_autoencoder failed!!!"
#     exit 1
# fi

echo "train_prior start!!!"
if python main.py --work_type train_prior --gpu $gpu --exp_name $exp_name --dataset $dataset 2>&1 | tee "models_own/$exp_name/$dataset_prior/$date.log"; then
    copy_wandb_files prior
else
    echo "train_prior failed!!!"
    exit 1
fi

echo "sample start!!!"
python main.py --work_type sample --gpu $gpu --exp_name $exp_name --dataset $dataset 2>&1 | tee "models_own/$exp_name/$dataset_sample/$date.log"
echo "sample end!!!"