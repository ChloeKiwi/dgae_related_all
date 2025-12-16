#!/bin/bash

set -e

date=$(date +%Y%m%d)

dataset='community'
# dataset='enzymes'
# dataset='qm9'
# dataset='zinc'
# dataset='ego'
gpu=3
# gpu=0
# gpu=1
# gpu=3
# codebook_size=1024
# codebook_size=64 #记得在命令行添加--codebook_size $codebook_size
# codebook_size=4096 #记得在命令行添加--codebook_size $codebook_size
codebook_size=256 #记得在命令行添加--codebook_size $codebook_size
# codebook_size=16 #记得在命令行添加--codebook_size $codebook_size
nc=1
# nc=2
wandb='online'

# exp='mask_T10'
# exp='mask_T30' #for enzymes
# exp_name="${exp}_${codebook_size}_${nc}"
exp_name='mask_T10_1024_1'
# exp_name='baseline-cb16_2-mlm' #运行bash前在这里设置exp_name
# exp_name='baseline-cb32_2-mlm' #运行bash前在这里设置exp_name
# exp_name='baseline-cb16_2' #运行bash前在这里设置exp_name
# exp_name='baseline-cb256_1' #运行bash前在这里设置exp_name
# exp_name='baseline-cb256_1-mlm' #运行bash前在这里设置exp_name
# exp_name='baseline-cb256_1-mlm-recon_plot' #运行bash前在这里设置exp_name
# exp_name='baseline-cb32_2-mlm-recon_plot' #运行bash前在这里设置exp_name
# exp_name='mlm_benchmark'
# exp_name='baseline-cb8_2-mlm'
# exp_name='mask_recon_nodes_test' #没有修改node loss处理版
# exp_name='mask_recon_nodes_sigmoid' #修改node loss为sigmoid处理版
# exp_name='256_collaps_exp' #测试4096codebook_size的实验
# exp_name='4096_collaps_exp' #测试4096codebook_size的实验
# exp_name='enzymes_collaps_exp' #测试4096codebook_size的实验
# exp_name='enzymes_collaps_exp_4096_1' #测试4096codebook_size的实验
# exp_name='not_sorted_indices' #测试不排序indices的实验
# exp_name='not_sorted_indices_mlm' #测试不排序indices的实验
# exp_name='not_sorted_indices_no_vq' #测试不排序indices的实验
# exp_name='not_sorted_indices_no_vq_mlm' #测试不排序indices的实验
# exp_name='sort_w_vq'
# exp_name='sort_wo_vq_2'
# exp_name='sort_w_vq_mlm_2'
# exp_name='sort_wo_vq_mlm'
# exp_name='eu_distance_sqrt'
# exp_name='cosine_distance'
# exp_name='euclidean_distance'
# exp_name='vae_recon2'
# exp_name='klvae_debug'
# exp_name='mlpvae'
# exp_name='mlpvae_gamma0.1'
exp_name='mlpvae_gamma16'

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
# mlm类记得 --use_mask
echo "train_autoencoder start!!!"
if python main.py --work_type train_autoencoder --gpu $gpu --exp_name $exp_name --dataset $dataset --use_mask --wandb $wandb --codebook_size $codebook_size --nc $nc 2>&1| tee "models_own/$exp_name/$dataset_autoencoder/$date.log"; then
# if python main.py --work_type train_autoencoder --gpu $gpu --exp_name $exp_name --dataset $dataset --wandb $wandb --codebook_size $codebook_size --nc $nc --quantization $quantization 2>&1| tee "models_own/$exp_name/$dataset_autoencoder/$date.log"; then
    copy_wandb_files autoencoder
else
    echo "train_autoencoder failed!!!"
    exit 1
fi
# echo "train_prior start!!!"
# if python main.py --work_type train_prior --gpu $gpu --exp_name $exp_name --dataset $dataset --use_mask --wandb $wandb --codebook_size $codebook_size --nc $nc 2>&1 | tee "models_own/$exp_name/$dataset'_prior/'$date'.log"; then
# # if python main.py --work_type train_prior --gpu $gpu --exp_name $exp_name --dataset $dataset --wandb $wandb --codebook_size $codebook_size --nc $nc 2>&1 | tee "models_own/$exp_name/$dataset'_prior/'$date'.log"; then
#     copy_wandb_files prior
# else
#     echo "train_prior failed!!!"
#     exit 1
# fi
# echo "sample start!!!"
# python main.py --work_type sample --gpu $gpu --exp_name $exp_name --dataset $dataset --use_mask 2>&1 | tee "models_own/$exp_name/$dataset'_sample/'$date'.log"
# # python main.py --work_type sample --gpu $gpu --exp_name $exp_name --dataset $dataset 2>&1 | tee "models_own/$exp_name/$dataset'_sample/'$date'.log"
# echo "sample end!!!"
