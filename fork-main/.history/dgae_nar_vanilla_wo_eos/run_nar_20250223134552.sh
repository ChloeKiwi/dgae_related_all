#!/bin/bash
# set -e

date=$(date +%Y%m%d-%H%M)
wandb='online'
# wandb='disabled'

# exp_name='dgae_baseline_different_hyper'
# exp_name='dgae_baseline_different_hyper'
# exp_name='mlm_moc_baseline'
# exp_name='citeseer_r4'
# run_name='community_cb_compare'
# exp_name='community_mlm_base'
exp_name='nar_vanilla'
exp_name='nar_vanilla'
# exp_name='nar_fix_loss'
# exp_name='mask_unmask_loss'

# 定义实验配置，现在包含数据集信息
declare -A experiment_configs=(
    # ['community_16_2_not_ignore_pad_linear']="community 16 2 0 linear"
    # ['community_16_2_not_ignore_pad_cosine']="community 16 2 1 cosine"
    # ['community_128_1_not_ignore_pad_linear_no_sort']="community 128 1 3 linear"
    # ['community_128_1_not_ignore_pad_cosine']="community 128 1 2 cosine"
    # ['community_256_1_not_ignore_pad_linear']="community 256 1 1 linear"
    # ['community_256_1_not_ignore_pad_cosine']="community 256 1 0 cosine"
    # ['community_16_2_not_ignore_pad_cosine']="community 16 2 1"
    # ['community_256_1']="community 256 1 2"
    # ['enzymes_512_1_linear']="enzymes 512 1 0 linear"
    # ['community_128_1_cosine_step_mask']='community 128 1 1 cosine'
    # ['community_128_1_cosine_0.5']='community 128 1 3 cosine 0.5'
    # ['community_64_cosine_0.5']='community 64 1 0 cosine 0.5'
    # ['community_64_linear_0.5']='community 64 1 1 linear 0.5'
    ['community_64_1_not_ignore_pad_linear']='community 64 1 0 linear 0.5'
    ['community_64_1_not_ignore_pad_cosine']='community 64 1 1 cosine 0.5'
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
    read -r dataset _ _ _ _ _ <<< "${experiment_configs[$exp_config]}"
    mkdir -p models_own/$exp_name/$exp_config/${dataset}_autoencoder
    mkdir -p models_own/$exp_name/$exp_config/${dataset}_prior
    mkdir -p models_own/$exp_name/$exp_config/${dataset}_sample
done

# 修改运行实验的函数
run_experiment() {
    local exp_config=$1
    read -r dataset codebook_size nc gpu mask_func iterations_rate <<< "${experiment_configs[$exp_config]}"
    
    echo "开始实验配置: $exp_config (数据集: $dataset, GPU: $gpu, Codebook Size: $codebook_size, NC: $nc, Mask Function: $mask_func, Iterations Rate: $iterations_rate)"
   
    # echo "[$exp_config] train_autoencoder 开始..."
    # python main.py --work_type train_autoencoder --gpu $gpu --exp_name $exp_name --run_name $exp_config --dataset $dataset --wandb $wandb --codebook_size $codebook_size --nc $nc 2>&1 | tee "models_own/$exp_name/$exp_config/${dataset}_autoencoder/$date.log"
    # echo "[$exp_config] train_autoencoder 完成"

    echo "[$exp_config] train_prior 开始..."
    python main.py --work_type train_prior --gpu $gpu --exp_name $exp_name --run_name $exp_config --dataset $dataset --wandb $wandb --codebook_size $codebook_size --nc $nc --use_mask --mask_func $mask_func --iterations_rate $iterations_rate 2>&1 | tee "models_own/$exp_name/$exp_config/${dataset}_prior/$date.log"
    echo "[$exp_config] train_prior 完成"


    echo "[$exp_config] sample 开始..."
    python main.py --work_type sample --gpu $gpu --exp_name $exp_name --run_name $exp_config --dataset $dataset  --wandb $wandb --codebook_size $codebook_size --nc $nc --use_mask --mask_func $mask_func --iterations_rate $iterations_rate 2>&1 | tee "models_own/$exp_name/$exp_config/${dataset}_sample/$date.log"
    echo "[$exp_config] sample 完成!"
}

# 并行运行所有实验配置
declare -A experiment_pids
for exp_config in "${!experiment_configs[@]}"; do
    run_experiment "$exp_config" &
    experiment_pids[$exp_config]=$!
done

# 等待所有任务完成
for pid in "${experiment_pids[@]}"; do
    wait $pid
done

# 显示完成状态
for exp_config in "${!experiment_configs[@]}"; do
    if wait ${experiment_pids[$exp_config]}; then
        echo "✅ 实验配置 $exp_config 处理完成"
    else
        echo "❌ 实验配置 $exp_config 处理失败"
    fi
done

#! debug mode
# run_experiment 'community_128_1_cosine_0.5'
# for exp_config in "${!experiment_configs[@]}"; do
#     run_experiment "$exp_config"   
# done