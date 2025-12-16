# !/bin/bash
set -e

date=$(date +%Y%m%d-%H%M)
# wandb='online'
wandb='disabled'

exp_name='icdm'

# 定义实验配置，现在包含数据集信息
declare -A experiment_configs=(
    # ['community_128_1_not_ignore_pad_linear_no_sort']="community 128 1 3 linear"
    # ['community_128_1_not_ignore_pad_cosine']="community 128 1 0 cosine"
    # ['community_256_1_not_ignore_pad_linear']="community 256 1 1 linear"
    # ['community_256_1_not_ignore_pad_cosine']="community 256 1 0 cosine"
    # ['community_256_1']="community 256 1 2"
    # ['enzymes_512_1_cosine_0.7']="enzymes 512 1 0 cosine 0.7"
    # ['planar_512_1_cosine_0.5']="planar 512 1 2 cosine 0.5"
    # ['community_128_1_cosine_step_mask']='community 128 1 1 cosine'
    # ['community_128_1_cosine_0.5']='community 128 1 3 cosine 0.5'
    # ['community_64_cosine_0.5']='community 64 1 0 cosine 0.5'
    # ['community_64_cosine_0.1']='community 64 1 2 cosine 0.1'
    # ['community_64_cosine_0.3']='community 64 1 3 cosine 0.3'
    # ['community_64_cosine_0.7']='community 64 1 0 cosine 0.7'
    # ['community_64_cosine_0.9']='community 64 1 1 cosine 0.9'
    # ['community_64_cosine_0.4']='community 64 1 1 cosine 0.4'
    # ['community_64_cosine_0.5']='community 64 1 2 cosine 0.5'
    # ['community_64_cosine_1']='community 64 1 2 cosine 1'
    # ['community_64_linear_0.5']='community 64 1 1 linear 0.5'
    # ['community_64_1_not_ignore_pad_linear']='community 64 1 0 linear 0.5'
    # ['community_64_1_not_ignore_pad_cosine']='community 64 1 2 cosine 0.5'
    # grid数据集的配置
    # ['grid_512_1_cosine_0.1']='grid 512 1 3 cosine 0.1'
    # icdm larger datasets
    # ['grid_1k_1024_1_cosine_1']='grid_1k 1024 1 2 cosine 1'
    # ['grid_1k_2048_1_cosine_1']='grid_1k 2048 1 2 cosine 1'
    # ['grid_1k_4096_1_cosine_1']='grid_1k 4096 1 2 cosine 1'

    # ['circular_ladder_1024_1_cosine_1']='circular_ladder 1024 1 3 cosine 1'
    # ['cycle_1024_1_cosine_1']='cycle 1024 1 2 cosine 1'

    # ['ba_512_1_cosine_1']='ba 512 1 0 cosine 1'
    # # ['gnp_512_1_cosine_1']='gnp 512 1 1 cosine 1'

    # ['circular_ladder_2048_1_cosine_1']='circular_ladder 2048 1 3 cosine 1'
    # ['cycle_2048_1_cosine_1']='cycle 2048 1 0 cosine 1'

    # ['ba_1024_1_cosine_1']='ba 1024 1 1 cosine 1'
    # ['gnp_512_1_cosine_1']='gnp 512 1 1 cosine 1'

)

for exp_config in "${!experiment_configs[@]}"; do
    read -r dataset _ _ _ _ _ <<< "${experiment_configs[$exp_config]}"
    mkdir -p models_own/$exp_name/$exp_config/${dataset}_autoencoder
    mkdir -p models_own/$exp_name/$exp_config/${dataset}_prior
    mkdir -p models_own/$exp_name/$exp_config/${dataset}_sample
done


run_experiment() {
    local exp_config=$1
    read -r dataset codebook_size nc gpu mask_func iterations_rate <<< "${experiment_configs[$exp_config]}"
    
    echo ": $exp_config (dataset: $dataset, GPU: $gpu, Codebook Size: $codebook_size, NC: $nc, Mask Function: $mask_func, Iterations Rate: $iterations_rate)"
   
    echo "[$exp_config] train_autoencoder begin..."
    python main.py --work_type train_autoencoder --gpu $gpu --exp_name $exp_name --run_name $exp_config --dataset $dataset --wandb $wandb --codebook_size $codebook_size --nc $nc 2>&1 | tee "models_own/$exp_name/$exp_config/${dataset}_autoencoder/$date.log"
    echo "[$exp_config] train_autoencoder done"

#     echo "[$exp_config] train_prior begin..."
#     python main.py --work_type train_prior --gpu $gpu --exp_name $exp_name --run_name $exp_config --dataset $dataset --wandb $wandb --codebook_size $codebook_size --nc $nc --use_mask --mask_func $mask_func --iterations_rate $iterations_rate 2>&1 | tee "models_own/$exp_name/$exp_config/${dataset}_prior/$date.log"
#     echo "[$exp_config] train_prior done"


#     echo "[$exp_config] sample 开始..."
#     python main.py --work_type sample --gpu $gpu --exp_name $exp_name --run_name $exp_config --dataset $dataset  --wandb $wandb --codebook_size $codebook_size --nc $nc --use_mask --mask_func $mask_func --iterations_rate $iterations_rate 2>&1 | tee "models_own/$exp_name/$exp_config/${dataset}_sample/$date.log"
#     echo "[$exp_config] sample 完成!"
}

declare -A experiment_pids
for exp_config in "${!experiment_configs[@]}"; do
    run_experiment "$exp_config" &
    experiment_pids[$exp_config]=$!
done

for pid in "${experiment_pids[@]}"; do
    wait $pid
done

for exp_config in "${!experiment_configs[@]}"; do
    if wait ${experiment_pids[$exp_config]}; then
        echo "✅ 实验配置 $exp_config 处理完成"
    else
        echo "❌ 实验配置 $exp_config 处理失败"
    fi
done