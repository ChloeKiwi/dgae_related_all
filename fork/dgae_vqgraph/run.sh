#!/bin/bash

date=$(date +%Y%m%d)
exp_name='initcodebook-eucodebook-mlpdecoder_wo_graph_layer2' #运行bash前在这里设置exp_name
dataset='community'

mkdir -p models_own/$exp_name/$dataset'_autoencoder'/files
mkdir -p models_own/$exp_name/$dataset'_prior'/files
mkdir -p models_own/$exp_name/$dataset'_sample'/files

echo "train_autoencoder start!!!"
# python main.py --work_type train_autoencoder 2>&1 | tee 'models_own/enzymes_autoencoder/files/train_autoencoder_'$date'.log'
# 添加错误检查
if ! python main.py --work_type train_autoencoder --gpu 3 --exp_name $exp_name --dataset $dataset 2>&1| tee 'models_own/'$exp_name'/'$dataset'_autoencoder/files/train_autoencoder_'$date'.log'; then
    echo "train_autoencoder failed!!!"
    exit 1
fi

echo "train_prior start!!!"
# python main.py --work_type train_prior 2>&1 | tee 'models_own/enzymes_prior/files/train_prior_'$date'.log'
# 添加错误检查
if ! python main.py --work_type train_prior --gpu 3 --exp_name $exp_name --dataset $dataset 2>&1 | tee 'models_own/'$exp_name'/'$dataset'_prior/files/train_prior_'$date'.log'; then
    echo "train_prior failed!!!"
    exit 1
fi

echo "sample start!!!"
python main.py --work_type sample --gpu 3 --exp_name $exp_name --dataset $dataset 2>&1 | tee 'models_own/'$exp_name'/'$dataset'_sample/files/sample_'$date'.log'
echo "sample end!!!"
