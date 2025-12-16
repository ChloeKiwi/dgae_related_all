#!/bin/bash

# set -e 命令使得脚本在遇到错误时立即退出
# 这行很有必要,因为它确保了如果脚本中的任何命令失败,整个脚本就会停止执行
set -e

date=$(date +%Y%m%d)
# exp_name='baseline-cb16_2-not_sorted' #运行bash前在这里设置exp_name
# exp_name='baseline-cb16_2-mlm' #运行bash前在这里设置exp_name
# exp_name='baseline-cb256_1' #运行bash前在这里设置exp_name
# exp_name='baseline-cb1024_1' #运行bash前在这里设置exp_name
exp_name='baseline-cb16_2' #运行bash前在这里设置exp_name
dataset='community'
# dataset='enzymes'
gpu=0

echo "sample start!!!"
python main.py --work_type sample --gpu $gpu --exp_name $exp_name --dataset $dataset 2>&1 | tee 'models_own/'$exp_name'/'$dataset'_sample/'$date'.log'
echo "sample end!!!"

exp_name='baseline-cb256_1' #运行bash前在这里设置exp_name
dataset='community'
# dataset='enzymes'
gpu=0

echo "sample start!!!"
python main.py --work_type sample --gpu $gpu --exp_name $exp_name --dataset $dataset 2>&1 | tee 'models_own/'$exp_name'/'$dataset'_sample/'$date'.log'
echo "sample end!!!"
