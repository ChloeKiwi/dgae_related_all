#!/bin/bash
set -e

date=$(date +%Y%m%d)
wandb='online'
# wandb='disabled'

# exp_name='dgae_baseline_different_hyper'
# exp_name='dgae_baseline_different_hyper'
# exp_name='citeseer_r4'
# run_name='community_cb_compare'
# exp_name='community_test_prior'
# exp_name='community_decode'
# exp_name='community_index'
# exp_name='community_index'
# exp_name='grid_base'
# exp_name='ar_baseline'
exp_name='ar_recon_gen_collaps'

# 定义实验配置，现在包含数据集信息
declare -A experiment_configs=(
    ['community_']


)