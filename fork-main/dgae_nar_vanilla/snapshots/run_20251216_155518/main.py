from trainer import Trainer
from data.dataset import get_dataset
from config.config import get_config, get_prior_config, get_sample_config
import os
import numpy as np  
import random
import torch
import argparse

######参数设置#######
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str,
        default='enzymes',
        help="Name of the dataset. Available: qm9, zinc, community, ego, enzymes"
    )

    parser.add_argument(
        "--work_type", type=str,
        default='sample', help="Options: train_autoencoder, train_prior, sample"
    )
    
    parser.add_argument(
        "--gpu", type=str,
        default='0',
        help="GPU to use for training"
    )

    parser.add_argument(
        "--exp_name", type=str,
        default='exp',
        help="Name of the experiment"
    )
    
    parser.add_argument(
        "--run_name", type=str,
        default='run',
        help='Name of the run'
    )
    
    parser.add_argument(
        "--use_mask",
        action='store_true',
        help="Whether to use mask prediction during training and sampling."
    )
    
    parser.add_argument(
        "--codebook_size", type=int,
        default=None,
        help="Size of the codebook"
    )
    
    parser.add_argument(
        "--nc", type=int,
        default=None,
        help="Number of channels"
    )
    parser.add_argument(
        "--wandb", type=str,
        default=None,
        help="Whether to use wandb, options: online, disabled"
    )
    
    parser.add_argument(
        "--mask_func", type=str,
        default='linear',
        help="Mask function to use, options: linear, cosine"
    )
    
    parser.add_argument(
        "--iterations_rate", type=float,
        default=1,
        help="Number of iterations to sample"
    )
    
    parser.add_argument(
        "--predict_method", tyep=str,
        default='predict_eos_pad',
        help="predict and caculate loss method during prior stage"
    )
    
    return parser.parse_args()

def set_seed(seed):
    """设置随机种子以确保可复现性"""
    random.seed(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

##### main function #####
def main() -> None:
    set_seed(42)
    args = parse_args()
    # Choose the appropriate configuration based on the work type
    if args.work_type == 'train_autoencoder':
        config = get_config(args)
        config.sample = False
        config.dataset = args.dataset
        config.iterations_rate = args.iterations_rate
        if args.codebook_size is not None:
            config.model.quantizer.codebook_size = args.codebook_size
        if args.nc is not None:
            config.model.quantizer.nc = args.nc 
        if args.wandb is not None:
            config.log.wandb = args.wandb
        
    elif args.work_type == 'train_prior':
        config = get_prior_config(args)
        config.sample = False
        config.dataset = args.dataset
        config.iterations_rate = args.iterations_rate
        if args.wandb is not None:
            config.log.wandb = args.wandb
            
    elif args.work_type == 'sample':
        config = get_sample_config(args)
        config.sample = True
        config.dataset = args.dataset
        config.iterations_rate = args.iterations_rate #1 or 1/2,1/3...
        if args.wandb is not None:
            config.log.wandb = args.wandb
    else:
        raise NotImplementedError('This is not a valid work type: check your spelling')

    # Set the model folder in the configuration
    config.gpu = args.gpu
    config.exp_name = args.exp_name
    config.run_name = args.run_name
    config.use_mask = args.use_mask
    config.mask_func = args.mask_func
    config.predict_method = args.predict_method
    print(f"exp_name: {config.exp_name}, run_name: {config.run_name}, dataset: {args.dataset},  \
          use_mask: {args.use_mask}, mask_func:{args.mask_func}, cb_size:{config.model.quantizer.codebook_size}, \
          nc:{config.model.quantizer.nc}, wandb:{config.log.wandb}, predict_method:{config.predict_method} \n")

    # Get the dataset loaders and updated configuration
    loaders, config, data_info = get_dataset(args.dataset, config)
    
    # Create a Trainer instance with the dataset loaders and configuration
    trainer = Trainer(loaders, config, data_info)

    # Execute the appropriate method based on the work type
    if args.work_type == 'train_autoencoder':        
        trainer.autoencoder()
        
    elif args.work_type == 'train_prior':                      
        trainer.prior()
        
    elif args.work_type == 'sample':
        trainer.sample()

def count_parameters(model):
    """计算模型的参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_parameters(trainer):
    # 计算并输出transformer的参数量
    transformer_params = count_parameters(trainer.transformer)
    formatted_params = format_params(transformer_params)
    print(f"\nTransformer Parameters: {transformer_params:,} ({formatted_params})")
    
    # 将参数量写入文件
    params_file = f'./models_own/{trainer.config.exp_name}/{trainer.config.run_name}/{trainer.config.dataset}_sample/model_params.txt'
    os.makedirs(os.path.dirname(params_file), exist_ok=True)
    with open(params_file, 'w') as f:
        f.write(f"Transformer Parameters: {transformer_params:,} ({formatted_params})\n")
        
        # 输出详细的参数统计
        f.write("\nParameters per component:\n")
        for name, module in trainer.transformer.named_children():
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            formatted = format_params(params)
            f.write(f"{name}: {params:,} ({formatted})\n")

def format_params(num_params):
    """格式化参数量，添加合适的单位"""
    if num_params >= 1e9:
        return f"{num_params/1e9:.2f}B"  # Billions
    elif num_params >= 1e6:
        return f"{num_params/1e6:.2f}M"  # Millions
    elif num_params >= 1e3:
        return f"{num_params/1e3:.2f}K"  # Thousands
    else:
        return f"{num_params}"
    
if __name__ == "__main__":
    main()

