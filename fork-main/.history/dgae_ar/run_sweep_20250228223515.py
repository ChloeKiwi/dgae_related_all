import wandb
import yaml
import argparse

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="enzymes")
    parser.add_argument("--codebook_size", type=int, default=256)
    parser.add_argument("--nc", type=int, default=1)
    parser.add_argument("--num_runs", type=int, default=20, 
                       help="Number of different hyperparameter combinations to try")
    args = parser.parse_args()
    
    # 加载sweep配置
    with open('./config/sweep.yaml', 'r') as f:
        sweep_config = yaml.safe_load(f)
        
        # 添加数据集参数
    sweep_config['parameters']['args'] = {
        'dataset': {'value': args.dataset},
        'codebook_size': {'value': args.codebook_size},
        'nc': {'value': args.nc}
    }
    
    # 初始化sweep
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=f'VQ-GAE_sweep'
    )
    
    # 启动sweep
    wandb.agent(sweep_id, function=train_function, count=20)  # 运行20次试验

def train_function():
    # 这里调用你的主训练函数
    from main import main
    main()

if __name__ == '__main__':
    main()