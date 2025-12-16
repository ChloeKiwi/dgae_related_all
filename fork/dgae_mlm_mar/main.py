import args_parse
from trainer import Trainer
from data.dataset import get_dataset
from config.config import get_config, get_prior_config, get_sample_config
import os
from process_result import *
    
def save_run_id(config, work_type):
    """保存当前 wandb run ID 到临时文件"""
    # import ipdb; ipdb.set_trace()
    run_id = config.wandb_dir  # 获取运行ID
    run_id_file = f'./models_own/{config.exp_name}/{config.dataset}_{work_type}/wandb_run_id'
    # 确保目标目录存在
    os.makedirs(os.path.dirname(run_id_file), exist_ok=True)
    with open(run_id_file, 'w') as f:
        f.write(run_id)
        print(f"run_id: {run_id} has been saved to {run_id_file}")


def main() -> None:
    # Parse command line arguments
    args = args_parse.parse_args()

    # Choose the appropriate configuration based on the work type
    if args.work_type == 'train_autoencoder':
        config = get_config(args)
        config.sample = False
        # 如果指定了codebook_size, nc, wandb,则覆盖config中的值
        if args.codebook_size is not None:
            config.model.quantizer.codebook_size = args.codebook_size
        if args.nc is not None:
            config.model.quantizer.nc = args.nc 
        if args.wandb is not None:
            config.log.wandb = args.wandb
    elif args.work_type == 'train_prior':
        config = get_prior_config(args)
        config.sample = False
         # 如果指定了codebook_size, nc, wandb,则覆盖config中的值
        if args.codebook_size is not None:
            config.model.quantizer.codebook_size = args.codebook_size
        if args.nc is not None:
            config.model.quantizer.nc = args.nc 
        if args.wandb is not None:
            config.log.wandb = args.wandb
    elif args.work_type == 'sample':
        config = get_sample_config(args)
        config.sample = True
        config.dataset = args.dataset
    else:
        raise NotImplementedError('This is not a valid work type: check your spelling')

    # Set the model folder in the configuration
    config.model_folder = args.model_folder
    config.gpu = args.gpu
    config.exp_name = args.exp_name
    config.use_mask = args.use_mask
        
    print(f"exp_name: {config.exp_name}")
    print(f"dataset: {args.dataset}")

    # Get the dataset loaders and updated configuration
    loaders, config, data_info = get_dataset(args.dataset, config)

    # Create a Trainer instance with the dataset loaders and configuration
    trainer = Trainer(loaders, config, data_info)

    # Execute the appropriate method based on the work type
    if args.work_type == 'train_autoencoder':
        # import ipdb; ipdb.set_trace()
        save_run_id(trainer.config, 'autoencoder')
        trainer.autoencoder()
    elif args.work_type == 'train_prior':
        save_run_id(trainer.config, 'prior')
        trainer.prior()
    elif args.work_type == 'sample':
        trainer.sample()
        # 处理结果
        sample_dir=f'./models_own/{trainer.config.exp_name}/{trainer.config.dataset}_sample'
        process_result(sample_dir)

if __name__ == "__main__":
    main()

