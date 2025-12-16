import args_parse
from trainer import Trainer
from data.dataset import get_dataset
from config.config import get_config, get_prior_config, get_sample_config

def main() -> None:
    # Parse command line arguments
    args = args_parse.parse_args()

    # Choose the appropriate configuration based on the work type
    if args.work_type == 'train_autoencoder':
        config = get_config(args)
        config.sample = False
    elif args.work_type == 'train_prior':
        config = get_prior_config(args)

        config.sample = False
    elif args.work_type == 'sample':
        config = get_sample_config(args)
        config.sample = True
        config.dataset = args.dataset
    else:
        raise NotImplementedError('This is not a valid work type: check your spelling')

    # Set the model folder in the configuration
    config.model_folder = args.model_folder

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

if __name__ == "__main__":
    main()

