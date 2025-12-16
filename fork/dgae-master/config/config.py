import yaml
from easydict import EasyDict as edict

def get_config(args):
    config_path = f'./config/{args.dataset}_autoencoder.yaml'
    config = edict(yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader))
    config.training.betas = (config.training.beta1, config.training.beta2)
    config.dataset = args.dataset
    config.work_type = args.work_type
    config.train_prior = False
    return config


def get_prior_config(args):
    config_path = args.model_folder
    config_autoencoder = edict(yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader))
    config_dir = f'./config/{args.dataset}_prior.yaml'
    config = yaml.load(open(config_dir, 'r'), Loader=yaml.FullLoader)
    config = edict({**config_autoencoder, **config})
    config.training.betas = (config.training.beta1, config.training.beta2)
    config.dataset = args.dataset
    config.work_type = args.work_type
    config.autoencoder_path = args.model_folder
    config.train_prior = True
    config.model.quantizer.init_steps = 0
    return config

def get_sample_config(args):
    config_path = args.model_folder
    config_dict = yaml.load(open(config_path, 'r'),Loader=yaml.FullLoader)

    dict_ = {}
    for key in config_dict:
        if key != 'wandb_version' and key != '_wandb':
            dict_[key] = config_dict[key]['value']
    config = edict(dict_)
    config = edict({**config})
    config.folder_name = args.model_folder
    config.betas = (config.training.beta1, config.training.beta2)
    config.dataset = args.dataset
    config.work_type = args.work_type
    config.sample = True
    config.log.wandb = False
    if args.dataset == 'qm9' or args.dataset == 'zinc':
        config.n_samples = 10000
    elif args.dataset == 'enzymes':
        config.n_samples = 117
    elif args.dataset == 'ego':
        config.n_samples = 40
    elif args.dataset == 'community':
        config.n_samples = 20
    else:
        raise NotImplementedError('Dataset not implemented. Check the spelling')
    return config

def only_numerics(seq):
    seq_type = type(seq)
    return seq_type().join(filter(seq_type.isdigit, seq))

