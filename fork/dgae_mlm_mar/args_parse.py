import argparse


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
        "--model_folder", type=str,
        default='./wandb/enzymes_prior/files/config.yaml',
        help="Name of the folder with the saved model "
             "(the prior model to sample or the auto-encoder model to train the prior)."
    )
    
    parser.add_argument(
        "--gpu", type=str,
        default='0',
        help="GPU to use for training"
    )

    parser.add_argument(
        "--exp_name", type=str,
        default='debug',
        help="Name of the experiment"
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
    
    return parser.parse_args()