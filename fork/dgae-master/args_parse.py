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
    return parser.parse_args()
