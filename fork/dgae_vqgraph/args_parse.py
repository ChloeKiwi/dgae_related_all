import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str,
        default='enzymes',
        help="Name of the dataset. Available: qm9, zinc, community, ego, enzymes" #TODO: ego data.x是Nonetype，why？ 
    )

    parser.add_argument(
        "--work_type", type=str,
        default='sample', help="Options: train_autoencoder, train_prior, sample"
    )

    parser.add_argument(
        "--model_folder", type=str,
        default='./models/enzymes_prior/files/config.yaml',
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
    return parser.parse_args()