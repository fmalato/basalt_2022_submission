import logging
import os
import argparse

import openai_vpt.train

import coloredlogs
coloredlogs.install(logging.DEBUG)

# The dataset and trained models are available in data/ directory from repository root.
MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT', 'data/')


def main(args):
    """
    This function will be called for training phase.
    This should produce and save same files you upload during your submission.
    All trained models should be placed under "train" directory!
    """
    openai_vpt.train.main(args)


if __name__ == "__main__":
    pars = argparse.ArgumentParser()
    pars.add_argument('-n', '--num_traj', required=True, default=100, type=int)
    pars.add_argument('-m', '--model', required=True)
    pars.add_argument('-w', '--weights', required=True)
    args = pars.parse_args()

    main(args)
