import logging
import os

import openai_vpt.train

import coloredlogs
coloredlogs.install(logging.DEBUG)

# The dataset and trained models are available in data/ directory from repository root.
MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT', 'data/')


def main():
    """
    This function will be called for training phase.
    This should produce and save same files you upload during your submission.
    All trained models should be placed under "train" directory!
    """
    openai_vpt.train.main()


if __name__ == "__main__":
    main()
