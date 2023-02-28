import logging
import os
import coloredlogs

import run_agent
from config import EVAL_EPISODES, EVAL_MAX_STEPS

coloredlogs.install(logging.DEBUG)

MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT', 'data/')


def main():
    MINERL_GYM_ENV = 'MineRLBasaltFindCave-v0'
    run_agent.main(MINERL_GYM_ENV, EVAL_EPISODES, EVAL_MAX_STEPS, counter_max=128, offset=-10, warmup=10)


if __name__ == "__main__":
    main()
