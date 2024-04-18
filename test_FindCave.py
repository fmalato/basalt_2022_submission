import logging
import os
import coloredlogs

import run_agent
from config import EVAL_EPISODES, EVAL_MAX_STEPS

from cog23_experiments import fixed_seed_search_based, fixed_seed_foundation_model, variable_seed_search_based, variable_seed_foundation_model
import cog_experiments
import cog_experiments_variable_seed


coloredlogs.install(logging.DEBUG)
logging.basicConfig()
logging.getLogger().setLevel(logging.WARNING)

MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT', 'data/')

# def test_fixed_seed_foundation_model_180_degrees_no_rotation(model_name, model_weights)


def main():
    MINERL_GYM_ENV = 'MineRLBasaltFindCave-v0'
    """random_seeds = [324033, 264254, 334116, 119233, 298032, 277606, 989289, 817725, 287190, 670899,
                    464240, 226451, 163349, 250195, 839121, 253806, 94873, 98614, 29692, 967745]"""
    """variable_seeds = [23, 29, 40, 42, 52, 73, 91, 105, 109, 113, 116, 122, 124, 128, 147, 163, 172, 173, 181, 186]
    folder_names = ['/data/federima/sbc_results/videos/search_based/']
    weights_names = ['/home/federima/basalt_2022_submission/train/states2actions_MineRLBasaltFindCave-v0.npz']
    for folder_name, weights_name in zip(folder_names, weights_names):
        variable_seed_search_based.main(MINERL_GYM_ENV, seeds=variable_seeds, max_steps=EVAL_MAX_STEPS, counter_max=32, divergence_scaling_factor=1.0,
                                        offset=-10, warmup=10, show=True, folder_name=folder_name, weight_location=weights_name,
                                        turn_timesteps=0, forward_timesteps=0, starting_run=0)"""
    import numpy as np
    seed = np.random.randint(1, 10000)
    variable_seed_search_based.main(MINERL_GYM_ENV, seeds=[seed], max_steps=EVAL_MAX_STEPS, counter_max=128,
                                    divergence_scaling_factor=2.0,
                                    offset=-10, warmup=10, show=True, folder_name='/data/federima/sbc_results/record_searches/',
                                    weight_location='/home/federima/basalt_2022_submission/train/states2actions_MineRLBasaltFindCave-v0.npz',
                                    turn_timesteps=0, forward_timesteps=0, starting_run=0, num_runs=1)

    """folder_names = ['videos', 'videos', 'videos']
    models_names = ['foundation-model-1x',
                    'foundation-model-2x',
                    'foundation-model-3x']
    weights_names = ['bc-model-1x',
                     'bc-model-2x',
                     'bc-model-3x']
    for folder_name, model_name, weights_name in zip(folder_names, models_names, weights_names):
        variable_seed_foundation_model.main(MINERL_GYM_ENV, seeds=variable_seeds, max_steps=EVAL_MAX_STEPS, show=True,
                                            folder_name=folder_name, model_name=model_name, model_weights=weights_name,
                                            turn_timesteps=0, forward_timesteps=0)"""
    #cog_experiments.main(MINERL_GYM_ENV, random_seeds[:10], EVAL_MAX_STEPS, counter_max=128, offset=-10, warmup=10, show=True, fixed_seed=False)
    """folder_names = ['varying_hyperparameters/plot_trajectory/10/',
                    'varying_hyperparameters/plot_trajectory/25/',
                    'varying_hyperparameters/plot_trajectory/50/',
                    'varying_hyperparameters/plot_trajectory/100/']
    weights_names = ['/home/federima/basalt_2022_submission/train/sorted_10_states2actions_MineRLBasaltFindCave-v0.npz',
                     '/home/federima/basalt_2022_submission/train/sorted_25_states2actions_MineRLBasaltFindCave-v0.npz',
                     '/home/federima/basalt_2022_submission/train/sorted_50_states2actions_MineRLBasaltFindCave-v0.npz',
                     '/home/federima/basalt_2022_submission/train/states2actions_MineRLBasaltFindCave-v0.npz']
    for folder_name, weights_name in zip(folder_names, weights_names):
        fixed_seed_search_based.main(MINERL_GYM_ENV, EVAL_EPISODES, EVAL_MAX_STEPS, counter_max=128, offset=-10, warmup=10,
                                     show=True, folder_name=folder_name, weight_location=weights_name,
                                     turn_timesteps=0, forward_timesteps=0, seed=42)"""

    """names = ['foundation-model-3x']
    weights = ['3x-out-50ep']
    for model_name, model_weights in zip(names, weights):
        folder_name = f'varying_hyperparameters/number_of_episodes/{model_name}'
        fixed_seed_foundation_model.main(MINERL_GYM_ENV, EVAL_EPISODES, EVAL_MAX_STEPS, show=True, folder_name=folder_name,
                                         model_name=model_name, model_weights=model_weights, turn_timesteps=0, forward_timesteps=0,
                                         seed=42)"""
    #folder_name = 'varying_hyperparameters/warmup/'
    #weights_name = '/home/federima/basalt_2022_submission/train/states2actions_MineRLBasaltFindCave-v0.npz'
    """num_runs = 3
    for r in range(0, num_runs, 1):
        folder_name = f'varying_hyperparameters/counter_max/run_{r}/'
        weights_name = '/home/federima/basalt_2022_submission/train/states2actions_MineRLBasaltFindCave-v0.npz'
        for i in [128]:
            # Made on seed=42
            fixed_seed_search_based.main(MINERL_GYM_ENV, 10, EVAL_MAX_STEPS, counter_max=i, offset=-10,
                                         warmup=10,
                                         show=True, folder_name=folder_name + f'{i}', weight_location=weights_name,
                                         turn_timesteps=0, forward_timesteps=0, seed=42)

        folder_name = f'varying_hyperparameters/divergence_scaling_factor/run_{r}/'
        weights_name = '/home/federima/basalt_2022_submission/train/states2actions_MineRLBasaltFindCave-v0.npz'
        for i in [2]:
            print(f"########## Divergence scaling factor: {i} ##########")
            fixed_seed_search_based.main(MINERL_GYM_ENV, 10, EVAL_MAX_STEPS, counter_max=128, offset=-10,
                                         warmup=10, divergence_scaling_factor=i,
                                         show=True, folder_name=folder_name + f'{i}', weight_location=weights_name,
                                         turn_timesteps=0, forward_timesteps=0, seed=42)"""

    # HumanSurvivalEnv good seeds
    # 385/386, 617/618, 808/809 (rot), 866/867 (rot), 973/974, 983/984 (mov), 1502/1503 (rot), 1786/1787 (rot),
    # 2009/2010 (?), 2393/2394 (rot), 2504/2505, 4043/4044 (rot), 4085/4086 (rot?), 4183/4184 (rot), 4780/4781 (rot)


if __name__ == "__main__":
    main()
