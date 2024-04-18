import os
import numpy as np
from argparse import ArgumentParser
import pickle

import aicrowd_gym

from openai_vpt.agent import MineRLAgent

MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT', 'data/')


def main(env_name, n_episodes, max_steps, show=False, counter_max=128, offset=0, power=1, warmup=0):
    """try:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.benchmark_limit = 0
    except Exception as e:
        print(e)"""
    # Using aicrowd_gym is important! Your submission will not work otherwise
    env = aicrowd_gym.make(env_name)
    # Load your model here
    # NOTE: The trained parameters must be inside "train" directory!
    in_model = os.path.join(MINERL_DATA_ROOT, "models/foundation-model-1x.model")
    in_weights = os.path.join(MINERL_DATA_ROOT, "models/foundation-model-1x.weights")

    agent_parameters = pickle.load(open(in_model, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    model = MineRLAgent(policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs, device="cuda",
                        do_custom=True, set_location=f"./train/states2actions_{env_name}.npz", counter_max=counter_max, offset=offset)
    model.load_weights(in_weights)
    diffs = []
    div_s = []
    time_s = []
    for i in range(n_episodes):
        obs = env.reset()
        model.reset()
        done = False
        closest_idx_old = -1
        if show:
            import cv2
            video = cv2.VideoWriter(f'{env_name}_episode_{i}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), float(20), (256, 256))
        for step_counter in range(max_steps):
            action, inv, curr_diff = model.get_action_custom(obs, step_counter < warmup, power=power)
            closest_idx = model.closest_idx
            if model.action_counter >= model.counter_max - 1 or model.action_counter + model.closest_idx >= len(model.actions):
                new_time_search = True
            else:
                new_time_search = False
            if model.action_counter >= (model.counter_max * 0.33) and (model.switch_counter >= 5 or curr_diff >= 0.45):
                new_divergence_search = True
            else:
                new_divergence_search = False
            if step_counter < max_steps and step_counter < 2000:
                if not inv:
                    action["ESC"] = 0
            diffs.append(curr_diff.cpu().numpy())
            div_s.append(new_divergence_search)
            time_s.append(new_time_search)
            obs, reward, done, info = env.step(action)
            if show:
                video.write(cv2.cvtColor(cv2.resize(env.render(mode="human"), (256, 256), cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
            if done:
                break
        if show:
            video.release()

        print(f"[{i}] Episode complete")

    # Close environment and clean up any bigger memory hogs.
    # Otherwise, you might start running into memory issues
    # on the evaluation server.
    env.close()
    np.savez_compressed(f"ep_distance", distance=np.array(diffs), div_based=np.array(div_s),
                        time_based=np.array(time_s))


if __name__ == "__main__":
    parser = ArgumentParser("Run pretrained models on MineRL environment")

    parser.add_argument("--weights", type=str, required=True, help="Path to the '.weights' file to be loaded.", default="data/models/foundation-model-1x.weights")
    parser.add_argument("--model", type=str, required=True, help="Path to the '.model' file to be loaded.", default="data/models/foundation-model-1x.model")
    parser.add_argument("--env", type=str, required=True, default="MineRLBasaltFindCave-v0")
    parser.add_argument("--show", action="store_true", help="Render the environment.")

    args = parser.parse_args()

    main(args.env, 1, 2000, show=args.show)
