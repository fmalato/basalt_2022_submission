# Basic behavioural cloning
# Note: this uses gradient accumulation in batches of ones
#       to perform training.
#       This will fit inside even smaller GPUs (tested on 8GB one),
#       but is slow.
# NOTE: This is _not_ the original code used for VPT!
#       This is merely to illustrate how to fine-tune the models and includes
#       the processing steps used.

# This will likely be much worse than what original VPT did:
# we are not training on full sequences, but only one step at a time to save VRAM.

import pickle

import torch
import torch as th
import numpy as np
import os

from openai_vpt.agent import PI_HEAD_KWARGS, MineRLAgent
from openai_vpt.data_loader import DataLoader
from openai_vpt.lib.tree_util import tree_map

EPOCHS = 1  # Keep at 1 for encoding
# Needs to be <= number of videos
BATCH_SIZE = 1
# Ideally more than batch size to create
# variation in datasets (otherwise, you will
# get a bunch of consecutive samples)
# Decrease this (and batch_size) if you run out of memory
N_WORKERS = 1
DEVICE = "cuda"

LOSS_REPORT_RATE = 100

LEARNING_RATE = 0.000181
WEIGHT_DECAY = 0.039428
MAX_GRAD_NORM = 5.0

# The dataset and trained models are available in data/ directory from repository root.
MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT', 'data/')

def load_model_parameters(path_to_model_file):
    agent_parameters = pickle.load(open(path_to_model_file, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs


def encode(data_dir, in_model, in_weights):
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)

    # To create model with the right environment.
    # All basalt environments have the same settings, so any of them works here
    agent = MineRLAgent(device=DEVICE, policy_kwargs=agent_policy_kwargs, pi_head_kwargs=agent_pi_head_kwargs)
    agent.load_weights(in_weights)

    policy = agent.policy

    data_loader = DataLoader(
        dataset_dir=data_dir,
        n_workers=N_WORKERS,
        batch_size=BATCH_SIZE,
        n_epochs=EPOCHS
    )

    # Keep track of the hidden state per episode/trajectory.
    # DataLoader provides unique id for each episode, which will
    # be different even for the same trajectory when it is loaded
    # up again
    episode_hidden_states = {}
    dummy_first = th.from_numpy(np.array((False,))).to(DEVICE)

    pi = {}
    img = {}
    actions = {}
    with torch.no_grad():
        for batch_i, (batch_images, batch_actions, batch_episode_id) in enumerate(data_loader):
            if batch_i > 300000:
                break
            for image, action, episode_id in zip(batch_images, batch_actions, batch_episode_id):
                agent_action = agent._env_action_to_agent(action, to_torch=True, check_if_null=True)
                if agent_action is None:
                    # Action was null
                    continue

                agent_obs = agent._env_obs_to_agent({"pov": image})
                if episode_id not in episode_hidden_states:
                    # TODO need to clean up this hidden state after worker is done with the work item.
                    #      Leaks memory, but not tooooo much at these scales (will be a problem later).
                    episode_hidden_states[episode_id] = policy.initial_state(1)
                    pi[episode_id] = []
                    img[episode_id] = []
                    actions[episode_id] = []
                agent_state = episode_hidden_states[episode_id]

                pi_distribution, v_prediction, new_agent_state, pi_h = policy.get_output_for_observation(
                    agent_obs,
                    agent_state,
                    dummy_first,
                    get_pi_h=True
                )
                new_agent_state = tree_map(lambda x: x.detach(), new_agent_state)
                episode_hidden_states[episode_id] = new_agent_state
                #if batch_i % 100 == 0:
                #    print(batch_i)
                # batch_size is 1
                pi[episode_id].append(pi_h.detach().cpu()[0, 0].numpy())
                actions[episode_id].append(action)
    pi = [np.array(v).astype('float16') for v in list(pi.values())]
    actions = [np.array(v) for v in list(actions.values())]
    np.savez_compressed("train/states2actions_"+data_dir.split("/")[-1], pi=np.array(pi), actions=np.array(actions))


def do_clustering(pi, n_clusters):
    import sklearn
    pi_flat = pi.reshape(-1, pi.shape[-1])
    kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters, n_init=500, max_iter=300, init="random")
    kmeans = kmeans.fit(pi_flat)
    dists = [[] for _ in range(len(pi))]
    bests = [[] for _ in range(len(pi))]
    bins = [0] * n_clusters
    for i in range(len(pi)):
        for p in pi[i]:
            dists[i].append(np.min(kmeans.transform(p.reshape(1, -1))))
            best = kmeans.predict(p.reshape(1, -1))
            bins[best[0]] += 1
            bests[i].append(best[0])
    print(np.mean(np.array(dists)))
    return bests, np.mean(np.array(dists)), bins, kmeans


def main():
    """try:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.benchmark_limit = 0
    except Exception as e:
        print(e)"""
    n_clusters = 10
    in_model = os.path.join(MINERL_DATA_ROOT, "VPT-models/foundation-model-1x.model")
    in_weights = os.path.join(MINERL_DATA_ROOT, "VPT-models/foundation-model-1x.weights")

    data_dir = os.path.join(MINERL_DATA_ROOT, "MineRLBasaltFindCave-v0")
    encode(data_dir, in_model, in_weights)

    data_dir = os.path.join(MINERL_DATA_ROOT, "MineRLBasaltMakeWaterfall-v0")
    encode(data_dir, in_model, in_weights)

    data_dir = os.path.join(MINERL_DATA_ROOT, "MineRLBasaltBuildVillageHouse-v0")
    encode(data_dir, in_model, in_weights)

    data_dir = os.path.join(MINERL_DATA_ROOT, "MineRLBasaltCreateVillageAnimalPen-v0")
    encode(data_dir, in_model, in_weights)
