import numpy as np
import torch as th
import cv2
from gym3.types import DictType
from gym import spaces

from openai_vpt.lib.action_mapping import CameraHierarchicalMapping
from openai_vpt.lib.actions import ActionTransformer
from openai_vpt.lib.policy import MinecraftAgentPolicy
from openai_vpt.lib.torch_util import default_device_type, set_default_torch_device


# Hardcoded settings
AGENT_RESOLUTION = (128, 128)

POLICY_KWARGS = dict(
    attention_heads=16,
    attention_mask_style="clipped_causal",
    attention_memory_size=256,
    diff_mlp_embedding=False,
    hidsize=2048,
    img_shape=[128, 128, 3],
    impala_chans=[16, 32, 32],
    impala_kwargs={"post_pool_groups": 1},
    impala_width=8,
    init_norm_kwargs={"batch_norm": False, "group_norm_groups": 1},
    n_recurrence_layers=4,
    only_img_input=True,
    pointwise_ratio=4,
    pointwise_use_activation=False,
    recurrence_is_residual=True,
    recurrence_type="transformer",
    timesteps=128,
    use_pointwise_layer=True,
    use_pre_lstm_ln=False,
)

PI_HEAD_KWARGS = dict(temperature=2.0)

ACTION_TRANSFORMER_KWARGS = dict(
    camera_binsize=2,
    camera_maxval=10,
    camera_mu=10,
    camera_quantization_scheme="mu_law",
)

ENV_KWARGS = dict(
    fov_range=[70, 70],
    frameskip=1,
    gamma_range=[2, 2],
    guiscale_range=[1, 1],
    resolution=[640, 360],
    cursor_size_range=[16.0, 16.0],
)

TARGET_ACTION_SPACE = {
    "ESC": spaces.Discrete(2),
    "attack": spaces.Discrete(2),
    "back": spaces.Discrete(2),
    "camera": spaces.Box(low=-180.0, high=180.0, shape=(2,)),
    "drop": spaces.Discrete(2),
    "forward": spaces.Discrete(2),
    "hotbar.1": spaces.Discrete(2),
    "hotbar.2": spaces.Discrete(2),
    "hotbar.3": spaces.Discrete(2),
    "hotbar.4": spaces.Discrete(2),
    "hotbar.5": spaces.Discrete(2),
    "hotbar.6": spaces.Discrete(2),
    "hotbar.7": spaces.Discrete(2),
    "hotbar.8": spaces.Discrete(2),
    "hotbar.9": spaces.Discrete(2),
    "inventory": spaces.Discrete(2),
    "jump": spaces.Discrete(2),
    "left": spaces.Discrete(2),
    "pickItem": spaces.Discrete(2),
    "right": spaces.Discrete(2),
    "sneak": spaces.Discrete(2),
    "sprint": spaces.Discrete(2),
    "swapHands": spaces.Discrete(2),
    "use": spaces.Discrete(2)
}


def validate_env(env):
    """Check that the MineRL environment is setup correctly, and raise if not"""
    for key, value in ENV_KWARGS.items():
        if key == "frameskip":
            continue
        if getattr(env.task, key) != value:
            raise ValueError(f"MineRL environment setting {key} does not match {value}")
    action_names = set(env.action_space.spaces.keys())
    if action_names != set(TARGET_ACTION_SPACE.keys()):
        raise ValueError(f"MineRL action space does match. Expected actions {set(TARGET_ACTION_SPACE.keys())}")

    for ac_space_name, ac_space_space in TARGET_ACTION_SPACE.items():
        if env.action_space.spaces[ac_space_name] != ac_space_space:
            raise ValueError(f"MineRL action space setting {ac_space_name} does not match {ac_space_space}")


def resize_image(img, target_resolution, interpolation=cv2.INTER_LINEAR):
    # For your sanity, do not resize with any function than INTER_LINEAR
    img = cv2.resize(img, target_resolution, interpolation=interpolation)
    return img


class MineRLAgent:
    def __init__(self, device=None, policy_kwargs=None, pi_head_kwargs=None, do_custom=False, set_location="./train/states2actions.npz", counter_max=20, offset=0):
        if device is None:
            device = default_device_type()
        self.device = th.device(device)
        # Set the default torch device for underlying code as well
        set_default_torch_device(self.device)
        self.action_mapper = CameraHierarchicalMapping(n_camera_bins=11)
        action_space = self.action_mapper.get_action_space_update()
        action_space = DictType(**action_space)

        self.action_transformer = ActionTransformer(**ACTION_TRANSFORMER_KWARGS)

        if policy_kwargs is None:
            policy_kwargs = POLICY_KWARGS
        if pi_head_kwargs is None:
            pi_head_kwargs = PI_HEAD_KWARGS

        agent_kwargs = dict(policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs, action_space=action_space)

        self.policy = MinecraftAgentPolicy(**agent_kwargs).to(device)
        self.hidden_state = self.policy.initial_state(1)
        self._dummy_first = th.from_numpy(np.array((False,))).to(device)

        self.offset = offset
        if do_custom:
            try:
                self.latents = th.Tensor(np.concatenate(np.load(set_location, allow_pickle=True)["pi"])).half().to(device)
                self.actions = np.load(set_location, allow_pickle=True)["actions"]
            except Exception as e:
                print(e)
                set_location = "./train/states2actions_MineRLBasaltFindCave-v0.npz"
                self.latents = th.Tensor(np.concatenate(np.load(set_location, allow_pickle=True)["pi"])).half().to(device)
                self.actions = np.load(set_location, allow_pickle=True)["actions"]
            self.fix_consistency(offset)
            self.actions = np.concatenate(self.actions)
            self.action_counter = 0
            self.diff = 0
            self.closest_idx = 0
            self.counter_max = counter_max
            self.current_rotation = np.array([offset, 0])
            self.in_inv = False
            self.last_was_inv = False
            self.repeat_inv = 0
            self.switch_counter = 0

    def fix_consistency(self, offset):
        for episode in self.actions:
            camera_total = np.array([offset, 0])
            in_inventory = False
            last_was_inv = False
            for action in episode:
                if action['inventory'] == 1:
                    if not last_was_inv:
                        in_inventory = not in_inventory
                    last_was_inv = True
                else:
                    last_was_inv = False
                if in_inventory and action['ESC'] == 1 and not last_was_inv:
                    in_inventory = False
                action['in_inventory'] = in_inventory
                if not in_inventory:
                    camera_total += action['camera']
                    camera_total[0] = np.clip(camera_total[0], -90, 90)
                action['camera_total'] = camera_total.copy()

    def load_weights(self, path):
        """Load model weights from a path, and reset hidden state"""
        self.policy.load_state_dict(th.load(path, map_location='cpu'), strict=False)
        self.reset()

    def reset(self):
        """Reset agent to initial state (i.e., reset hidden state)"""
        self.hidden_state = self.policy.initial_state(1)
        self.action_counter = 0
        self.diff = 0
        self.closest_idx = 0
        self.in_inv = False
        self.last_was_inv = False
        self.repeat_inv = 0
        self.switch_counter = 0
        self.current_rotation = np.array([self.offset, 0])

    def _env_obs_to_agent(self, minerl_obs):
        """
        Turn observation from MineRL environment into model's observation

        Returns torch tensors.
        """
        agent_input = resize_image(minerl_obs["pov"], AGENT_RESOLUTION)[None]
        agent_input = {"img": th.from_numpy(agent_input).to(self.device)}
        return agent_input

    def _env_obs_to_agent_custom(self, minerl_obs):
        """
        Turn observation from MineRL environment into model's observation

        Returns torch tensors.
        """
        agent_input = resize_image(minerl_obs["pov"], AGENT_RESOLUTION)[None]
        agent_input_tensor = {"img": th.from_numpy(agent_input).to(self.device)}
        return agent_input_tensor, agent_input

    def _agent_action_to_env(self, agent_action):
        """Turn output from policy into action for MineRL"""
        # This is quite important step (for some reason).
        # For the sake of your sanity, remember to do this step (manual conversion to numpy)
        # before proceeding. Otherwise, your agent might be a little derp.
        action = agent_action
        if isinstance(action["buttons"], th.Tensor):
            action = {
                "buttons": agent_action["buttons"].cpu().numpy(),
                "camera": agent_action["camera"].cpu().numpy()
            }
        minerl_action = self.action_mapper.to_factored(action)
        minerl_action_transformed = self.action_transformer.policy2env(minerl_action)
        return minerl_action_transformed

    def _env_action_to_agent(self, minerl_action_transformed, to_torch=False, check_if_null=False):
        """
        Turn action from MineRL to model's action.

        Note that this will add batch dimensions to the action.
        Returns numpy arrays, unless `to_torch` is True, in which case it returns torch tensors.

        If `check_if_null` is True, check if the action is null (no action) after the initial
        transformation. This matches the behaviour done in OpenAI's VPT work.
        If action is null, return "None" instead
        """
        minerl_action = self.action_transformer.env2policy(minerl_action_transformed)
        if check_if_null:
            if np.all(minerl_action["buttons"] == 0) and np.all(minerl_action["camera"] == self.action_transformer.camera_zero_bin):
                return None

        # Add batch dims if not existant
        if minerl_action["camera"].ndim == 1:
            minerl_action = {k: v[None] for k, v in minerl_action.items()}
        action = self.action_mapper.from_factored(minerl_action)
        if to_torch:
            action = {k: th.from_numpy(v).to(self.device) for k, v in action.items()}
        return action

    def get_action(self, minerl_obs):
        """
        Get agent's action for given MineRL observation.

        Agent's hidden state is tracked internally. To reset it,
        call `reset()`.
        """
        agent_input = self._env_obs_to_agent(minerl_obs)
        # The "first" argument could be used to reset tell episode
        # boundaries, but we are only using this for predicting (for now),
        # so we do not hassle with it yet.
        agent_action, self.hidden_state, _ = self.policy.act(
            agent_input, self._dummy_first, self.hidden_state,
            stochastic=True
        )
        minerl_action = self._agent_action_to_env(agent_action)
        return minerl_action

    def get_action_custom(self, minerl_obs, initial=False, power=1):
        """
        Get agent's action for given MineRL observation.

        Agent's hidden state is tracked internally. To reset it,
        call `reset()`.
        """
        with th.no_grad():
            agent_input, agent_input_np = self._env_obs_to_agent_custom(minerl_obs)
            if np.min(self.get_inventory_diffs(agent_input_np[0])) < 0.35:  # or self.in_inv:
                if not self.in_inv:
                    self.in_inv = True
                #elif not np.min(self.get_inventory_diffs(agent_input_np[0])) < 1:
                #    pass

            _, _, new_state, pi = self.policy.get_output_for_observation(agent_input, self.hidden_state, self._dummy_first, True)
            self.hidden_state = new_state
            if self.action_counter + self.closest_idx < len(self.actions):
                curr_diff = th.mean(th.abs((self.latents[self.closest_idx + self.action_counter] - pi[0, 0].half())))
            else:
                curr_diff = 0.4
            if curr_diff >= 2 * self.diff and curr_diff >= 0.2:
                self.switch_counter = min(8, self.switch_counter + 1)
            else:
                self.switch_counter = max(0, self.switch_counter - 1)

            if initial or self.action_counter >= self.counter_max - 1 \
                    or self.action_counter + self.closest_idx >= len(self.actions) \
                    or (self.action_counter >= (self.counter_max * 0.5) and (self.switch_counter >= 8
                                                                                 or curr_diff >= 0.45)):
                #if self.switch_counter >= 8 or curr_diff >= 0.45:
                #    print("Too far off, recalculating...", curr_diff.cpu().item(), self.diff)

                self.action_counter = 0
                self.switch_counter = 0
                diff_arr = th.mean(th.abs((self.latents - pi[0, 0].half()) ** power), dim=1)
                self.find_nearest_latent(diff_arr)
                self.diff = th.mean(th.abs((self.latents - pi[0, 0].half())), dim=1)[self.closest_idx].cpu().item()

            if self.action_counter + self.closest_idx < len(self.actions):
                ret_val = self.actions[self.closest_idx + self.action_counter].copy()
            else:
                ret_val = self.actions[self.closest_idx + 0].copy()

            self.action_counter += 1
            if self.in_inv != ret_val['in_inventory']:
                self.repeat_inv = 1

            if initial or self.repeat_inv > 0:
                ret_val['inventory'] = 0
                ret_val['ESC'] = 0
                ret_val['back'] = 0
                ret_val['drop'] = 0
                ret_val['forward'] = 0
                ret_val['jump'] = 0
                ret_val['left'] = 0
                ret_val['right'] = 0
                ret_val['sprint'] = 0
                ret_val['swapHands'] = 0
                ret_val['use'] = 0
                ret_val['pickItem'] = 0
                ret_val['attack'] = 0
            if self.repeat_inv > 0:
                ret_val['inventory'] = 1
                self.repeat_inv -= 1
            else:
                ret_val['inventory'] = 0

            if ret_val['inventory'] == 1:
                if not self.last_was_inv:
                    self.in_inv = not self.in_inv

                self.last_was_inv = True
            else:
                self.last_was_inv = False

            if self.in_inv and ret_val['ESC'] == 1 and not self.last_was_inv:
                was_in_inv = True
                self.in_inv = False
            else:
                was_in_inv = False

            if not self.in_inv and not self.last_was_inv:
                ret_val['camera'][0] = (ret_val['camera_total'][0] - self.current_rotation[0])
                ret_val['camera'][0] = np.clip(ret_val['camera'][0], -20, 20)
                ret_val['camera'][1] = np.clip(ret_val['camera'][1], -20, 20)
                self.current_rotation = self.current_rotation + ret_val['camera']

            del ret_val['camera_total']
            del ret_val['in_inventory']
            return ret_val, was_in_inv

    def find_nearest_latent(self, diff_arr):
        self.closest_idx = th.argmin(diff_arr).cpu().item()

    def get_inventory_diffs(self, img):
        # Trick for now. Usually train simple model for that, which also recognizes more.
        c = np.array([198, 198, 198])
        ret = []
        ret.append(
            np.mean([
                np.abs(img[60, 81] - c),
                np.abs(img[60, 88] - c),
                np.abs(img[45, 60] - c),
            ]),
        )
        ret.append(
            np.mean([
                np.abs(img[58, 91] - c),
                np.abs(img[40, 92] - c),
            ]),
        )
        ret.append(
            np.mean([
                np.abs(img[58, 76] - c),
                np.abs(img[40, 77] - c),
            ]),
        )
        return np.array(ret)
