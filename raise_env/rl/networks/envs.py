import logging
import os

import gym
import numpy as np
import torch
from gym.spaces.box import Box
from gym.spaces.dict import Dict

from baselines import bench
from baselines.common.vec_env import VecEnvWrapper
from rl.networks.dummy_vec_env import DummyVecEnv
from rl.networks.shmem_vec_env import ShmemVecEnv
from baselines.common.vec_env.vec_normalize import \
    VecNormalize as VecNormalize_
from rl.vec_env.vec_pretext_normalize import VecPretextNormalize

logger = logging.getLogger(__name__)


def make_env(env_id, seed, rank, log_dir, allow_early_resets, config=None, envNum=1, ax=None, test_case=-1,
             reward_fn=None):
    def _thunk():
        # Windows spawn workers do not inherit gym registrations from the parent.
        import crowd_sim  # noqa: F401

        env = gym.make(env_id)

        is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            # Lazy: Atari wrappers pull opencv; CrowdSim never hits this branch.
            from baselines.common.atari_wrappers import make_atari
            env = make_atari(env_id)

        env.configure(config)

        # Optional pluggable scalar reward (Phase 1 RewardFunction).
        if reward_fn is not None:
            raw = env.unwrapped if hasattr(env, "unwrapped") else env
            if hasattr(raw, "set_reward_fn"):
                raw.set_reward_fn(reward_fn)
            else:
                raw.reward_fn = reward_fn
                raw._reward_fn_override = reward_fn

        envSeed = seed + rank if seed is not None else None
        # environment.render_axis = ax
        env.thisSeed = envSeed
        env.nenv = envNum
        if envNum > 1:
            env.phase = 'train'
        else:
            env.phase = 'test'

        if ax:
            env.render_axis = ax
            if test_case >= 0:
                env.test_case = test_case
        env.seed(seed + rank)

        if str(env.__class__.__name__).find('TimeLimit') >= 0:
            env = TimeLimitMask(env)

        # if log_dir is not None:
        env = bench.Monitor(
            env,
            None,
            allow_early_resets=allow_early_resets)
        # Per-env construction is noisy at INFO; keep for DEBUG only.
        logger.debug("make_env: %s", env)

        if isinstance(env.observation_space, Box):
            if is_atari:
                if len(env.observation_space.shape) == 3:
                    from baselines.common.atari_wrappers import wrap_deepmind
                    env = wrap_deepmind(env)
            elif len(env.observation_space.shape) == 3:
                raise NotImplementedError(
                    "CNN models work only for atari,\n"
                    "please use a custom wrapper for a custom pixel input env.\n"
                    "See wrap_deepmind for an example.")

            # If the input has shape (W,H,3), wrap for PyTorch convolutions

            obs_shape = env.observation_space.shape
            if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
                env = TransposeImage(env, op=[2, 0, 1])

        return env

    return _thunk


def make_vec_envs(env_name,
                  seed,
                  num_processes,
                  gamma,
                  log_dir,
                  device,
                  allow_early_resets,
                  num_frame_stack=None,
                  config=None,
                  ax=None, test_case=-1, wrap_pytorch=True, pretext_wrapper=False,
                  reward_fn=None):
    envs = [
        make_env(env_name, seed, i, log_dir, allow_early_resets, config=config,
                 envNum=num_processes, ax=ax, test_case=test_case,
                 reward_fn=reward_fn)
        for i in range(num_processes)
    ]
    test = False if len(envs) > 1 else True

    if len(envs) > 1:
        # fork is unavailable / unsafe on Windows; use spawn (same as historical
        # OpenAI Baselines ShmemVecEnv practice). Active env factory is this module
        # (rl/networks/envs.py); there is no rl/vec_env/envs.py in this tree.
        context = "spawn" if os.name == "nt" else "fork"
        envs = ShmemVecEnv(envs, context=context)
    else:
        envs = DummyVecEnv(envs)
    # for collect data in supervised learning, we don't need to wrap pytorch
    if wrap_pytorch:
        if isinstance(envs.observation_space, Box):
            if len(envs.observation_space.shape) == 1:
                if gamma is None:
                    envs = VecNormalize(envs, ret=False, ob=False)
                else:
                    envs = VecNormalize(envs, gamma=gamma, ob=False, ret=False)

        envs = VecPyTorch(envs, device)
    if pretext_wrapper:
        if gamma is None:
            envs = VecPretextNormalize(envs, ret=False, ob=False, config=config, test=test)
        else:
            envs = VecPretextNormalize(envs, gamma=gamma, ob=False, ret=False, config=config, test=test)

    if num_frame_stack is not None:
        envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
    elif isinstance(envs.observation_space, Box):
        if len(envs.observation_space.shape) == 3:
            envs = VecPyTorchFrameStack(envs, 4, device)

    return envs


# Checks whether done was caused my timit limits or not
class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# Can be used to test recurrent policies for Reacher-v2
class MaskGoal(gym.ObservationWrapper):
    def observation(self, observation):
        if self.env._elapsed_steps > 0:
            observation[-2:] = 0
        return observation


class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class TransposeImage(TransposeObs):
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, "Error: Operation, " + str(op) + ", must be dim3"
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], [
                obs_shape[self.op[0]], obs_shape[self.op[1]],
                obs_shape[self.op[2]]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, ob):
        return ob.transpose(self.op[0], self.op[1], self.op[2])


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        if isinstance(obs, dict):
            for key in obs:
                obs[key]=torch.from_numpy(obs[key]).to(self.device)
        else:
            obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        if isinstance(obs, dict):
            for key in obs:
                obs[key] = torch.from_numpy(obs[key]).to(self.device)
        else:
            obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info

    def render_traj(self, path, episode_num):
        if self.venv.num_envs == 1:
            return self.venv.envs[0].env.render_traj(path, episode_num)
        else:
            for i, curr_env in enumerate(self.venv.envs):
                curr_env.env.render_traj(path, str(episode_num) + '.' + str(i))


class VecNormalize(VecNormalize_):
    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs, update=True):
        if self.ob_rms:
            if self.training and update:
                self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) /
                          np.sqrt(self.ob_rms.var + self.epsilon),
                          -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        if device is None:
            device = torch.device('cpu')
        self.stacked_obs = torch.zeros((venv.num_envs, ) +
                                       low.shape).to(device)

        observation_space = gym.spaces.Box(
            low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stacked_obs[:, :-self.shape_dim0] = \
            self.stacked_obs[:, self.shape_dim0:].clone()
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        if torch.backends.cudnn.deterministic:
            self.stacked_obs = torch.zeros(self.stacked_obs.shape)
        else:
            self.stacked_obs.zero_()
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()
