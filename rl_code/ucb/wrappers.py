import gym
import numpy as np
import ucb.envs
from functools import lru_cache


class Wrapper:
    def __init__(self, env_name, **kwargs):
        """
        env_name: obvious, string
        dim_num_bins: can either be an integer or an iterable of integers that is the length of the action space of the env
                  (for varying numbers of bins per dimension)
        """
        self._wrapped_env = gym.make(env_name)
        self.wrapped_action_space = self._wrapped_env.action_space
        self.action_space = self._wrapped_env.action_space

    @property
    def horizon(self):
        return self._wrapped_env.horizon

    @property
    def observation_space(self):
        return self._wrapped_env.observation_space

    def reset(self, obs=None):
        return self._wrapped_env.reset(obs=obs)

    def step(self, action):
        obs, rew, done, info = self._wrapped_env.step(action)
        rew = self.normalize_reward(rew)
        return obs, rew, done, info

    def discrete_to_continuous(self, action):
        # this needs to work whether action is a single int or an array of ints
        return action.reshape(-1, 1)

    def normalize_reward(self, rew):
        rew -= self._wrapped_env.reward_bounds[0]
        rew_range = (
            self._wrapped_env.reward_bounds[1] - self._wrapped_env.reward_bounds[0]
        )
        rew /= rew_range
        return np.minimum(1, np.maximum(0, rew))


class DiscreteActionWrapper(Wrapper):
    def __init__(self, env_name, dim_num_bins, **kwargs):
        """
        env_name: obvious, string
        dim_num_bins: can either be an integer or an iterable of integers that is the length of the action space of the env
                  (for varying numbers of bins per dimension)
        """
        super().__init__(env_name, **kwargs)
        self.action_space_size = self.wrapped_action_space.high.size
        if type(dim_num_bins) is int:
            self.dim_num_bins = [dim_num_bins] * self.action_space_size
        else:
            self.dim_num_bins = dim_num_bins[: self.action_space_size]

        self.dim_spaces = [
            np.linspace(
                self.wrapped_action_space.low[i],
                self.wrapped_action_space.high[i],
                num=self.dim_num_bins[i],
            )
            for i in range(self.action_space_size)
        ]
        self.dim_deltas = [space[1] - space[0] for space in self.dim_spaces]
        self.dim_mods = np.cumprod(self.dim_num_bins[::-1])[::-1]
        self.action_space = gym.spaces.Discrete(np.product(self.dim_num_bins))

    # @lru_cache(maxsize=8)
    def discrete_to_continuous(self, action):
        # this needs to work whether action is a single int or an array of ints
        if isinstance(action, int) or isinstance(action, np.int64):
            continuous_action = np.empty(self.action_space_size)
            for i in reversed(range(self.action_space_size)):
                continuous_action[i] = (
                    action % self.dim_num_bins[i]
                ) * self.dim_deltas[i] + self.wrapped_action_space.low[i]
                action //= self.dim_num_bins[i]
            return continuous_action
        continuous_actions = np.empty((action.shape[0], self.action_space_size))
        for i in reversed(range(self.action_space_size)):
            continuous_actions[:, i] = (
                action % self.dim_num_bins[i]
            ) * self.dim_deltas[i] + self.wrapped_action_space.low[i]
            action //= self.dim_num_bins[i]
        return continuous_actions

    def continuous_to_discrete(self, action):
        raise NotImplementedError()

    def step(self, action):
        continuous_action = self.discrete_to_continuous(action).flatten()
        obs, rew, done, info = self._wrapped_env.step(continuous_action)
        rew = self.normalize_reward(rew)
        return obs, rew, done, info
