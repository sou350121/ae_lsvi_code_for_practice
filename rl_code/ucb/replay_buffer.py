import numpy as np
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class BufferResult:
    obs: np.ndarray
    actions: np.ndarray
    next_obs: np.ndarray
    rewards: np.ndarray


class ReplayBuffer:
    def __init__(self, horizon):
        self.horizon = horizon
        self.data = [defaultdict(list) for _ in range(horizon)]
        self.categories = ("obs", "actions", "next_obs", "rewards")
        self.ndata = 0
        self.discrete_actions = True

    def add_point(self, obs, action, next_obs, reward, time):
        time_data = self.data[time]
        time_data["obs"].append(obs)
        time_data["actions"].append(action)
        time_data["next_obs"].append(next_obs)
        time_data["rewards"].append(reward)
        self.ndata += 1

    def get_data(self, time=None):
        if time is None:
            return self.get_all_data()
        time_data = self.data[time]
        args = {name: np.array(time_data[name]) for name in self.categories}
        return BufferResult(**args)

    def get_all_data(self, time=None):
        all_data = defaultdict(list)
        for time_data in self.data:
            for k, v in time_data.items():
                all_data[k].extend(v)
        args = {name: np.array(all_data[name]) for name in self.categories}
        return BufferResult(**args)
