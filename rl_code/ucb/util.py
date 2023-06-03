import logging
import pickle
import numpy as np
from collections import defaultdict
from pathlib import Path

log = logging.getLogger(__name__)


class Dumper:
    def __init__(self, experiment_name):
        cwd = Path.cwd()
        # this should be the root of the repo
        self.expdir = cwd
        log.info(f"Dumper dumping to {cwd}")
        self.info = defaultdict(list)
        self.info_path = self.expdir / "info.pkl"

    def add(self, name, val, verbose=True, log_mean_std=False):
        if verbose:
            try:
                val = float(val)
                log.info(f"{name}: {val:.3f}")
            except TypeError:
                log.info(f"{name}: {val}")
            if log_mean_std:
                valarray = np.array(val)
                log.info(
                    f"{name}: mean={valarray.mean():.3f} std={valarray.std():.3f}"
                )
        self.info[name].append(val)

    def replace(self, name, val, verbose=False):
        if verbose:
            try:
                val = float(val)
                log.info(f"{name}: {val:.3f}")
            except TypeError:
                log.info(f"{name}: {val}")
        self.info[name] = val

    def extend(self, name, vals, verbose=False):
        if verbose:
            disp_vals = [f"{val:.3f}" for val in vals]
            log.info(f"{name}: {disp_vals}")
        self.info[name].extend(vals)

    def save(self):
        with self.info_path.open("wb") as f:
            pickle.dump(self.info, f)


def execute_time_varying_policy(env, agent, replay_buffer, dumper):
    obs = env.reset()
    total_return = 0.0
    agent.reset()
    policy = agent.policy
    for t in range(env.horizon):
        action = policy(obs, t)
        next_obs, rew, _, _ = env.step(action)
        replay_buffer.add_point(obs, action, next_obs, rew, t)
        total_return += rew
        obs = next_obs
    dumper.add("Exploration Returns", total_return)
    add_policy_stats(dumper, agent.stats)


def execute_tqrl_policy(env, agent, replay_buffer, dumper):
    policy = agent.policy
    for t in range(env.horizon):
        obs = agent.choose_start(t, env.observation_space)
        env.reset(obs)
        action = policy(obs, t)
        next_obs, rew, _, _ = env.step(action)
        replay_buffer.add_point(obs, action, next_obs, rew, t)
    add_policy_stats(dumper, agent.stats)


def execute_us(env, agent, replay_buffer, dumper):
    for t in range(env.horizon):
        obs, action = agent.choose_start(t, env.observation_space, env.action_space)
        env.reset(obs)
        next_obs, rew, _, _ = env.step(action)
        replay_buffer.add_point(obs, action, next_obs, rew, t)
    add_policy_stats(dumper, agent.stats)


def execute_random_tqrl(env, agent, replay_buffer, dumper):
    for t in range(env.horizon):
        obs, action = env.observation_space.sample(), env.action_space.sample()
        env.reset(obs)
        next_obs, rew, _, _ = env.step(action)
        replay_buffer.add_point(obs, action, next_obs, rew, t)
    add_policy_stats(dumper, agent.stats)


def add_policy_stats(dumper, stats):
    for name, stat in stats.items():
        dumper.add(name, stat, verbose=False)


def execute_random_policy(env, replay_buffer, dumper):
    obs = env.reset()
    total_return = 0.0
    for t in range(env.horizon):
        action = env.action_space.sample()
        next_obs, rew, _, _ = env.step(action)
        replay_buffer.add_point(obs, action, next_obs, rew, t)
        total_return += rew
        obs = next_obs
    dumper.add("Exploration Returns", total_return)


def eval_time_varying_policy(env, agent, dumper, num_trials):
    returns = []
    policy = agent.policy
    all_stats = []
    episodes = []
    for trial_num in range(num_trials):
        episode = defaultdict(list)
        obs = env.reset()
        episode["obs"].append(obs)
        agent.reset()
        total_return = 0.0
        for t in range(env.horizon):
            action = policy(obs, t)
            next_obs, rew, _, _ = env.step(action)
            episode["next_obs"].append(next_obs)
            episode["reward"].append(rew)
            episode["action"].append(action)
            total_return += rew
            obs = next_obs
            episode["obs"].append(obs)
        returns.append(total_return)
        all_stats.append(agent.stats)
        episodes.append(episode)
    stats = flatten_stats(all_stats)
    add_policy_stats(dumper, stats)
    dumper.add("Eval Returns", returns, verbose=False)
    log.info(f"Mean Return: {np.mean(returns)}")
    log.info(f"Stderr Returns: {np.std(returns) / np.sqrt(len(returns))}")
    dumper.add("Eval Episodes", episodes, verbose=False)


def flatten_stats(all_stats):
    stats = defaultdict(list)
    for one_stats in all_stats:
        for k, v in one_stats.items():
            stats[k].append(v)
    return stats


def set_seed(seed):
    np.random.seed(seed)
