"""
Run DQN on CartPole-v0.
"""
import argparse
import gym
import os
import random

import torch
from torch import nn as nn
import numpy as np
from rlkit.exploration_strategies.base import PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
from rlkit.policies.argmax import ArgmaxDiscretePolicy
from rlkit.torch.dqn.double_dqn import DoubleDQNTrainer
from rlkit.torch.networks import Mlp
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

import ucb.envs
from ucb.wrappers import Wrapper, DiscreteActionWrapper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', required=True, type=str)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--cuda_device', type=int, default=None)
    return parser.parse_args()


def experiment(variant):
    if 'mountaincar' in variant['env']:
        expl_env = Wrapper(variant['env'])
        eval_env = Wrapper(variant['env'])
    else:
        expl_env = DiscreteActionWrapper(variant['env'], dim_num_bins=10)
        eval_env = DiscreteActionWrapper(variant['env'], dim_num_bins=10)
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.n

    qf = Mlp(
        hidden_sizes=[256, 256],
        input_size=obs_dim,
        output_size=action_dim,
    )
    target_qf = Mlp(
        hidden_sizes=[256, 256],
        input_size=obs_dim,
        output_size=action_dim,
    )
    qf_criterion = nn.MSELoss()
    eval_policy = ArgmaxDiscretePolicy(qf)
    expl_policy = PolicyWrappedWithExplorationStrategy(
        EpsilonGreedy(expl_env.action_space, prob_random_action=0.2),
        eval_policy,
    )
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        expl_policy,
    )
    trainer = DoubleDQNTrainer(
        qf=qf,
        target_qf=target_qf,
        qf_criterion=qf_criterion,
        **variant["trainer_kwargs"]
    )
    replay_buffer = EnvReplayBuffer(
        variant["replay_buffer_size"],
        expl_env,
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant["algorithm_kwargs"]
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    args = parse_args()
    env = gym.make(args.env)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    variant = dict(
        algorithm="DDQN",
        version="normal",
        env=args.env,
        layer_size=256,
        replay_buffer_size=int(1e6),
        algorithm_kwargs=dict(
            num_epochs=250,
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1e3,
            # num_expl_steps_per_train_loop=10 * env.horizon,
            min_num_steps_before_training=1000,
            max_path_length=env.horizon,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            learning_rate=3e-4,
        ),
    )
    setup_logger(
        f"{args.env}",
        log_dir=os.path.join(os.getcwd(), f'rlruns/{args.env}/{args.seed}'),
        variant=variant,
    )
    if args.cuda_device is not None:
        ptu.set_gpu_mode(True, gpu_id=args.cuda_device)
    else:
        ptu.set_gpu_mode(False)
    experiment(variant)
