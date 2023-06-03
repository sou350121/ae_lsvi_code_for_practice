import numpy as np
import logging
import jax
import jax.numpy as jnp
from flax.training import train_state
from collections import defaultdict

# from flax import serialization
from functools import partial
from tqdm import trange, tqdm
from copy import deepcopy

log = logging.getLogger(__name__)


class StubQRegressor:
    def __init__(self, env, replay_buffer, **kwargs):
        self.env = env
        self.num_actions = env.action_space.n
        self.obs_space = env.observation_space
        self.obs_dim = self.obs_space.high.size
        self.replay_buffer = replay_buffer
        self.horizon = env.horizon
        self.stats = defaultdict(list)

    def reset(self):
        self.stats = defaultdict(list)

    def add_stat(self, name, stat):
        self.stats[name].append(stat)
        log.debug(f"{name}: {stat}")

    def policy(self, obs, time):
        pass

    def fit_q_functions(self):
        pass

    def __call__(self, obs, time, action=None, all_q=False):
        pass

    def get_q_dataset(self, t):
        pass

    def normalize_obs(self, obs):
        pass

    def clip(self, vals, time):
        pass

    @property
    def params(self):
        pass


class FiniteHorizonQRegressor(StubQRegressor):
    def __init__(self, beta, use_all_data, Xtrains=[], Ytrains=[], **kwargs):
        """
        env needs to be a DiscreteActionWrapper
        replay_buffer needs to be a ReplayBuffer
        """
        super().__init__(**kwargs)
        self.q_function_list = []
        self.beta = beta
        self.use_all_data = use_all_data
        self.train_states = [None] * self.horizon
        self.pred_fns = [None] * self.horizon
        self.Xtrains = Xtrains
        self.Ytrains = Ytrains
        self.model = None
        self.trainer = None

    def policy(self, obs, time):
        q_values = self(obs, time, all_q=True)
        maxq = float(jnp.max(q_values, axis=0))
        minq = float(jnp.min(q_values, axis=0))
        spread = maxq - minq
        self.add_stat("Q spread", spread)
        self.add_stat("Q value", maxq)
        opt_actions = jnp.argmax(q_values, axis=0)
        if obs.ndim == 1:
            opt_actions = int(opt_actions)
        return opt_actions

    def fit_q_functions(self):
        Xtrains = []
        Ytrains = []
        for t in tqdm(reversed(range(self.horizon))):
            # 1. write a function that makes a ML dataset for each timestep (configurable on what times to use)
            if len(self.Xtrains) == 0:
                Xtrain, Ytrain = self.get_q_dataset(t)
                upper_bound = self.horizon - t
                # scale model outputs to [-0.5, 0.5]
                Ytrain /= upper_bound
                Ytrain -= 0.5
            else:
                Xtrain = self.Xtrains[t]
                Ytrain = self.Ytrains[t]
            # 2. train a model for that dataset
            model = deepcopy(self.model)
            self.train_states[t] = self.trainer.train(
                model, Xtrain, Ytrain, t
            )  # do we wanna do anything smarter with the diagonal?

            # 3. set up prediction function for that timestep (incl. __call__ and JIT)
            if model.is_jax:
                self.pred_fns[t] = jax.jit(
                    partial(
                        self.trainer._pred,
                        train_state=self.train_states[t],
                        Xtrain=Xtrain,
                        Ytrain=Ytrain,
                        train_diag=None,
                    )
                )
            else:
                self.pred_fns[t] = model.predict
            # TODO: maybe something with the diagonal
            Xtrains.append(Xtrain)
            Ytrains.append(Ytrain)
        self.all_data = self.replay_buffer.get_all_data()
        self.Xtrains = np.array(Xtrains)
        self.Ytrains = np.array(Ytrains)

    def __call__(self, obs, time, action=None, all_q=False):
        if time >= self.horizon:
            if obs.ndim == 1:
                return 0.0
            else:
                return jnp.zeros(obs.shape[0])
        obs = jnp.atleast_2d(obs)
        obs = self.normalize_obs(obs)
        num_obs = obs.shape[0]
        upper_bound = self.horizon - time
        if action is None or all_q:
            all_actions = self.env.discrete_to_continuous(
                jnp.array(range(self.num_actions))
            )
            # do a standard value function thing (argmax)
            # obs num_obs x obs dim -> num_actions x num_obs x obs_dim
            tiled_obs = jnp.tile(obs, (self.num_actions, 1, 1))
            tiled_actions = jnp.tile(all_actions[:, None, :], (1, num_obs, 1))
            tiled_obs_actions = jnp.concatenate([tiled_obs, tiled_actions], axis=2)
            obs_action_dim = tiled_obs_actions.shape[2]
            flat_obs_actions = jnp.reshape(
                tiled_obs_actions, (self.num_actions * num_obs, obs_action_dim)
            )
            # TODO: if beta == 0, don't predict variances (idk if you can even avoid this)
            means, variances = self.pred_fns[time](flat_obs_actions)
            means = (means + 0.5) * upper_bound
            variances = variances * upper_bound**2
            means = jnp.reshape(means, (self.num_actions, num_obs))
            variances = jnp.reshape(variances, (self.num_actions, num_obs))
            fn_vals = means + variances * self.beta
            if all_q:
                return self.clip(fn_vals, time)
            values = fn_vals.max(axis=0)
            return self.clip(values, time)
        else:
            actions = jnp.atleast_2d(action)
            flat_obs_actions = jnp.concatenate([obs, actions], axis=1)
            means, variances = self.pred_fns[time](flat_obs_actions)
            means = (means + 0.5) * upper_bound
            variances = variances * upper_bound**2
            fn_vals = means + variances * self.beta
            return self.clip(fn_vals, time)

    def get_q_dataset(self, t):
        if self.use_all_data:
            data = self.replay_buffer.get_all_data()
        else:
            data = self.replay_buffer.get_data(time=t)
        # convert actions to continuous values
        if self.replay_buffer.discrete_actions:
            data.actions = self.env.discrete_to_continuous(data.actions)

        # get values for next states
        nxt_values = self(data.next_obs, t + 1)

        targets = nxt_values + data.rewards
        norm_obs = self.normalize_obs(data.obs)
        X = jnp.concatenate([norm_obs, data.actions], axis=1)
        return X, targets

    def normalize_obs(self, obs):
        obs = obs - self.env.observation_space.low
        diffs = self.env.observation_space.high - self.env.observation_space.low
        obs = (obs / diffs) * 2 - 1
        return obs

    def clip(self, vals, time):
        upper_bound = (
            self.horizon - time
        )  # since our time is zero based, we don't need the + 1 I believe.
        # for example, Q_{H - 1}(s, a) = r(s, a) \in [0, 1]
        return np.maximum(0, np.minimum(upper_bound, vals))

    @property
    def params(self):
        return [ts.params for ts in self.train_states]


class TQRLQRegressor(FiniteHorizonQRegressor):
    def __init__(self, sample_visited_states=False, nsamps=1000, **kwargs):
        super().__init__(**kwargs)
        self.sample_visited_states = sample_visited_states
        self.nsamps = nsamps

    def get_samples(self, observation_space):
        if self.sample_visited_states:
            next_obs = self.replay_buffer.get_all_data().next_obs
            try:
                idxes = np.random.choice(next_obs.shape[0], self.nsamps, replace=False)
                samps = next_obs[idxes, ...]
            except ValueError:
                samps = next_obs
        else:
            samps = np.array([observation_space.sample() for _ in range(self.nsamps)])
        return samps


class USQRegressor(TQRLQRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def choose_start(self, time, observation_space, action_space):
        upper_bound = self.horizon - time
        obs = np.array(self.get_samples(observation_space))
        discrete_actions = [action_space.sample() for _ in range(self.nsamps)]
        actions = self.env.discrete_to_continuous(np.array(discrete_actions))
        obs_actions = jnp.concatenate([obs, actions], axis=-1)
        means, variances = self.pred_fns[time](obs_actions)
        idx = np.argmax(variances)
        return obs[idx, :], discrete_actions[idx]


class UCBLCBQRegressor(TQRLQRegressor):
    def __init__(
        self,
        env,
        replay_buffer,
        beta_ucb,
        beta_lcb,
        nsamps,
        use_all_data,
        sample_visited_states=False,
    ):
        super().__init__(
            env=env,
            replay_buffer=replay_buffer,
            beta=None,
            use_all_data=use_all_data,
            sample_visited_states=sample_visited_states,
            nsamps=nsamps,
        )
        self.beta_ucb = beta_ucb
        self.beta_lcb = beta_lcb
        self.use_ucb = True
        self.train_states_ucb = [None] * self.horizon
        self.pred_fns_ucb = [None] * self.horizon
        self.train_states_lcb = [None] * self.horizon
        self.pred_fns_lcb = [None] * self.horizon

    @property
    def train_states(self):
        if self.use_ucb:
            return self.train_states_ucb
        else:
            return self.train_states_lcb

    @train_states.setter
    def train_states(self, x):
        # will just ignore
        pass

    @property
    def pred_fns(self):
        if self.use_ucb:
            return self.pred_fns_ucb
        else:
            return self.pred_fns_lcb

    @pred_fns.setter
    def pred_fns(self, x):
        # ignore
        pass

    @property
    def beta(self):
        if self.use_ucb:
            log.debug("using upper bound beta")
            return self.beta_ucb
        else:
            log.debug("using lower bound beta")
            return self.beta_lcb

    @beta.setter
    def beta(self, x):
        # we don't care about beta, so we'll just ignore it
        pass

    def fit_q_functions(self):
        self.use_ucb = True
        super().fit_q_functions()
        Xtrains_ucb = self.Xtrains
        Ytrains_ucb = self.Ytrains
        self.use_ucb = False
        super().fit_q_functions()
        Xtrains_lcb = self.Xtrains
        Ytrains_lcb = self.Ytrains
        self.Xtrains = [Xtrains_ucb, Xtrains_lcb]
        self.Ytrains = [Ytrains_ucb, Ytrains_lcb]
        self.use_ucb = True

    @property
    def params(self):
        return [
            (tsl.params, tsu.params)
            for tsl, tsu in zip(self.train_states_lcb, self.train_states_ucb)
        ]

    def choose_start(self, time, observation_space):
        samps = self.get_samples(observation_space)
        self.use_ucb = True
        ucb_vals = self(samps, time=time)
        self.use_ucb = False
        lcb_vals = self(samps, time=time)
        self.use_ucb = True
        gaps = ucb_vals - lcb_vals
        max_gap = np.argmax(gaps)
        return samps[max_gap, :]
