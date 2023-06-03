"""
Fit GP hyperaparams on a fixed dataset.
"""
import h5py
import hydra
import logging
from hydra.utils import instantiate, to_absolute_path
import jax.numpy as jnp
from tqdm import trange
import numpy as np
import pickle
import logging
from ucb.util import Dumper

from ucb.replay_buffer import ReplayBuffer
from ucb.util import Dumper


log = logging.getLogger(__name__)


@hydra.main(config_path="cfg", config_name="gp_fitting")
def main(config):
    data_path = to_absolute_path(config.env.fit_data_path)
    if "hdf5" in data_path:
        data = {}
        with h5py.File(data_path, "r") as hdata:
            for k, v in hdata.items():
                data[k] = v[()]
    else:
        data = np.load(data_path)
    dumper = Dumper(config.name)
    env = instantiate(config.env)
    replay_buffer = ReplayBuffer(env.horizon)
    replay_buffer.discrete_actions = False
    ndata = data["obs"].shape[0]
    for i in range(config.skip_size, ndata - config.test_set_size):
        obs = data["obs"][i, :]
        action = data["action"][i, :]
        next_obs = data["next_obs"][i, :]
        reward = data["reward"][i]
        time = i % env.horizon
        replay_buffer.add_point(obs, action, next_obs, reward, time)

    q = instantiate(config.q, env=env, replay_buffer=replay_buffer)
    params = q.params
    log.debug(params)
    with open("params.pkl", "wb") as f:
        pickle.dump(params, f)
    print(params)
    assert config.test_set_size % env.horizon == 0
    test_obs = data["obs"][-config.test_set_size :].reshape(
        (config.test_set_size // env.horizon, env.horizon, -1)
    )
    test_action = data["action"][-config.test_set_size :].reshape(
        (config.test_set_size // env.horizon, env.horizon, -1)
    )
    test_next_obs = data["next_obs"][-config.test_set_size :].reshape(
        (config.test_set_size // env.horizon, env.horizon, -1)
    )
    test_reward = data["reward"][-config.test_set_size :].reshape(
        (config.test_set_size // env.horizon, env.horizon)
    )
    errs = []
    for t in trange(env.horizon):
        nxt_values = q(test_next_obs[:, t, :], t + 1)
        targets = nxt_values + test_reward[:, t]
        values = q(test_obs[:, t, :], t, action=test_action[:, t, :])[:, 0]
        mse = np.mean(np.square(values - targets))
        errs.append(mse)
        log.debug(f"{targets=}")
        log.debug(f"{values=}")
        log.debug(f"{mse=}")
    log.debug(f"{errs=}")
    mean_err = np.mean(errs)
    log.info(f"{mean_err=}")
    if not np.isfinite(mean_err):
        return 20.0
    with open("params.pkl", "wb") as f:
        pickle.dump(params, f)
    dumper.add("Eval ndata", replay_buffer.ndata)
    dumper.replace("Xtrain", q_eval.Xtrains)
    dumper.replace("Ytrain", q_eval.Ytrains)
    dumper.replace("all_data", q_eval.all_data)
    dumper.save()
    return float(mean_err)


if __name__ == "__main__":
    main()
