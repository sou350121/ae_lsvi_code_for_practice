"""
Make a dataset from performing random actions.
"""
import h5py
import hydra
import numpy as np


@hydra.main(config_path="../cfg", config_name="make_random_policy_dataset")
def main(config):
    env = hydra.utils.instantiate(config.env)
    data = {k: [] for k in ["obs", "action", "next_obs", "reward"]}
    for _ in range(config.num_rollouts):
        obs = env.reset()
        for t in range(env.horizon):
            action = env.action_space.sample()
            next_obs, rew, _, _ = env.step(action)
            data["obs"].append(obs)
            data["action"].append([action])
            data["next_obs"].append(next_obs)
            data["reward"].append(rew)
            obs = next_obs
    data = {k: np.array(v) for k, v in data.items()}
    with h5py.File("dataset.hdf5", "w") as hdata:
        for k, v in data.items():
            hdata.create_dataset(k, data=v)


if __name__ == "__main__":
    main()
