"""
A fusion environment for fitting the gym framework.

"""
from typing import Dict, Sequence, Tuple, Optional
from pathlib import Path
import pickle as pkl

from dynamics_toolbox.models.abstract_model import AbstractModel
from dynamics_toolbox.utils.storage.qdata import load_from_hdf5
from dynamics_toolbox.utils.storage.model_storage import load_model_from_log_dir
import gym
import numpy as np
import torch
from tqdm import tqdm

from fusion_control.envs.rewards import RewardFunction, TrackingReward
from fusion_control.envs.target_distributions import (
    TargetDistribution,
    UniformTargetDistribution,
)
from fusion_control.util.data_util import sort_by_shot


DEFAULT_MODEL_PATH1 = BASE_PATH + "/models/dynamics/v1/scalar_simplex/2022-03-05/train"
DEFAULT_DATA_PATH1 = BASE_PATH + "/data/v1/scalar/2022-03-05"
DEFAULT_MODEL_PATH2 = BASE_PATH + "/models/dynamics/tipenv/"
DEFAULT_DATA_PATH2 = BASE_PATH + "/data/tipenv"
DEFAULT_SHOT_ID = 161306
DEFAULT_SHOT_START = 3000


class FusionGymEnv(gym.Env):
    def __init__(
        self,
        dynamics_model: AbstractModel,
        reward_function: RewardFunction,
        target_distribution: TargetDistribution,
        data_path: str,
        states_in_obs: Sequence[str],
        actuators_in_obs: Sequence[str],
        action_space: Sequence[str],
        actuator_bounds: Dict[str, Tuple[float, float]],
        uncertainty_in_dynamics: bool = False,
        shot_id: int = DEFAULT_SHOT_ID,
        start_time: int = DEFAULT_SHOT_START,
        fixed_target: Optional[np.ndarray] = None,
        state_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        horizon: int = 20,
        targets_in_obs: bool = True,
        silent: bool = True,
        uniform_start: bool = False,
        shifted: bool = False,
    ):
        """Constructor.
        Args:
            uncertainty_in_dynamics: Whether there should be uncertainty in the
                dynamics.
            dynamics_model: The model of the dynamics.
            reward_function: The reward function to use.
            data_path: Path to the data containing tr.hdf5, te.hdf5, info.pkl
                and optionally val.hdf5.
            states_in_obs: The members of the state space that are observed by
                the policy.
            acts_in_obs: The members of the actuator space that are observed by the
                policy.
            action_space: Names of the actuators that the policy can control.
                It is assumed that the actions here will be a change in actuator and
                thus all of the fields should have the suffix "_velocity".
            actuator_bounds: The bounds for actuators. the key takes the form
                <action> and <action>_velocity. Every action in action_space should
                have a corresponding <action> and <action>_velocity.
            info: Information about the dataset.
            max_horizon: The maximum horizon we can be expected to use for unrolling.
            state_bounds: The bounds to impose on predicted states. the key takes the
               form <state> and <state>_velocity. If not provided no bounding will
                occur.
            targets_in_obs: Whether targets should be in observations.
            source_types: Data source types to be loaded in. There should be
                corresponding files for each of these.
            bound_tinj_by_pinj: If a number is provided and both tinj and pinj are
                in the actuator space, then tinj cannot be any greater than this
                number provided * pinj. This coefficient assumes pinj is in MW and
                tinj is in Nm. The default number was found after doing analysis
                with the beams set at 75kV and holds for the beams going up to 7MW
                of power. Past this point, there is also a lower bound that applies.
            max_start_idx: The maximum time steps into the shot that can be considered
                a starting point for collecting trajectories. Not a tight upper bound
        """
        self.uncertainty_in_dynamics = uncertainty_in_dynamics
        self.shot_id = shot_id
        self.start_time = start_time
        self.fixed_target = fixed_target
        self.horizon = horizon
        self.silent = silent
        self.dynamics_model = dynamics_model
        self.reward_function = reward_function
        self.target_distribution = target_distribution
        self.targets_in_obs = targets_in_obs
        self._print("Loading in data...")
        self._load_in_data(data_path)
        self._print("Pre-processing for faster indexing...")
        self._preprocess_indices(states_in_obs, actuators_in_obs, action_space)
        self._print("Pre-processing for faster bounding...")
        self._preprocess_bounds(actuator_bounds, action_space, state_bounds)
        # Set the dimensionality based on settings.
        self.observation_dim = (
            len(self.state_ob_idxs)
            + len(self.actuator_ob_idxs)
            + targets_in_obs * self.reward_function.num_targets
        )
        self.act_dim = len(action_space)
        self.observation_space = gym.spaces.Box(
            -5 * np.ones(self.observation_dim), 5 * np.ones(self.observation_dim)
        )
        self.action_space = gym.spaces.Box(
            -1 * np.ones(self.act_dim), np.ones(self.act_dim)
        )
        self._print("Environment loaded!")
        self.curr_obs = None
        self.shot_id = shot_id
        self.start_time = start_time
        self.uniform_start = uniform_start
        self.shifted = shifted

    def reset(self, obs: Optional[np.ndarray] = None) -> None:
        """Reset the environment.

        Args:
            obs: The observation to start from. k

        This will ensure that new random trajectories are drawn.
        """
        if self.shot_id is None:
            self.replay_idx = np.random.randint(len(self.shot_ids))
        else:
            self.replay_idx = self.shot_ids.index(self.shot_id)
        if self.start_time is None:
            self.start_idx = 0
        else:
            self.start_idx = np.argmin(
                np.abs(self.start_time - self.trajectories[self.replay_idx]["time"])
            )
        self.states = np.array(
            [
                self.trajectories[self.replay_idx]["states"][self.start_idx]
                for _ in range(self.horizon + 1)
            ]
        )
        self.actuators = np.array(
            [
                self.trajectories[self.replay_idx]["actuators"][self.start_idx]
                for _ in range(self.horizon + 1)
            ]
        )
        self.next_actuators = np.zeros(
            (
                self.horizon,
                self.trajectories[self.replay_idx]["next_actuators"].shape[-1],
            )
        )
        self.tidx = 0
        if obs is not None:
            i1 = len(self.state_ob_idxs)
            i2 = i1 + len(self.actuator_ob_idxs)
            self.states[0, self.state_ob_idxs] = obs[:i1]
            self.actuators[0, self.actuator_ob_idxs] = obs[i1:i2]
            self.target = obs[i2:]
        elif self.shifted:
            offset = 0.04
            self.states = self.states + offset
            self.actuators = self.actuators + offset
            if self.fixed_target is None:
                self.target = self.target_distribution.draw_targets(1).flatten()
            else:
                self.target = self.fixed_target
            self.target = self.target - offset
        elif self.uniform_start:
            obs = self.observation_space.sample()
            i1 = len(self.state_ob_idxs)
            i2 = i1 + len(self.actuator_ob_idxs)
            self.states[0, self.state_ob_idxs] = obs[:i1]
            self.actuators[0, self.actuator_ob_idxs] = obs[i1:i2]
            self.target = obs[i2:]
        else:
            if self.fixed_target is None:
                self.target = self.target_distribution.draw_targets(1).flatten()
            else:
                self.target = self.fixed_target
        if self.uncertainty_in_dynamics:
            self.dynamics_model.reset()
        else:
            # Make sure that the same dynamics model is used always.
            self.dynamics_model._curr_sample = (
                torch.ones(
                    (
                        1,
                        self.dynamics_model._num_layers,
                        self.dynamics_model._num_vertices,
                    )
                )
                / self.dynamics_model._num_vertices
            )
        return self._form_observation(self.states[0], self.actuators[0], self.target)

    def step(self, action: np.ndarray):
        """
        Make a step in the environment.

        Args:
            action: The action to take should have shape (action_dim,)

        Returns:
            next_observation, reward, terminal, {}
        """
        # Update the actuators.
        self.actuators, self.next_actuators = self._update_actuators(
            self.actuators,
            self.next_actuators,
            action,
            self.tidx,
        )
        # Make predictions.
        model_out, _ = self.dynamics_model.predict(
            np.hstack(
                [
                    self.states[self.tidx],
                    self.actuators[self.tidx],
                    self.next_actuators[self.tidx],
                ]
            ).reshape(1, -1)
        )
        self.states = self._update_states(self.states, model_out, self.tidx)
        obs = self._form_observation(
            self.states[self.tidx + 1],
            self.actuators[self.tidx + 1],
            self.target,
        )
        reward = self.reward_function.get_reward(
            self.states[self.tidx + 1].reshape(1, -1),
            self.info,
            self.target.reshape(1, -1),
        )
        self.tidx += 1
        return obs, float(reward), self.tidx >= self.horizon, {}

    def unscale_actions(self, actions: np.ndarray) -> np.ndarray:
        """Transform actions from [-1, 1] back into original amount.
        Args:
            actions: The actions in [-1, 1].
        Returns: The unscaled actions.
        """
        lows, highs = self.bounds["actions"]["velocity"]
        unscaled = (actions + 1) / 2
        return unscaled * (highs - lows) + lows

    def get_reward(self, sa: np.ndarray, next_obs: np.ndarray) -> float:
        """Get the reward.

        Args:
            sa: The state and action concatted. This has no affect on the reward.
            next_obs: The next observation observed. This should have shape
                (num_samples, obs_dim)

        Returns:
            Float of the reward.
        """
        if len(next_obs.shape) == 1:
            next_obs = next_obs.reshape(1, -1)
        stripped_obs = next_obs[:, : -self.reward_function.num_targets]
        targets = next_obs[:, -self.reward_function.num_targets :]
        states = np.array(
            [self.trajectories[0]["states"][0] for _ in range(len(stripped_obs))]
        )
        states[:, self.state_ob_idxs] = stripped_obs[:, : len(self.state_ob_idxs)]
        return self.reward_function.get_reward(states, self.info, targets)

    def _form_observation(
        self,
        state: np.ndarray,
        actuator: np.ndarray,
        targets: Optional[np.ndarray],
    ) -> np.ndarray:
        """Form observation from the underlying state and actuator.

        Args:
            state: The current underlying state with shape (num_states, state dim).
            actuator: The current actuator configuration with shape
                (num_actuators, actuator dim)
            targets: The targets to append if given.

        Returns:
            The observation from the state and actuator.
        """
        obs = np.hstack([state[self.state_ob_idxs], actuator[self.actuator_ob_idxs]])
        if targets is not None:
            obs = np.hstack([obs.flatten(), targets])
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _form_model_input(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        actuator: np.ndarray,
    ) -> np.ndarray:
        """Form model input from observation and action."""
        state = self.trajectories[self.replay_idx]["states"][self.start_idx]
        actuator = self.trajectories[self.replay_idx]["actuators"][self.start_idx]
        state[self.state_ob_idxs] = obs[: -self.reward_function.num_targets]
        # Update the actuators.
        deltas = self.unscale_actions(act)
        deltas = np.clip(
            deltas,
            np.maximum(
                self.bounds["actions"]["velocity"][0],
                self.bounds["actions"]["position"][0]
                - actuator[self.actuator_act_idxs],
            ),
            np.minimum(
                self.bounds["actions"]["velocity"][1],
                self.bounds["actions"]["position"][1]
                - actuator[self.actuator_act_idxs],
            ),
        )
        next_actuators = np.zeros(
            self.trajectories[self.replay_idx]["next_actuators"].shape[1]
        )
        next_actuators[self.nxts_act_idxs] = deltas
        if len(self.actuator_vel_idxs):
            actuator[self.actuator_vel_idxs] = next_actuators
        actuator[self.actuator_posn_idxs] = (
            actuator[self.actuator_posn_idxs] + next_actuators
        )
        # Create the model input.
        model_input = np.hstack([state, actuator])
        return model_input, actuator

    def _preprocess_indices(
        self,
        states_in_obs: Sequence[str],
        actuators_in_obs: Sequence[str],
        action_space: Sequence[str],
    ):
        """Do work to identify indices for later computations.

        Args:
            states_in_obs: The members of the state space that are observed by
                the policy.
            acts_in_obs: The members of the actuator space that are observed by the
                policy.
            action_space: Names of the actuators that the policy can control.
                It is assumed that the actions here will be a change in actuator and
                thus all of the fields should have the suffix "_velocity".
        """
        # The indices for things in the observation space.
        self.state_ob_idxs = [self.info["state_space"].index(o) for o in states_in_obs]
        self.actuator_ob_idxs = [
            self.info["actuator_space"].index(a) for a in actuators_in_obs
        ]
        # The indices for things in the action space.
        self.actuator_act_idxs = [
            self.info["actuator_space"].index(a[: -len("_velocity")])
            for a in action_space
        ]
        self.nxts_act_idxs = [
            self.info["next_actuator_space"].index(a) for a in action_space
        ]
        if "pinj" in self.info["actuator_space"]:
            self.pinj_actuator_idx = self.info["actuator_space"].index("pinj")
            self.pinj_next_actuator_idx = self.info["next_actuator_space"].index(
                "pinj_velocity"
            )
        else:
            self.pinj_actuator_idx = None
            self.pinj_next_actuator_idx = None
        if "tinj" in self.info["actuator_space"]:
            self.tinj_actuator_idx = self.info["actuator_space"].index("tinj")
            self.tinj_next_actuator_idx = self.info["next_actuator_space"].index(
                "tinj_velocity"
            )
        else:
            self.tinj_actuator_idx = None
            self.tinj_next_actuator_idx = None
        # Indices to differentiate between positions and velocities in actuators.
        self.actuator_posn_idxs, self.actuator_vel_idxs = [], []
        for i in range(len(self.info["actuator_space"])):
            if "velocity" in self.info["actuator_space"][i]:
                if self.info["actuator_space"][i] in action_space:
                    self.actuator_vel_idxs.append(i)
            else:
                self.actuator_posn_idxs.append(i)
        # Indices to differentiate between positions and velocities in actuators.
        self.state_posn_idxs, self.state_vel_idxs = [], []
        for i in range(len(self.info["state_space"])):
            if "velocity" in self.info["state_space"][i]:
                self.state_vel_idxs.append(i)
            else:
                self.state_posn_idxs.append(i)
        # Indices to differentiate between positions and velocities in actuators.
        self.ob_posn_idxs, self.ob_vel_idxs = [], []
        for i in range(len(self.info["state_space"])):
            if self.info["state_space"][i] in states_in_obs:
                if "velocity" in self.info["state_space"][i]:
                    self.ob_vel_idxs.append(i)
                else:
                    self.ob_posn_idxs.append(i)
        # The indices of the next states that have velocity in state representation.
        self.pred_vel_idxs = [
            i
            for i in range(len(self.info["next_state_space"]))
            if self.info["next_state_space"][i] in self.info["state_space"]
        ]
        self.obs_vel_idxs = [
            i
            for i in range(len(self.info["next_state_space"]))
            if self.info["next_state_space"][i] in self.info["state_space"]
            and self.info["next_state_space"][i] in states_in_obs
        ]

    def _preprocess_bounds(
        self,
        actuator_bounds: Dict[str, Tuple[float, float]],
        action_space: Sequence[str],
        state_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        """Prprocessing for the bounds.

        Args:
            actuator_bounds: The bounds for actuators. the key takes the form
                <action> and <action>_velocity. Every action in action_space should
                have a corresponding <action> and <action>_velocity.
            action_space: Names of the actuators that the policy can control.
                It is assumed that the actions here will be a change in actuator and
                thus all of the fields should have the suffix "_velocity".
            state_bounds: The bounds to impose on predicted states. the key takes the
                form <state> and <state>_velocity.
        """
        # Set up the action bounds.
        for actname in action_space:
            assert actname in actuator_bounds, f"{actname} not found in action bounds."
            assert (
                actname[: -len("_velocity")] in actuator_bounds
            ), f'{actname[:-len("_velocity")]} not found in action bounds.'
        action_bounds = {"position": [[], []], "velocity": [[], []]}
        action_bounds["position"][0] = np.array(
            [actuator_bounds[a[: -len("_velocity")]][0] for a in action_space]
        ).reshape(1, -1)
        action_bounds["position"][1] = np.array(
            [actuator_bounds[a[: -len("_velocity")]][1] for a in action_space]
        ).reshape(1, -1)
        action_bounds["velocity"][0] = np.array(
            [actuator_bounds[a][0] for a in action_space]
        ).reshape(1, -1)
        action_bounds["velocity"][1] = np.array(
            [actuator_bounds[a][1] for a in action_space]
        ).reshape(1, -1)
        self.bounds = {"actions": action_bounds}
        # Set up the state bounds.
        if state_bounds is not None:
            state_dict = {"position": [[], []], "velocity": [[], []]}
            for vel_name in self.info["next_state_space"]:
                pos_name = vel_name[: -len("_velocity")]
                for key, key_type in [(pos_name, "position"), (vel_name, "velocity")]:
                    if key in state_bounds:
                        lower, upper = state_bounds[key]
                    else:
                        lower, upper = -np.inf, np.inf
                    state_dict[key_type][0].append(lower)
                    state_dict[key_type][1].append(upper)
            for k, v in state_dict.items():
                state_dict[k] = [np.array(b).reshape(1, -1) for b in v]
            self.bounds["states"] = state_dict

    def _load_in_data(self, data_path: str):
        """Load in the appropriate data from the data path.

        Args:
            data_path: Path to the data containing tr.hdf5, te.hdf5, info.pkl
                and optionally val.hdf5.
        """
        base_path = Path(data_path)
        with open(str(base_path / "info.pkl"), "rb") as f:
            self.info = pkl.load(f)
        dtype = "full"
        proc_dict = self._process_trajectories(str(base_path / f"{dtype}.hdf5"))
        for k, v in proc_dict.items():
            setattr(self, k, v)

    def _process_trajectories(
        self,
        data_file_path: str,
    ) -> Dict[str, np.ndarray]:
        """Process trajectories to make start states and actuator replays.

        Args:
            data_file_path: Path to the data file.
            trajs: The trajectories to process.
            max_horizon: The maximum horizon allowed by the environment.

        Returns:
            The start states to be done and the actuator replays.
        """
        data = load_from_hdf5(data_file_path)
        # TODO: This automatically chops off after a time jump happens.
        # more code is needed to allow to jump in time if we even want that.
        trajs = sort_by_shot(data, self.info, chop_off_time_jump=True)
        shot_ids = [t["shotnum"] for t in trajs]
        start_states = []
        for t in trajs:
            traj_starts = []
            for idx in range(1):
                if idx >= len(t["states"]):
                    traj_starts.append(t["states"][-1])
                else:
                    traj_starts.append(t["states"][idx])
            start_states.append(np.array(traj_starts))
        start_states = np.array(start_states)
        actuator_replays = []
        next_actuator_replays = []
        if not self.silent:
            pbar = tqdm(total=len(trajs))
        amt_info_needed = 50
        for traj in trajs:
            if traj["actuators"].shape[0] < amt_info_needed:
                pad_amount = amt_info_needed - traj["actuators"].shape[0]
                actuator_replays.append(
                    np.vstack(
                        [
                            traj["actuators"],
                            traj["actuators"][-1]
                            .repeat(pad_amount)
                            .reshape(pad_amount, -1),
                        ]
                    )
                )
                next_actuator_replays.append(
                    np.vstack(
                        [
                            traj["next_actuators"],
                            np.zeros((pad_amount - 1, traj["next_actuators"].shape[1])),
                        ]
                    )
                )
            else:
                actuator_replays.append(traj["actuators"][:amt_info_needed])
                next_actuator_replays.append(
                    traj["next_actuators"][: amt_info_needed - 1]
                )
            if not self.silent:
                pbar.update(1)
        if not self.silent:
            pbar.close()
        actuator_replays = np.array(actuator_replays)
        next_actuator_replays = np.array(next_actuator_replays)
        return {
            "trajectories": trajs,
            "start_states": start_states,
            "actuator_replays": actuator_replays,
            "next_actuator_replays": next_actuator_replays,
            "shot_ids": [si[0] for si in shot_ids],
        }

    def _update_actuators(
        self,
        actuators: np.ndarray,
        next_actuators: np.ndarray,
        act: np.ndarray,
        time_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Update the actuator information to include the policy's actions.

        Args:
            actuators: The history of actuators as shape
                (num_unrolls, horizon + 1, actuator_dim)
            next_actuators: The history of next actutor change to be applied.
                Has shape (num_unrolls, horizon, num actuator signals).
            act: The actions the policy give at this time step with shape
                (num_unrolls, action dimension)
            time_idx: The index of the time step this action was made.

        Returns:
            actuators but also altered in place.
        """
        deltas = self.unscale_actions(act)
        deltas = np.clip(
            deltas,
            np.maximum(
                self.bounds["actions"]["velocity"][0],
                self.bounds["actions"]["position"][0]
                - actuators[time_idx, self.actuator_act_idxs],
            ),
            np.minimum(
                self.bounds["actions"]["velocity"][1],
                self.bounds["actions"]["position"][1]
                - actuators[time_idx, self.actuator_act_idxs],
            ),
        )
        next_actuators[time_idx, self.nxts_act_idxs] = deltas
        if len(self.actuator_vel_idxs):
            actuators[time_idx + 1, self.actuator_vel_idxs] = deltas
        actuators[time_idx + 1, self.actuator_posn_idxs] = (
            actuators[time_idx, self.actuator_posn_idxs] + next_actuators[time_idx]
        )
        return actuators, next_actuators

    def _update_states(
        self,
        states: np.ndarray,
        preds: np.ndarray,
        time_idx: int,
    ) -> np.ndarray:
        """Update the states to reflect the model predictions.

        Args:
            states: The history of all the states so far and to come.
                Has shape (num_unrolls, horizon + 1, state_dim)
            preds: The predictions made by the model should have shape
                (num_unrolls, number of state signals). In most cases
                last dimension will be half of states because states
                will often have position + velocity.
            time_idx: The current time index.
        """
        # Optionally make bounds on the predictions..
        if "states" in self.bounds:
            # Form the minimum bound.
            min_bound = self.bounds["states"]["velocity"][0].reshape(1, -1)
            min_bound = np.repeat(min_bound, len(states), axis=0)
            posn_min_bd = self.bounds["states"]["position"][0].reshape(1, -1)
            posn_min_bd = np.repeat(posn_min_bd, len(states), axis=0)
            posn_diffs = posn_min_bd - states[time_idx, self.state_posn_idxs]
            # We only count if the signal is above the low bound. However, if this
            # is the case don't allow the signal to drop anymore.
            invalid_idxs = np.argwhere(posn_diffs >= 0)
            posn_diffs[invalid_idxs[:, 0], invalid_idxs[:, -1]] = 0
            min_bound = np.maximum(min_bound, posn_diffs)
            # Form the maximum bound.
            max_bound = self.bounds["states"]["velocity"][1].reshape(1, -1)
            max_bound = np.repeat(max_bound, len(states), axis=0)
            posn_max_bd = self.bounds["states"]["position"][1].reshape(1, -1)
            posn_max_bd = np.repeat(posn_max_bd, len(states), axis=0)
            posn_diffs = posn_max_bd - states[time_idx, self.state_posn_idxs]
            # We only count if the signal is below the high bound. We don't allow the
            # signal to grow anymore if this is the case however.
            invalid_idxs = np.argwhere(posn_diffs <= 0)
            posn_diffs[invalid_idxs[:, 0], invalid_idxs[:, -1]] = 0
            max_bound = np.minimum(max_bound, posn_diffs)
            preds = np.clip(preds, min_bound, max_bound)
        # Only change the pred_vel_idxs.
        if len(self.obs_vel_idxs):
            states[time_idx + 1, self.ob_vel_idxs] = preds[0, self.obs_vel_idxs]
        states[time_idx + 1, self.ob_posn_idxs] = (
            states[time_idx, self.ob_posn_idxs] + preds[0, self.obs_vel_idxs]
        )
        return states

    def _print(self, statement: str) -> None:
        """Utility for printing if not silent.

        Args:
            statement: What to print.
        """
        if not self.silent:
            print(statement)


class BetaTrackingGymEnv(FusionGymEnv):
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH1,
        data_path: str = DEFAULT_DATA_PATH1,
        uncertainty_in_dynamics: bool = False,
        shot_id: int = DEFAULT_SHOT_ID,
        start_time: int = DEFAULT_SHOT_START,
        fixed_target: Optional[np.ndarray] = None,
        horizon: int = 20,
        targets_in_obs: bool = True,
    ):
        """Constructor see arguments above."""
        dynamics_model = load_model_from_log_dir(str(model_path))
        reward_function = TrackingReward(
            track_signals=["betan_EFIT01"],
            track_coefficients=[1],
        )
        target_distribution = UniformTargetDistribution(
            target_lows=[-0.1759333],  # This is a low of 1.5
            target_highs=[1.08474],  # This is a high of 2.5
        )
        states_in_obs = ["betan_EFIT01", "betan_EFIT01_velocity"]
        actuators_in_obs = ["pinj", "pinj_velocity"]
        action_space = ["pinj_velocity"]
        actuator_bounds = {
            "pinj": [-1.424, 1.696],
            "pinj_velocity": [-0.5567, 0.6844],
        }
        super().__init__(
            dynamics_model=dynamics_model,
            reward_function=reward_function,
            target_distribution=target_distribution,
            data_path=data_path,
            states_in_obs=states_in_obs,
            actuators_in_obs=actuators_in_obs,
            action_space=action_space,
            actuator_bounds=actuator_bounds,
            uncertainty_in_dynamics=uncertainty_in_dynamics,
            shot_id=shot_id,
            start_time=start_time,
            horizon=horizon,
            fixed_target=fixed_target,
            targets_in_obs=targets_in_obs,
        )


class BetaRotationTrackingGymEnv(FusionGymEnv):
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH1,
        data_path: str = DEFAULT_DATA_PATH1,
        uncertainty_in_dynamics: bool = False,
        shot_id: int = DEFAULT_SHOT_ID,
        start_time: int = DEFAULT_SHOT_START,
        fixed_target: Optional[np.ndarray] = None,
        horizon: int = 20,
        targets_in_obs: bool = True,
        uniform_start: bool = False,
        shifted: bool = False,
    ):
        """Constructor see arguments above."""
        dynamics_model = load_model_from_log_dir(str(model_path))
        reward_function = TrackingReward(
            track_signals=["betan_EFIT01", "rotation_0"],
            track_coefficients=[1, 1],
        )
        target_distribution = UniformTargetDistribution(
            target_lows=[-0.1759333, -0.3475],  # This is a low of 1.5
            target_highs=[1.08474, 1.3775],  # This is a high of 2.5
        )
        states_in_obs = [
            "betan_EFIT01",
            "rotation_0",
            "betan_EFIT01_velocity",
            "rotation_0_velocity",
        ]
        actuators_in_obs = ["pinj", "tinj", "pinj_velocity", "tinj_velocity"]
        action_space = ["pinj_velocity", "tinj_velocity"]
        actuator_bounds = {
            "pinj": [-1.424, 1.696],
            "tinj": [-1.7613, 1.8382],
            "pinj_velocity": [-0.5567, 0.6844],
            "tinj_velocity": [-0.6975, 0.7395],
        }
        self.reward_bounds = [-6, 0]
        super().__init__(
            dynamics_model=dynamics_model,
            reward_function=reward_function,
            target_distribution=target_distribution,
            data_path=data_path,
            states_in_obs=states_in_obs,
            actuators_in_obs=actuators_in_obs,
            action_space=action_space,
            actuator_bounds=actuator_bounds,
            uncertainty_in_dynamics=uncertainty_in_dynamics,
            shot_id=shot_id,
            start_time=start_time,
            horizon=horizon,
            fixed_target=fixed_target,
            targets_in_obs=targets_in_obs,
            uniform_start=uniform_start,
            shifted=shifted,
        )


class BetaRotationTrackingMultiGymEnv(FusionGymEnv):
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH2,
        data_path: str = DEFAULT_DATA_PATH2,
        uncertainty_in_dynamics: bool = False,
        shot_id: int = None,
        start_time: int = None,
        fixed_target: Optional[np.ndarray] = None,
        horizon: int = 20,
        targets_in_obs: bool = True,
    ):
        """Constructor see arguments above."""
        dynamics_model = load_model_from_log_dir(str(model_path))
        reward_function = TrackingReward(
            track_signals=["betan_EFIT01", "rotation_0"],
            track_coefficients=[1, 1],
        )
        target_distribution = UniformTargetDistribution(
            target_lows=[-1, -1],  # This is a low of 1.5
            target_highs=[1.5, 2],  # This is a high of 2.5
        )
        states_in_obs = [
            "betan_EFIT01",
            "rotation_0",
            "betan_EFIT01_velocity",
            "rotation_0_velocity",
        ]
        actuators_in_obs = ["pinj", "tinj", "pinj_velocity", "tinj_velocity"]
        action_space = ["pinj_velocity", "tinj_velocity"]
        actuator_bounds = {
            "pinj": [-1.424, 1.696],
            "tinj": [-1.7613, 1.8382],
            "pinj_velocity": [-0.5567, 0.6844],
            "tinj_velocity": [-0.6975, 0.7395],
        }
        super().__init__(
            dynamics_model=dynamics_model,
            reward_function=reward_function,
            target_distribution=target_distribution,
            data_path=data_path,
            states_in_obs=states_in_obs,
            actuators_in_obs=actuators_in_obs,
            action_space=action_space,
            actuator_bounds=actuator_bounds,
            uncertainty_in_dynamics=uncertainty_in_dynamics,
            shot_id=shot_id,
            start_time=start_time,
            horizon=horizon,
            fixed_target=fixed_target,
            targets_in_obs=targets_in_obs,
        )


if __name__ == "__main__":
    env = BetaRotationTrackingGymEnv()
    print(f"{env.horizon=}")
    ntrials = 1000
    min_rew = np.inf
    max_rew = -np.inf
    for i in range(ntrials):
        obs = env.reset()
        for t in range(env.horizon):
            a = env.action_space.sample()
            no, rew, _, _ = env.step(a)
            min_rew = min(min_rew, rew)
            max_rew = max(max_rew, rew)
    print(f"{min_rew=}")
    print(f"{max_rew=}")
