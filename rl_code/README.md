# infinite-MDP-UCB

## Installation
This codebase was written on Python 3.9 and builds on top of [JAX](https://github.com/google/jax), [Flax](https://github.com/google/flax), [TinyGP](https://tinygp.readthedocs.io/en/stable/index.html), [Hydra](https://hydra.cc/), and [gym](https://github.com/openai/gym). If you are using a virtual environment all dependencies should be able to be installed via `pip install -r requirements.txt`.

## Invocation
To run with the default settings simply try `python run.py`.

As a first step to understanding what is configurable and can be changed, try `python run.py -h`. This should print the entire config that is used to specify what the code should do.

To change something in the config, the general rule is to simply add a key=value argument to the call. For example `python run.py num_expl_episodes=20` would change the number of exploration episodes from the default of 10 to 20. When additional environments are added, you could also do `python run.py env=pendulum` (for example) to use the other environment.

**Alternatively**, we have included scripts for running bulk experiments with many of the RL algorithms we aimed to implement in this repository in `shell/`. Feel free to check out or execute these scripts to see how to call this code. If you remove the `launcher` and `seed=range` arguments from each line it is a valid invocation for a single experiment.

## Major components
The main component is a Q function table defined in `ucb/value_functions.py`. It currently is built on a sequence of Gaussian Processes built using JAX and tinyGP.

We also have a replay buffer which is pretty straightforward and a wrapper that converts environments with continuous actions to discrete ones.

## Outputs
We dump outputs to a subdirectory of `experiments/` that is based on the name of the experiment (default `"default"`) and the date + time of invocation. The directory will contain a pickle file `info.pkl` with python lists containing various performance measures from execution.
