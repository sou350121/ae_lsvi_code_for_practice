import hydra
from hydra.utils import to_absolute_path
import logging
import pickle
from hydra.utils import instantiate, call
from ucb.replay_buffer import ReplayBuffer
from pathlib import Path
from ucb.util import (
    Dumper,
    execute_time_varying_policy,
    execute_random_policy,
    set_seed,
)
from copy import deepcopy

log = logging.getLogger(__name__)


@hydra.main(config_path="cfg", config_name="evaluation")
def main(config):
    set_seed(config.seed)
    dumper = Dumper(config.name)
    base_path = Path(to_absolute_path(config.expt_dir))
    info_path = Path(to_absolute_path(config.expt_dir)) / 'info.pkl'
    log.info(f"{base_path=}")
    log.info(f"{config.env.env_name=}")
    if info_path.exists():
        eval_policy(config, info_path, dumper)
    else:
        i = 0
        while True:
            seed_info_path = base_path / f'seed_{i}' / 'info.pkl'
            if not seed_info_path.exists():
                break
            log.info(f"{seed_info_path=}")
            eval_policy(config, seed_info_path, dumper)
            i += 1

def eval_policy(config, info_path, dumper):
    if config.env is not None:
        env = instantiate(config.env)
    else:
        raise NotImplementedError()
    model = instantiate(config.model)
    trainer = instantiate(config.trainer, params_path=config.env.gp_params)
    replay_buffer = ReplayBuffer(env.horizon)
    with info_path.open('rb') as f:
        data = pickle.load(f)
    q_eval = instantiate(
        config.q_eval,
        env=env,
        replay_buffer=replay_buffer,
        Xtrains=data['Xtrain'],
        Ytrains=data['Ytrain'],
    )
    q_eval.model = deepcopy(model)
    q_eval.trainer = deepcopy(trainer)
    q_eval.fit_q_functions()
    call(config.eval_fn, env=env, agent=q_eval, dumper=dumper)
    dumper.add("Eval ndata", replay_buffer.ndata)
    dumper.replace("Xtrain", q_eval.Xtrains)
    dumper.replace("Ytrain", q_eval.Ytrains)
    dumper.replace("all_data", q_eval.all_data)
    dumper.save()


if __name__ == "__main__":
    main()
