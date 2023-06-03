import hydra
import logging
from hydra.utils import instantiate, call
from ucb.replay_buffer import ReplayBuffer
from ucb.util import (
    Dumper,
    execute_time_varying_policy,
    execute_random_policy,
    set_seed,
)
from copy import deepcopy

log = logging.getLogger(__name__)


@hydra.main(config_path="cfg", config_name="main")
def main(config):
    set_seed(config.seed)
    dumper = Dumper(config.name)
    env = instantiate(config.env)
    replay_buffer = ReplayBuffer(env.horizon)
    model = instantiate(config.model)
    trainer = instantiate(config.trainer, params_path=config.env.gp_params)

    for i in range(config.num_expl_episodes):
        if i < config.random_burn_in:
            execute_random_policy(env, replay_buffer, dumper)
            continue
        q_expl = instantiate(
            config.q_expl,
            env=env,
            replay_buffer=replay_buffer,
        )
        q_expl.model = deepcopy(model)
        q_expl.trainer = deepcopy(trainer)
        q_expl.fit_q_functions()
        call(
            config.expl_fn,
            env=env,
            agent=q_expl,
            replay_buffer=replay_buffer,
            dumper=dumper,
        )
        q_eval = instantiate(
            config.q_eval,
            env=env,
            replay_buffer=replay_buffer,
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
        log.info(f"Iteration {i} complete")


if __name__ == "__main__":
    main()
