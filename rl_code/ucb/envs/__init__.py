import logging
from gym.envs.registration import register
from ucb.envs.pendulum import PendulumEnv
from ucb.envs.cartpole import CartPoleSwingUpEnv
from ucb.envs.mountaincar import DenseMountainCarEnv
from ucb.envs.weird_gain import WeirdGainEnv
from ucb.envs.navigation import NavigationEnv

log = logging.getLogger(__name__)

# register each environment we wanna use
register(
    id="bacpendulum-v0",
    entry_point=PendulumEnv,
)

register(
    id="cartpoleswingup-v0",
    entry_point=CartPoleSwingUpEnv,
)

register(
    id="uniformcartpoleswingup-v0",
    entry_point=CartPoleSwingUpEnv,
    kwargs={"uniform_start": True},
)

register(
    id="cartpoleswingup-dense-v0",
    entry_point=CartPoleSwingUpEnv,
    kwargs={"use_dense_rewards": True},
)

register(
    id="denseuniformcartpoleswingup-v0",
    entry_point=CartPoleSwingUpEnv,
    kwargs={"uniform_start": True, "use_dense_rewards": True},
)

register(
    id="dense-shift-cartpoleswingup-v0",
    entry_point=CartPoleSwingUpEnv,
    kwargs={"shift_start": True, "use_dense_rewards": True},
)

register(
    id="cartpoleswingup-dense-tight-v0",
    entry_point=CartPoleSwingUpEnv,
    kwargs={"use_dense_rewards": True, "restrict_obs_space": True},
)

register(
    id="densemountaincar-v0",
    entry_point=DenseMountainCarEnv,
    kwargs={
        "dt": 1,
        "horizon": 200,
    },
)

register(
    id="densemountaincar-dt10-v0",
    entry_point=DenseMountainCarEnv,
    kwargs={
        "dt": 10,
        "horizon": 25,
    },
)

register(
    id="weirdgain-v0",
    entry_point=WeirdGainEnv,
)

register(
    id="uniformweirdgain-v0",
    entry_point=WeirdGainEnv,
    kwargs={
        "uniform_start": True,
    },
)

register(
    id="navigation-v0",
    entry_point=NavigationEnv,
    )

register(
    id="uniform-navigation-v0",
    entry_point=NavigationEnv,
    kwargs={
        "uniform_start": True,
    },
)

register(
    id="navigation-easy-v0",
    entry_point=NavigationEnv,
    kwargs={
        "easy": True,
    },

)

register(
    id="uniform-navigation-easy-v0",
    entry_point=NavigationEnv,
    kwargs={
        "uniform_start": True,
        "easy": True,
    },
)

register(
    id="shifted-navigation-easy-v0",
    entry_point=NavigationEnv,
    kwargs={
        "shifted": True,
        "easy": True,
    },
)

"""
try:
    from ucb.envs.reacher_v4 import ReacherEnv

    register(
        id="ourreacher-v0",
        entry_point=ReacherEnv,
    )
except:
    log.info("mujoco failed")
"""
try:
    from ucb.envs.beta_rotation import (
        BetaRotationTrackingGymEnv,
    )

    register(
        id="newbetarotation-v0",
        entry_point=BetaRotationTrackingGymEnv,
    )
    register(
        id="uniformbetarotation-v0",
        entry_point=BetaRotationTrackingGymEnv,
        kwargs={"uniform_start": True},
    )
    register(
        id="shiftedbetarotation-v0",
        entry_point=BetaRotationTrackingGymEnv,
        kwargs={"shifted": True},
    )
except:
    log.warning("new fusion dependencies not found, skipping")
try:
    from ucb.envs.beta_tracking_env import BetaTrackingGymEnv

    register(
        id="betatracking-v0",
        entry_point=BetaTrackingGymEnv,
    )
    register(
        id="uniformbetatracking-v0",
        entry_point=BetaTrackingGymEnv,
        kwargs={"uniform_start": True},
    )
    register(
        id="shiftedbetatracking-v0",
        entry_point=BetaTrackingGymEnv,
        kwargs={"shifted": True},
    )
except:
    log.warning("old fusion dependencies not found, skipping")
try:
    import gym_anm
except:
    log.warning("grid dependencies not found")
