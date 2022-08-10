"""Implements a base class for MPPI-like sampling-based MPC control algorithms.
"""
from abc import ABC, abstractmethod
import numpy as np
from jax_mppi.rollout import build_rollout_fn, lax_wrapper_step

class Controller(ABC):
    """MPPI controller base class.

    Parameters
    ----------
    env : gym.Env
        The model environment instance. Should match the step function
        defined in the build_rollout_fn.

    config : dict
        Controller parameters.
    """

    def __init__(self, env, config):

        self.cfg = config
        self.rng = np.random.default_rng(self.cfg["seed"])
        self.rollout_fn = build_rollout_fn(lax_wrapper_step)

        self.act_dim = env.action_space.shape[0]
        self.act_max = env.action_space.high
        self.act_min = env.action_space.low

    @abstractmethod
    def reset(self):
        """Reset the control trajectory at the start of an episode.

        Assumes that the action space is symmetric around zero).
        """

    @abstractmethod
    def get_action(self, obs):
        """Get the next optimal action based on current state observation.

        Parameters
        ----------
        obs : float array-like, shape=(obs_dim,)
            Most recent state observation from the system.

        Returns
        -------
        act : float array_like, shape=(act_dim,)
            Next control action to take.
        """

    @abstractmethod
    def _sample_noise(self):
        """Get noise for constructing perturbed action sequences."""
