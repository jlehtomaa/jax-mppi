"""Helper functions and classes."""

import numpy as np
import gym
from scipy.linalg import block_diag

def make_env(env_id, seed):
    """Initializes a training environment.

    Parameters
    ----------
    env_id : str
        Environment id.
    seed : int
        Random number generator seed

    Returns
    -------
    env : gym.Env
        An initialized environment

    Notes
    -----
    Adapted from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
    """

    env = gym.make(env_id, render_mode="human")
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    return env


def vec_to_block_diag(vec, repeats):
    """Turn a one-dimensional vector to a block diagonal matrix.

    Parameters
    ----------
    vec : float array-like
        A one-dimensional vector.

    repeats : int
        How many times to repeat vec in the resulting matrix.

    Returns
    -------
    float array-like
        A block diagonal matrix.
    """
    if isinstance(vec, np.ndarray):
        assert vec.ndim == 1, "vec must be one-dimensional."

    diag_mat = np.diag(vec)
    return block_diag(*[diag_mat for _ in range(repeats)])
