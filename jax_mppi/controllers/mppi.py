"""
This module implements a Model Predictive Path Integral (MPPI) controller.
Implementation based on the following papers:

Grady Williams, Nolan Wagener, Brian Goldfain, Paul Drews, James M.
Rehg, Byron Boots, and Evangelos A. Theodorou. Information theoretic MPC
for model-based reinforcement learning. In 2017 IEEE International Con-
ference on Robotics and Automation (ICRA), pages 1714–1721, 2017. doi:
10.1109/ICRA.2017.7989202.

Anusha Nagabandi, Kurt Konolige, Sergey Levine, and Vikash Kumar. Deep
dynamics models for learning dexterous manipulation. In Leslie Pack Kael-
bling, Danica Kragic, and Komei Sugiura, editors, Proceedings of the Con-
ference on Robot Learning, volume 100 of Proceedings of Machine Learning
Research, pages 1101–1112. PMLR, 30 Oct–01 Nov 2020.
"""

import numpy as np
from jax_mppi.controllers import Controller


class MPPI(Controller):
    """ Model Predictive Path Integral (MPPI) controller.

    Parameters
    ----------
    env : gym.Env
        The model environment instance. Should match the step function
        defined in the build_rollout_fn.

    config : dict
        Controller parameters.

    Notes
    -----

    References:
    Williams et al. 2017, https://ieeexplore.ieee.org/document/7989202
    Nagabandi et al. 2019, https://github.com/google-research/pddm

    Implementation follows the MPPI implementation by Shunichi09, see
    https://github.com/Shunichi09/PythonLinearNonlinearControl

    Assume terminal cost phi(x) = 0.
    """
    def __init__(self, env, config):
        super().__init__(env, config)

        self.plan = None
        self.reset()

    def reset(self):
        """Reset the control trajectory at the start of an episode.

        Assumes that the action space is symmetric around zero).
        """
        self.plan = np.zeros((self.cfg["horizon"], self.act_dim))

    def _sample_noise(self):
        """Get noise for constructing perturbed action sequences.

        Returns
        -------
        float array-like, shape=(n_samples, horizon, act_dim)
            Random (Gaussian) noise vector.
        """

        size = (self.cfg["n_samples"], self.cfg["horizon"], self.act_dim)
        return self.rng.normal(size=size) * self.cfg["noise_sigma"]

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

        acts = self.plan + self._sample_noise() # (n_samples, horizon, act_dim)
        acts = np.clip(acts, self.act_min, self.act_max)

        _, costs = self.rollout_fn(obs, acts) # (num_samples, horizon, 1)
        costs = costs.sum(axis=1).squeeze()  # (num_samples,)
        exp_costs = np.exp(self.cfg["temperature"] * (np.min(costs) - costs))
        denom = np.sum(exp_costs) + 1e-10

        weighted_inputs = exp_costs[:, np.newaxis, np.newaxis] * acts
        sol = np.sum(weighted_inputs, axis=0) / denom # (horizon, act_dim)

        # Update the initial plan, and only return the first action as an
        # immediate control input.
        self.plan = np.roll(sol, shift=-1, axis=0)
        self.plan[-1] = sol[-1] # Repeat the last step.

        return sol[0]
