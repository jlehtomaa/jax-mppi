"""
This module implements the Cross Entropy Model Predictive Path Integral
control algorithm from the MPOPIS paper. See:

Dylan M. Asmar, Ransalu Senanayake, Shawn Manuel, and Mykel J. Kochen-
derfer. Model predictive optimized path integral strategies, 2022.
https://arxiv.org/abs/2203.16633.
"""

from copy import deepcopy
import numpy as np
from jax_mppi.utils import vec_to_block_diag
from jax_mppi.controllers import Controller


class CEMPPI(Controller):
    """Cross entropy MPPI controller.

    Parameters
    ----------
    env : gym.Env
        The model environment instance. Should match the step function
        defined in the build_rollout_fn.

    config : dict
        Controller parameters.

    Notes
    -----
    From Asmar et al. 2022, https://arxiv.org/abs/2203.16633.
    Assumes that the action space is symmetric around zero.
    """

    def __init__(self, env, config):
        super().__init__(env, config)

        self.ctrl_dim = self.act_dim * self.cfg["horizon"]
        self.n_elites = round(self.cfg["n_samples"] * self.cfg["elite_frac"])

        # Initialize the noise distribution.
        self.mean = np.zeros(self.ctrl_dim)
        self.cov = vec_to_block_diag(self.cfg["cov_diag"], self.cfg["horizon"])

        # Reshape control bounds, which allows clipping actions directly in
        # the 'ctrl_dim' space, without reshaping controls first to the
        # original 'act_dim' space.
        self._act_min = np.tile(self.act_min, self.cfg["horizon"])
        self._act_max = np.tile(self.act_max, self.cfg["horizon"])

        self.plan = None
        self.reset()

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

        plan_orig = deepcopy(self.plan) # (ctrl_dim,)
        cov = self.cov # (ctrl_dim, ctrl_dim)

        for i in range(self.cfg["opt_iters"]):

            noise = self._sample_noise(cov) # (n_samples, ctrl_dim)
            acts = np.clip(self.plan + noise, self._act_min, self._act_max)
            acts = acts.reshape(
                (self.cfg["n_samples"], self.cfg["horizon"], self.act_dim))

            _, costs = self.rollout_fn(obs, acts) # Ignore state trajectory.
            costs = costs.sum(axis=1).squeeze() # (n_samples,)
            costs /= (costs.std() + 1e-10) # Normalize

            # if self.add_ctrl_cost:
            #     cov_inv = np.linalg.inv(cov)
            #     gamma = (1. / self.cfg["temperature"]) * (1 - self.cfg["alpha"])
            #     ctrl_cost = gamma * plan_orig @ cov_inv @ (acts - plan_orig).T
            #     costs += ctrl_cost

            if i < self.cfg["opt_iters"] - 1:
                order = np.argsort(costs)
                elite = noise[order[:self.n_elites], :] # (n_elites, ctrl_dim)
                cov = np.cov(elite, rowvar=False) + np.eye(self.ctrl_dim) * 1e-8
                self.plan += np.mean(elite, axis=0) # Avg. best performing noise.

        noise += (self.plan - plan_orig) # (n_samples, ctrl_dim)
        self.plan = plan_orig

        weights = np.exp((self.cfg["temperature"]) * (np.min(costs) - costs))
        weights /= np.sum(weights) # (n_samples,)

        # Add weighted noise to current control plan.
        weighted_ctrls = self.plan + weights.T @ noise # (ctrl_dim,)
        weighted_ctrls = np.clip(weighted_ctrls, self._act_min, self._act_max)

        # Roll over one action, and repeat the last action.
        self.plan = np.roll(weighted_ctrls, -self.act_dim)
        self.plan[-self.act_dim:] = self.plan[-2*self.act_dim:-self.act_dim]
        return weighted_ctrls[:self.act_dim]

    def _sample_noise(self, cov=None):
        """Get noise for constructing perturbed action sequences.

        Parameters
        ----------
        cov : float array-like, shape=(ctrl_dim, ctrl_dim), default=None.
            Covariance matrix

        Returns
        -------
        float array-like, shape=(n_samples, ctrl_dim)
            Noise vector.
        """
        if cov is None:
            cov = self.cov

        return self.rng.multivariate_normal(self.mean, cov, self.cfg["n_samples"])

    def reset(self):
        """Reset the control trajectory at the start of an episode.

        Assumes that the action space is symmetric around zero).
        """
        self.plan = np.zeros(self.ctrl_dim) # (ctrl_dim,)
