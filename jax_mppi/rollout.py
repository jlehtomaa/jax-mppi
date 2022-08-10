"""
This module defines the nominal model that the controller has
access to. We write the transition dynamics of the system in terms of
the current state and a single control input. We then formulate the
rollout of N steps by using the jax.lax.scan machinery, which is an
efficient way to get rid of for-loops. Finally, we vectorize the rollout
over K samples of different action trajectories to run the MPPI algorithm.


As a simple test environment, we use Pendulum-v0 from OpenAI gym:
https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py

The MPPI method, and the scan procedure, easily extend to more complex models,
including those that with time-varying dynamics.
"""

import jax
from jax_mppi.environments.pendulum import step

def lax_wrapper_step(carry, action):
    """ Wrapper of a step function.

    This step is not strictly necessary, but makes using the jax.lax.scan
    easier in situations where the dynamics are, say, varying in time,
    or have random components that require the JAX PRNGKey.

    See https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html
    for more details.

    Arguments:
    ---------
    carry : tuple The item that lax carries (and accumulates) from one step
            to the next. Here, it only contains the current state.
    input : array The current control input.
    params : dict Parameters for the nominal environment model.

    Output:
    -------
    new_carry : tuple Updated carry value for the next iteration. Carry must
                have the same shape and dtype across all iterations.

    output : The outputs we want to track at each step over the scan loop.
    """
    state = carry[0]
    next_state, reward = step(state, action)

    new_carry = (next_state, )
    output = (next_state, reward)

    return new_carry, output


def build_rollout_fn(step_fn):
    """ Build a vectorized call to the rollout function."""

    def rollout_fn(obs, act_sequence):
        """
        Arguments:
        ---------
        obs : (obs_dim) - shaped array, starting state of sequence.
        act_sequence : (n_steps, act_dim) - shaped array.

        """
        carry = (obs, )
        _, obs_and_rews = jax.lax.scan(f=step_fn, init=carry, xs=act_sequence)

        return obs_and_rews

    # Vectorize the rollout_fn over the first dim of the action sequence.
    # That is, the act_sequence for func should have the following shape:
    # (n_samples, n_steps, act_dim).
    func = jax.jit(jax.vmap(rollout_fn, in_axes=(None, 0)))

    return func
