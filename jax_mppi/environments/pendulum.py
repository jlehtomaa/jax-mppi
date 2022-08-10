"""
A step function corresponding to the OpenAI gym Pendulum-v0 environment.

Following documentation is from:
https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py

The environment details are (from OpenAI gym):

Action Space:
The action is a `ndarray` with shape `(1,)` representing the torque
applied to free end of the pendulum.

| Num | Action | Min  | Max |
|-----|--------|------|-----|
| 0   | Torque | -2.0 | 2.0 |

Observation Space:
The observation is a `ndarray` with shape `(3,)` representing
the x-y coordinates of the pendulum's free end and its angular velocity.

| Num | Observation      | Min  | Max |
|-----|------------------|------|-----|
| 0   | x = cos(agnle)   | -1.0 | 1.0 |
| 1   | y = sin(angle)   | -1.0 | 1.0 |
| 2   | Angular Velocity | -8.0 | 8.0 |

Starting State:
The starting state is a random angle in *[-pi, pi]* and a random
angular velocity in *[-1,1]*.
"""

import jax.numpy as jnp

MAX_TORQUE = 2.0
MIN_TORQUE = -2.0
MAX_ANGULAR_VELO = 8.0
MIN_ANGULAR_VELO = -8.0

DT = 0.05 # Time delta
MASS = 1.0 # Point mass of the bob.
LENGTH = 1.0 # Length of the rod
GRAVITY = 10.0 # Acceleration of gravity.


def angle2coords(angle):
    """ Transforms an angle on a 2D plane to (x, y) coordinates.

    Parameters
    ----------
    angle : float
        Angle from the vertical to the pendulum.

    Returns
    -------
    float array-like
        Coordinate vector corresponding to the angle.

    """
    return jnp.array([jnp.cos(angle), jnp.sin(angle)])

def coords2angle(coords):
    """ Transforms (x, y) coordinates into the corresponding vector angle.

    Parameters
    ----------
    coords : float array-like
        Coordinates on an (x,y)-plane.

    Returns
    -------
    float
        Pendulum angle.
    """
    cos, sin = coords
    return jnp.arctan2(sin, cos)


def angle_normalize(angle) -> float:
    """ Normalize angle to the [-pi, pi] interval.

    Parameters
    ----------
    angle : float
        Pendulum angle.

    Returns
    -------
    float
        New angle in [-pi, pi].

    Notes
    -----
    From the Gym repository at :
    /gym/blob/master/gym/envs/classic_control/pendulum.py

    """
    return ((angle + jnp.pi) % (2 * jnp.pi)) - jnp.pi


def step(obs, act):
    """ Take one step forward in the pendulum environment.

    Follows the Pendulum-v0 model from OpenAI Gym.

    Parameters
    ----------
    obs : float array-like
        Most recent observation of the system state.
    act : float array-like
        Control input to apply to the system.

    Returns
    -------
    new_obs : float array-like, shape=(3,)
        The state observation after applying the control input.
    cost : float
        Immediate cost response from the system.

    Notes
    -----
    Observation corresponds to [cos(theta), sin(theta), theta_dot], that is,
    angle of the pendulum and the angular velocity.

    The control input is the torque.

    Theta is the pendulum's angle in [-pi, pi], with 0 denoting the
    upright position.
    """
    x_coord, y_coord, thdot = obs
    theta = coords2angle((x_coord, y_coord))
    act = jnp.clip(act, MIN_TORQUE, MAX_TORQUE)

    cost = angle_normalize(theta) ** 2 + 0.1 * thdot ** 2 + 0.001 * (act ** 2)

    thdot += (3 * GRAVITY / (2 * LENGTH) * jnp.sin(theta)) * DT
    thdot += (3.0 / (MASS * LENGTH ** 2) * act) * DT

    thdot = jnp.clip(thdot, MIN_ANGULAR_VELO, MAX_ANGULAR_VELO)
    newth = theta + thdot * DT

    newx, newy = angle2coords(newth)
    newobs = jnp.array([newx, newy, thdot])

    return newobs, cost
