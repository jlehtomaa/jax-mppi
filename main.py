"""
This script runs one episode using the specified control algorithm.

Example use:
python main.py --algo cemppi --horizon 30
"""
import argparse
from jax_mppi.controllers import MPPI, CEMPPI
from jax_mppi.utils import make_env


def main():
    """Run one episode with the controller."""

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", "-s", type=int, default=42,
        help="Random number generator seed for MPPI.")
    parser.add_argument("--algo", "-a", type=str, default="cemppi",
        choices=["mppi", "cemppi"], help="Control algorithm.")
    parser.add_argument("--env_name", "-e", type=str, default="Pendulum-v1",
        choices=["mppi", "cemppi"], help="Control algorithm.")
    parser.add_argument("--horizon", type=int, default=20,
        help="MPC planning horizon.")
    parser.add_argument("--temperature", type=float, default=1.0,
        help="MPPI weight temperature.")
    parser.add_argument("--n_samples", type=int, default=256,
        help="How many random trajectories to generate.")
    parser.add_argument("--noise_sigma", type=float, default=0.9,
        help="MPPI noise magnitude (Ignored for CEMPPI).")
    parser.add_argument("--cov_diag", type=list, default=[1.],
        help="Noise sampling covariance (CEMPPI only).")
    parser.add_argument("--opt_iters", type=int, default=10,
        help="Number adaptive importance sampling iterations (CEMPPI only).")
    parser.add_argument("--elite_frac", type=float, default=0.2,
        help="Sample fraction treated as elite in cross entropy sampling.")
    parser.add_argument("--alpha", type=float, default=1.0,
        help="Control cost parameter (only CEMPPI).")

    args = vars(parser.parse_args())
    env = make_env(args["env_name"], seed=0)

    if args["algo"] == "mppi":
        controller = MPPI(env, args)
    elif args["algo"] == "cemppi":
        controller = CEMPPI(env, args)
    else:
        raise ValueError("Unknown control algorithm.")

    controller.reset()
    obs, done, cum_rew = env.reset(), False, 0.

    while not done:
        act = controller.get_action(obs.reshape((-1,1)))
        obs, rew, done, _ = env.step(act)
        cum_rew += rew
    env.close()

if __name__ == "__main__":
    main()
