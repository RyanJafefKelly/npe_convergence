"""Run simple Gauss example."""
import jax.numpy as jnp
import jax.random as random

from npe_convergence.examples.gauss import gauss, get_summaries


def run_gauss_gmm():
    # TODO
    # NOTE: prior
    # TODO: current... 3 dim, unknown mean, identity covariance
    key = random.PRNGKey(0)
    loc = jnp.zeros(10)
    scale = jnp.eye(10)
    n_obs = 100
    true_samples = gauss(key, loc=loc, scale=scale, batch_size=n_obs)
    x_obs = get_summaries(true_samples)

    prior_scale = 10 * jnp.eye(10)
    num_sims = 10_000
    prior_samples = random.multivariate_normal(key, loc, prior_scale, shape=(num_sims,))
    x = gauss(key, loc=prior_samples, scale=scale, batch_size=n_obs)
    ssx = get_summaries(x)

    # TODO: Gaussian MoE

    pass


if __name__ == '__main__':
    run_gauss_gmm()