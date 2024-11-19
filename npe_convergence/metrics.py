"""Sample-based metrics between two statistical distributions."""

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from scipy.spatial import KDTree  # type: ignore


def rbf_kernel(x, y, lengthscale=1.0):
    """Compute the RBF kernel between two sets of points."""
    return jnp.exp(-jnp.sum((x - y) ** 2) / (2 * lengthscale ** 2))


def median_heuristic(x, batch_size=1000):
    """Compute median heuristic using a batched approach to save memory."""
    n = x.shape[0]

    def batch_distances(i):
        start = i * batch_size
        end = min((i + 1) * batch_size, n)
        batch = x[start:end]
        dists = jnp.sqrt(jnp.sum((batch[:, None, :] - x[None, :, :]) ** 2,
                                 axis=-1))
        return dists.ravel()

    num_batches = (n + batch_size - 1) // batch_size
    all_dists = jnp.concatenate([batch_distances(i)
                                 for i in range(num_batches)])

    return jnp.sqrt(jnp.median(all_dists) / 2)  # NOTE: memory-intensive step


def unbiased_mmd(npe_posterior_samples: Array,
                 exact_posterior_samples: Array,
                 lengthscale: int = 1
                 ) -> Array:
    m = npe_posterior_samples.shape[0]
    n = exact_posterior_samples.shape[0]

    xx = jnp.sum(npe_posterior_samples**2, axis=1)[:, None]
    yy = jnp.sum(exact_posterior_samples**2, axis=1)[None, :]
    xy = jnp.dot(npe_posterior_samples, exact_posterior_samples.T)

    k_simulated = jnp.exp(-(xx + xx.T - 2 * jnp.dot(npe_posterior_samples,
                                                    npe_posterior_samples.T)) /
                          (2 * lengthscale**2))
    k_obs = jnp.exp(-(yy + yy.T - 2 * jnp.dot(exact_posterior_samples,
                                              exact_posterior_samples.T)) /
                    (2 * lengthscale**2))
    k_sim_obs = jnp.exp(-(xx + yy - 2 * xy) / (2 * lengthscale**2))

    k_simulated = k_simulated.at[jnp.diag_indices(m)].set(0)
    k_obs = k_obs.at[jnp.diag_indices(n)].set(0)

    k_sim_term = jnp.sum(k_simulated) / (m * (m-1))
    k_obs_term = jnp.sum(k_obs) / (n * (n-1))
    k_sim_obs_term = -2 * jnp.sum(k_sim_obs) / (m*n)

    mmd_value = k_sim_term + k_obs_term + k_sim_obs_term

    return mmd_value


def kullback_leibler(true_samples, sim_samples):
    """Compute the Kullback-Leibler divergence between two sets of samples.

    Args:
        true_samples (jax.Array): samples from the true distribution
        sim_samples (jax.Array): samples from the simulated distribution

    Returns:
        kl_estimate (float): estimate of the KL divergence

    See Pérez-Cruz (2008) "Kullback-Leibler divergence estimation of continuous
    distributions" for more details.
    """
    true_samples = np.array(true_samples)
    sim_samples = np.array(sim_samples)

    n, d = true_samples.shape
    m, _ = sim_samples.shape

    true_tree = KDTree(true_samples)
    sim_tree = KDTree(sim_samples)

    r = true_tree.query(true_samples,
                        k=2,  # num neighbours
                        eps=.01,  # (1+eps)-approximation upper bound
                        p=2  # p-norm
                        )[0][:, 1]  # skip first column as includes itself
    s = sim_tree.query(true_samples, k=1, eps=.01, p=2)[0]

    kl_estimate = -np.log(r/s).sum() * d / n + np.log(m / (n - 1.0))

    return kl_estimate
