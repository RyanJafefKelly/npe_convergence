"""Metrics."""

import numpy as np
import jax.numpy as jnp
from scipy.spatial import KDTree


def rbf_kernel(x, y, lengthscale=1.0):
    return jnp.exp(-jnp.sum((x - y) ** 2) / (2 * lengthscale ** 2))


def median_heuristic(x):
    """median distance between points in the aggregate sample."""
    pairwise_dists = jnp.sqrt(jnp.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1))
    return jnp.sqrt(jnp.median(pairwise_dists) / 2)


# TODO: confirm?
#TODO! DO PROPERLY
def unbiased_mmd(npe_posterior_samples, exact_posterior_samples, lengthscale=1):
    # TODO: doing things slow ... fine for now but do better matrix stuff in need
    m = npe_posterior_samples.shape[0]
    n = exact_posterior_samples.shape[0]
    k_simulated = jnp.array([[rbf_kernel(x, y, lengthscale) for jj, x in enumerate(npe_posterior_samples) if ii != jj] for ii, y in enumerate(npe_posterior_samples)])

    # Compute kernel values between simulated and observed statistics
    k_obs = jnp.array([[rbf_kernel(x, y, lengthscale) for jj, x in enumerate(exact_posterior_samples) if ii != jj] for ii, y in enumerate(exact_posterior_samples)])

    k_sim_obs = jnp.array([[rbf_kernel(x, y, lengthscale) for jj, x in enumerate(npe_posterior_samples)] for ii, y in enumerate(exact_posterior_samples)])

    # Calculate MMD
    mmd_value = (jnp.sum(k_simulated) / (m * (m-1))) - (2 * jnp.sum(k_sim_obs) / (m*n)) + (jnp.sum(k_obs) / (n * (n-1)))

    return mmd_value


def unbiased_mmd_optimized(npe_posterior_samples, exact_posterior_samples, lengthscale=1):
    # TODO! GO THROUGH THIS - CHECK LEGIT
    m = npe_posterior_samples.shape[0]
    n = exact_posterior_samples.shape[0]

    # Compute pairwise distances
    xx = jnp.sum(npe_posterior_samples**2, axis=1)[:, None]
    yy = jnp.sum(exact_posterior_samples**2, axis=1)[None, :]
    xy = jnp.dot(npe_posterior_samples, exact_posterior_samples.T)

    # Compute kernel matrices
    k_simulated = jnp.exp(-(xx + xx.T - 2 * jnp.dot(npe_posterior_samples, npe_posterior_samples.T)) / (2 * lengthscale**2))
    k_obs = jnp.exp(-(yy + yy.T - 2 * jnp.dot(exact_posterior_samples, exact_posterior_samples.T)) / (2 * lengthscale**2))
    k_sim_obs = jnp.exp(-(xx + yy - 2 * xy) / (2 * lengthscale**2))

    # Set diagonal elements to zero
    k_simulated = k_simulated.at[jnp.diag_indices(m)].set(0)
    k_obs = k_obs.at[jnp.diag_indices(n)].set(0)

    # Calculate MMD
    mmd_value = (jnp.sum(k_simulated) / (m * (m-1))) - (2 * jnp.sum(k_sim_obs) / (m*n)) + (jnp.sum(k_obs) / (n * (n-1)))

    return mmd_value


def total_variation(P, Q):
    # TODO: return max over grid
    # TODO? Check same number of points
    pass

def kullback_leibler(true_samples, sim_samples):
    """_summary_

    Args:
        true_samples (_type_): _description_
        sim_samples (_type_): _description_

    Returns:
        kl_estimate _type_: _description_

    See Pérez-Cruz...
    """
    # TODO: convert samples to numpy for KDTree
    true_samples = np.array(true_samples)
    sim_samples = np.array(sim_samples)

    n, d = true_samples.shape
    m, _ = sim_samples.shape

    # TODO: For fun, could implement k-d tree in JAX? Although might not work nice with jit if structure varies ... but maybe good for just querying? idk

    true_tree = KDTree(true_samples)
    sim_tree = KDTree(sim_samples)

    r = true_tree.query(true_samples,
                        k=2,  # num neighbours
                        eps=.01,  # (1+eps)-approximation upper bound
                        p=2  # p-norm
                        )[0][:,1] #  skip first column as includes itself
    s = sim_tree.query(true_samples, k=1, eps=.01, p=2)[0]

    kl_estimate = -np.log(r/s).sum() * d / n + np.log(m / (n - 1.0))

    return kl_estimate