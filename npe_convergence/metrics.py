"""Metrics."""

import numpy as np
import jax.numpy as jnp
from scipy.spatial import KDTree


def unbiased_mmd(P, Q):
    pass


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