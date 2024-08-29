"""Run gnk model using SMC ABC."""

import argparse
import jax.numpy as jnp
import jax.random as random
import numpy as np

import os
from npe_convergence.examples.gnk import gnk
# from npe_convergence.metrics import kullback_leibler, unbiased_mmd, median_heuristic


import matplotlib.pyplot as plt
import pickle as pkl

from functools import partial

import elfi  # type: ignore
from elfi.examples.gnk import GNK as elfi_GNK  # type: ignore


def elfi_GNK_mask(A, B, g, k, c=0.8, n_obs=50, batch_size=1, random_state=None):
    res = elfi_GNK(A, B, g, k, c=c, n_obs=n_obs, batch_size=batch_size, random_state=random_state)
    return jnp.squeeze(res).reshape((batch_size, -1))  # TODO: HACKY


def elfi_ss_octile(y):
    octiles = np.linspace(12.5, 87.5, 7)
    return np.percentile(y, octiles, axis=-1).T


def get_model(n_obs=50, true_params=None, seed=None):
    """Initialise the g-and-k model.

    Parameters
    ----------
    n_obs : int, optional
        Number of the observations.
    true_params : array_like, optional
        Parameters defining the model.
    seed : np.random.RandomState, optional

    Returns
    -------
    elfi.ElfiModel

    """
    m = elfi.new_model()

    # Initialising the parameters as in Allingham et al. (2009).
    if true_params is None:
        true_params = [3.0, 1.0, 2.0, .5]

    # Initialising the prior settings as in Allingham et al. (2009).
    priors = []
    priors.append(elfi.Prior('uniform', 0, 10, model=m, name='A'))
    priors.append(elfi.Prior('uniform', 0, 10, model=m, name='B'))
    priors.append(elfi.Prior('uniform', 0, 10, model=m, name='g'))
    priors.append(elfi.Prior('uniform', 0, 10, model=m, name='k'))

    y_obs = elfi_GNK_mask(*true_params, n_obs=n_obs,
                          random_state=np.random.RandomState(seed))

    fn_simulator = partial(elfi_GNK_mask, n_obs=n_obs)
    elfi.Simulator(fn_simulator, *priors, observed=y_obs, name='GNK')

    octile_ss = elfi.Summary(elfi_ss_octile, m['GNK'], name='ss_octile')

    elfi.Distance("euclidean", octile_ss, name='d')
    return m


def run_gnk_smc_abc(args):
    seed, n_obs = args.seed, args.n_obs
    dirname = "res/gnk_smc_abc/npe_n_obs_" + str(n_obs) + "_seed_" + str(seed) +  "/"
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    a, b, g, k = 3.0, 1.0, 2.0, 0.5
    true_params = jnp.array([a, b, g, k])
    key = random.PRNGKey(0)

    # follow normal approach so get same observed data
    z = random.normal(key, shape=(n_obs,))
    x_obs = gnk(z, *true_params)

    m = get_model(n_obs=n_obs, true_params=np.array([a, b, g, k]))
    m.observed['GNK'] = x_obs.reshape((1, -1))

    np.random.seed(seed)

    max_iter = 20
    num_posterior_samples = 10_000
    adaptive_smc = elfi.AdaptiveThresholdSMC(m['d'],
                                             batch_size=1_000,
                                             seed=seed,
                                             q_threshold=0.99)
    adaptive_smc_samples = adaptive_smc.sample(num_posterior_samples,
                                               max_iter=max_iter)

    print(adaptive_smc_samples)

    for i, pop in enumerate(adaptive_smc_samples.populations):
        s = pop.samples
        for k, v in s.items():
            plt.hist(v, bins=30)
            plt.title(k)
            plt.savefig(dirname + k + "_pop_" + str(i) + ".pdf")
            plt.clf()

    with open(dirname + "adaptive_smc_samples.pkl", "wb") as f:
        pkl.dump(adaptive_smc_samples.samples_array, f)

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run_gnk_smc_abc.py",
        description="Run gnk model with SMC ABC.",
        epilog="Example usage: python run_gnk_smc_abc.py"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_obs", type=int, default=1_000)
    args = parser.parse_args()
    run_gnk_smc_abc(args)
