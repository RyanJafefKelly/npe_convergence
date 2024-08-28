"""Run gnk model using SMC ABC."""

import argparse
import jax.numpy as jnp
import jax.random as random
import numpy as np

import os
from npe_convergence.examples.gnk import gnk, run_nuts, ss_octile
from npe_convergence.metrics import kullback_leibler, unbiased_mmd, median_heuristic

from flowjax.bijections import RationalQuadraticSpline  # type: ignore
from flowjax.distributions import Normal  # type: ignore
from flowjax.flows import coupling_flow  # type: ignore
from flowjax.train.data_fit import fit_to_data  # type: ignore
from jax.scipy.special import logit, expit

import numpyro  # type: ignore
import numpyro.distributions as dist  # type: ignore

import matplotlib.pyplot as plt
import pickle as pkl

from functools import partial

import elfi
# from elfi.examples.gnk import get_model
# from elfi.examples.gnk import ss_octile as elfi_ss_octile
from elfi.examples.gnk import GNK as elfi_GNK

import numpy as np
import matplotlib.pyplot as plt


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
        true_params = [3, 1, 2, .5]

    # Initialising the prior settings as in Allingham et al. (2009).
    priors = []
    priors.append(elfi.Prior('uniform', 0, 10, model=m, name='A'))
    priors.append(elfi.Prior('uniform', 0, 10, model=m, name='B'))
    priors.append(elfi.Prior('uniform', 0, 10, model=m, name='g'))
    priors.append(elfi.Prior('uniform', 0, 10, model=m, name='k'))

    # Obtaining the observations.
    y_obs = elfi_GNK_mask(*true_params, n_obs=n_obs, random_state=np.random.RandomState(seed))

    # Defining the simulator.
    fn_simulator = partial(elfi_GNK_mask, n_obs=n_obs)
    elfi.Simulator(fn_simulator, *priors, observed=y_obs, name='GNK')

    # Initialising the summary statistics as in Allingham et al. (2009).
    octile_ss = elfi.Summary(elfi_ss_octile, m['GNK'], name='ss_order')
    # Using the multi-dimensional Euclidean distance function as
    # the summary statistics' implementations are designed for multi-dimensional cases.
    elfi.Distance("euclidean", octile_ss, name='d')
    return m


def run_gnk_smc_abc(args):
    seed, n_obs, n_sims = args.seed, args.n_obs, args.n_sims
    dirname = "res/gnk_smc_abc/npe_n_obs_" + str(n_obs) + "_n_sims_" + str(n_sims) + "_seed_" + str(seed) +  "/"
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
    
    max_iter = 6  # TODO! INCREASE
    adaptive_smc = elfi.AdaptiveThresholdSMC(m['d'], batch_size=1_000, seed=seed, q_threshold=0.99)
    adaptive_smc_samples = adaptive_smc.sample(2_000, max_iter=max_iter)
    
    print(adaptive_smc_samples)
    
    for i, pop in enumerate(adaptive_smc_samples.populations):
        s = pop.samples
        for k, v in s.items():
            # print(k, v.shape)
            plt.hist(v, bins=30)
            plt.title(k)
            plt.show()
            plt.clf()
    
    return None

if __name__ == "__main__":
    # numpyro.set_host_device_count(4)
    parser = argparse.ArgumentParser(
        prog="run_gnk_smc_abc.py",
        description="Run gnk model with SMC ABC.",
        epilog="Example usage: python run_gnk_smc_abc.py"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_obs", type=int, default=1_000)
    parser.add_argument("--n_sims", type=int, default=10_000)
    args = parser.parse_args()
    run_gnk_smc_abc(args)
