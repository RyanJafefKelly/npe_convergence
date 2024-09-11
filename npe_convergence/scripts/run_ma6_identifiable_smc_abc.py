import jax.numpy as jnp
import jax.random as random

import os
import numpy as np
from npe_convergence.examples.mak import MAK, autocov, generate_valid_samples

import matplotlib.pyplot as plt
import pickle as pkl

import elfi  # type: ignore
# from elfi.examples.ma2 import get_model  # type: ignore
import argparse

from functools import partial


def elfi_MAK(*thetas, n_obs=100, batch_size=1, random_state=None):
    thetas = jnp.atleast_2d(thetas)
    thetas = thetas.T  # TODO: hacky, ELFI putting in different order
    batch_size, ma_order = thetas.shape  # assume 2d array
    random_state = random_state or np.random
    w = np.random.normal(0, 1, size=(batch_size, n_obs + ma_order))
    x = w[:, ma_order:]
    for i in range(ma_order):
        x += thetas[:, i].reshape((-1, 1)) * w[:, ma_order - i - 1: -i - 1]

    return x


def get_model(n_obs=100, true_params=None, seed_obs=None):
    ma_order = len(true_params.ravel())
    # y = elfi_MAK(*true_params, n_obs=n_obs, random_state=np.random.RandomState(seed_obs))
    y = jnp.ones(n_obs)
    sim_fn = partial(elfi_MAK, n_obs=n_obs)

    m = elfi.ElfiModel()
    priors = []
    for i in range(ma_order):
        priors.append(elfi.Prior('uniform', -1, 2, model=m, name='t' + str(i + 1)))

    elfi.Simulator(sim_fn, *priors, model=m, observed=y, name='MAK')
    summaries = []
    for i in range(ma_order):
        summaries.append(elfi.Summary(autocov, m['MAK'], i+1, name='S' + str(i + 1)))

    elfi.Distance('euclidean', *summaries, name='d')
    return m


def run_ma6_identifiable_smc_abc(*args, **kwargs):
    try:
        seed, n_obs, n_sims = args
    except ValueError:
        args = args[0]
        seed = args.seed
        n_obs = args.n_obs
        n_sims = args.n_sims

    dirname = "res/ma6_smc_abc/npe_n_obs_" + str(n_obs) + "_n_sims_" + str(n_sims) + "_seed_" + str(seed) +  "/"
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    key = random.PRNGKey(seed)
    true_params = generate_valid_samples(key, k=6, num_samples=1)
    print("true_params: ", true_params)

    key, sub_key = random.split(key)
    y_obs_full = MAK(key=sub_key, thetas=true_params, n_obs=n_obs)
    y_obs = jnp.array([[jnp.var(y_obs_full)], autocov(y_obs_full, lag=1), autocov(y_obs_full, lag=2)]).ravel()

    y_obs_original = y_obs.copy()
    print("y_obs: ", y_obs)
    # prior predictive samples
    # num_prior_pred_samples = 10_000
    # prior_pred_sim_data = MA2(jnp.repeat(true_params[0], num_prior_pred_samples), jnp.repeat(true_params[1], num_prior_pred_samples), batch_size=num_prior_pred_samples, n_obs=n_obs, key=key)
    # prior_pred_summ_data = jnp.array((jnp.var(prior_pred_sim_data, axis=1), autocov(prior_pred_sim_data, lag=1), autocov(prior_pred_sim_data, lag=2)))
    # print("stdev: ", jnp.std(prior_pred_summ_data, axis=1))
    # # test_t1 = CustomPrior_t1.rvs(2., size=(1,), random_state=10)
    # # test_t2 = CustomPrior_t2.rvs(test_t1, 1., size=(1,), random_state=10)
    # sample_summ_var = sample_autocov_variance(true_params, k=1, n_obs=n_obs, ma_order=2)
    # print("sample_summ_std k = 1: ", jnp.sqrt(sample_summ_var))
    max_iter = 40
    num_posterior_samples = 10_000

    m = get_model(n_obs=n_obs, true_params=true_params, seed_obs=seed)
    m.observed['MAK'] = y_obs_full

    np.random.seed(seed)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="run_ma6_identifiable_smc_abc.py",
        description="Run MA(6) model with SMC ABC.",
        epilog="Example usage: python run_ma6_identifiable_smc_abc.py"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_obs", type=int, default=100)
    parser.add_argument("--n_sims", type=int, default=None)
    args = parser.parse_args()
    run_ma6_identifiable_smc_abc(args)
