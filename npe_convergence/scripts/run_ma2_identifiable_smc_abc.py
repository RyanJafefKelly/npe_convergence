import jax.numpy as jnp
import jax.random as random

import os
import numpy as np
from npe_convergence.examples.ma2 import MA2, autocov

import matplotlib.pyplot as plt
import pickle as pkl

import elfi  # type: ignore
from elfi.examples.ma2 import get_model  # type: ignore
import argparse


def run_ma2_identifiable_smc_abc(*args, **kwargs):
    try:
        seed, n_obs, n_sims = args
    except ValueError:
        args = args[0]
        seed = args.seed
        n_obs = args.n_obs
        n_sims = args.n_sims

    dirname = "res/stereological_smc_abc/npe_n_obs_" + str(n_obs) + "_n_sims_" + str(n_sims) + "_seed_" + str(seed) +  "/"
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    key = random.PRNGKey(seed)

    true_params = jnp.array([0.6, 0.2])
    y_obs = MA2(*true_params, n_obs=n_obs, key=key)
    y_obs = jnp.array([[jnp.var(y_obs)], autocov(y_obs, lag=1), autocov(y_obs, lag=2)]).ravel()

    y_obs_original = y_obs.copy()

    # prior predictive samples
    # num_prior_pred_samples = 10_000
    # prior_pred_sim_data = MA2(jnp.repeat(true_params[0], num_prior_pred_samples), jnp.repeat(true_params[1], num_prior_pred_samples), batch_size=num_prior_pred_samples, n_obs=n_obs, key=key)
    # prior_pred_summ_data = jnp.array((jnp.var(prior_pred_sim_data, axis=1), autocov(prior_pred_sim_data, lag=1), autocov(prior_pred_sim_data, lag=2)))
    # print("stdev: ", jnp.std(prior_pred_summ_data, axis=1))
    # # test_t1 = CustomPrior_t1.rvs(2., size=(1,), random_state=10)
    # # test_t2 = CustomPrior_t2.rvs(test_t1, 1., size=(1,), random_state=10)
    # sample_summ_var = sample_autocov_variance(true_params, k=1, n_obs=n_obs, ma_order=2)
    # print("sample_summ_std k = 1: ", jnp.sqrt(sample_summ_var))
    max_iter = 5
    num_posterior_samples = 2_000

    m = get_model(n_obs=n_obs, true_params=true_params, seed_obs=seed)
    m.observed['MA2'] = y_obs_original

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
        prog="run_ma2_identifiable_smc_abc.py",
        description="Run MA(2) model with SMC ABC.",
        epilog="Example usage: python run_ma2_identifiable_smc_abc.py"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_obs", type=int, default=100)
    parser.add_argument("--n_sims", type=int, default=None)
    args = parser.parse_args()
    run_ma2_identifiable_smc_abc(args)
