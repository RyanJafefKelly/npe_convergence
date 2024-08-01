import jax.numpy as jnp
import jax.random as random

import os
from npe_convergence.examples.ma2 import MA2, autocov, CustomPrior_t1, CustomPrior_t2
from flowjax.bijections import RationalQuadraticSpline  # type: ignore
from flowjax.distributions import Normal, StandardNormal, Uniform  # type: ignore
from flowjax.flows import CouplingFlow  # type: ignore
from flowjax.train.data_fit import fit_to_data  # type: ignore
import flowjax.bijections as bij

import matplotlib.pyplot as plt

def run_ma2_abc():
    # TODO: num sim, n_obs, effects
    # TODO! TRANSFORM YOUR VARIABLES!!
    key = random.PRNGKey(0)
    num_sims = 10_000
    true_params = jnp.array([0.6, 0.2])
    n_obs = 100
    y_obs = MA2(*true_params, n_obs=n_obs, batch_size=1, key=key)
    y_obs = jnp.array([[jnp.var(y_obs)], autocov(y_obs, lag=1), autocov(y_obs, lag=2)]).ravel()
    y_obs_original = y_obs.copy()

    key, sub_key = random.split(key)
    t1 = 2*random.uniform(sub_key, shape=(num_sims,)) - 1

    key, sub_key = random.split(key)
    t2 = random.uniform(sub_key, shape=(num_sims,))

    key, sub_key = random.split(key)
    sim_data = MA2(t1, t2, n_obs=n_obs, batch_size=num_sims, key=sub_key)

    sim_summ_data = jnp.array((jnp.var(sim_data, axis=1),
                               autocov(sim_data, lag=1),
                               autocov(sim_data, lag=2)))

    thetas = jnp.column_stack([t1, t2])
    # thetas_mean = thetas.mean(axis=0)
    # thetas_std = thetas.std(axis=0)
    # thetas = (thetas - thetas_mean) / thetas_std

    sim_summ_data = sim_summ_data.T
    # sim_summ_data_mean = sim_summ_data.mean(axis=0)
    # sim_summ_data_std = sim_summ_data.std(axis=0)
    # sim_summ_data = (sim_summ_data - sim_summ_data_mean) / sim_summ_data_std

    # y_obs = (y_obs - sim_summ_data_mean) / sim_summ_data_std

    key, sub_key = random.split(sub_key)
    theta_dims = 2
    summary_dims = 3

    # euclidean distance ABC rejection (NOTE: could do SMC / better distance)
    def distance_fn(x, y):
        return jnp.sum((x - y) ** 2, axis=-1)

    distances = distance_fn(sim_summ_data, y_obs)
    # accept best 1% of simulations
    percent_cutoff = 5
    accepted_idx = distances < jnp.percentile(distances, percent_cutoff)
    t1 = t1[accepted_idx]
    t2 = t2[accepted_idx]


    key, sub_key = random.split(key)

    plt.hist(t1, bins=50)
    plt.xlim(0, 1)
    plt.axvline(true_params[0], color='red')
    plt.savefig('t1_posterior_abc.pdf')
    plt.clf()

    plt.hist(t2, bins=50)
    plt.axvline(true_params[1], color='red')
    plt.xlim(0, 1)
    plt.savefig('t2_posterior_abc.pdf')
    plt.clf()

    num_ppc_samples = len(t1)
    ppc_samples = MA2(t1, t2, n_obs=n_obs, batch_size=num_ppc_samples, key=sub_key)
    ppc_summaries = jnp.array((autocov(ppc_samples, lag=1), autocov(ppc_samples, lag=2)))
    plt.hist(ppc_summaries[0], bins=50)
    plt.axvline(y_obs_original[0], color='red')
    plt.savefig('ppc_var_abc.pdf')
    plt.clf()

    plt.hist(ppc_summaries[-2], bins=50)
    plt.axvline(y_obs_original[-2], color='red')
    plt.savefig('ppc_ac1_abc.pdf')
    plt.clf()

    plt.hist(ppc_summaries[-1], bins=50)
    plt.axvline(y_obs_original[-1], color='red')
    plt.savefig('ppc_ac2_abc.pdf')
    plt.clf()

    return


if __name__ == '__main__':
    print(os.getcwd())
    run_ma2_abc()
