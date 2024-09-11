import os

import flowjax.bijections as bij
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import scipy.io as sio
from flowjax.bijections import RationalQuadraticSpline  # type: ignore
from flowjax.distributions import (Normal, StandardNormal,  # type: ignore
                                   Uniform)
from flowjax.flows import CouplingFlow  # type: ignore
from flowjax.train.data_fit import fit_to_data  # type: ignore
from jax.scipy.special import expit, logit

from npe_convergence.examples.stereological import (get_prior_samples,
                                                    get_summaries,
                                                    stereological,
                                                    transform_to_bounded,
                                                    transform_to_unbounded)


def run_stereological_npe():
    key = random.PRNGKey(0)
    prior_samples = get_prior_samples(key, 1)
    true_params = jnp.array([100, 2, -0.1])
    # x_mat = sio.loadmat("npe_convergence/data/data_stereo_real.mat")
    # y_obs = jnp.array(x_mat["y"])
    # y_obs = get_summaries(y_obs)
    key, subkey = random.split(key)
    y_obs = stereological(subkey, *true_params, num_samples=1)
    y_obs = get_summaries(y_obs)
    # plt.hist(x_obs.ravel())
    # plt.savefig("x_obs.pdf")
    y_obs_original = y_obs.copy()
    num_sims = 10_000

    key, subkey = random.split(key)
    thetas = get_prior_samples(key, num_sims)

    key, subkey = random.split(key)
    sim_data = stereological(subkey, *thetas.T, num_samples=num_sims)
    sim_summ_data = get_summaries(sim_data)

    # TODO LOGIT/EXPIT

    thetas = transform_to_unbounded(thetas)

    thetas_mean = thetas.mean(axis=0)
    thetas_std = thetas.std(axis=0)
    thetas = (thetas - thetas_mean) / thetas_std

    sim_summ_data_mean = sim_summ_data.mean(axis=0)
    sim_summ_data_std = sim_summ_data.std(axis=0)

    sim_summ_data = (sim_summ_data - sim_summ_data_mean) / sim_summ_data_std
 
    def distance_fn(x, y):
        return jnp.sum((x - y) ** 2, axis=-1)

    y_obs = (y_obs - sim_summ_data_mean) / sim_summ_data_std
    distances = distance_fn(sim_summ_data, y_obs)
    percent_cutoff = 5
    accepted_idx = distances < jnp.percentile(distances, percent_cutoff)
    posterior_samples = thetas[accepted_idx, :]

    posterior_samples = posterior_samples * thetas_std + thetas_mean
    posterior_samples = transform_to_bounded(posterior_samples)
    # standardise y_obs

    plt.hist(posterior_samples[:, 0], bins=50)
    # plt.xlim(0, 1)
    plt.axvline(true_params[0], color='red')
    plt.savefig('t1_posterior_stereo_abc_sim.pdf')
    plt.clf()

    plt.hist(posterior_samples[:, 1], bins=50)
    plt.axvline(true_params[1], color='red')
    # plt.xlim(0, 1)
    plt.savefig('t2_posterior_stereo_abc_sim.pdf')
    plt.clf()

    plt.hist(posterior_samples[:, 2], bins=50)
    plt.axvline(true_params[2], color='red')
    # plt.xlim(0, 1)
    plt.savefig('t3_posterior_stereo_abc_sim.pdf')
    plt.clf()

    y_obs_original = jnp.squeeze(y_obs_original)

    key, sub_key = random.split(key)
    ppc_samples = stereological(sub_key, *posterior_samples.T, num_samples=posterior_samples.shape[0])
    ppc_summaries = get_summaries(ppc_samples)
    ppc_summaries = jnp.squeeze(ppc_summaries)
    plt.hist(ppc_summaries[:, 0], bins=50)
    plt.axvline(y_obs_original[0], color='red')
    plt.savefig('num_inclusions_posterior_stereo_abc_sim.pdf')
    plt.clf()

    plt.hist(ppc_summaries[:, 1], bins=50)
    plt.axvline(y_obs_original[1], color='red')
    plt.savefig('min_inclusions_posterior_stereo_abc_sim.pdf')
    plt.clf()

    plt.hist(ppc_summaries[:, 2], bins=50)
    plt.axvline(y_obs_original[2], color='red')
    plt.savefig('mean_inclusions_posterior_stereo_abc_sim.pdf')
    plt.clf()

    plt.hist(ppc_summaries[:, 3], bins=50)
    plt.axvline(y_obs_original[3], color='red')
    plt.savefig('max_inclusions_posterior_stereo_abc_sim.pdf')
    plt.clf()

    pass


if __name__ == "__main__":
    run_stereological_npe()