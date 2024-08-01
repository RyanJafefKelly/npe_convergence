import jax.numpy as jnp
import jax.random as random

import os
from npe_convergence.examples.stereological import stereological, get_prior_samples, get_summaries, transform_to_unbounded, transform_to_bounded
from flowjax.bijections import RationalQuadraticSpline  # type: ignore
from flowjax.distributions import Normal, StandardNormal, Uniform  # type: ignore
from flowjax.flows import CouplingFlow  # type: ignore
from flowjax.train.data_fit import fit_to_data  # type: ignore
import flowjax.bijections as bij

import matplotlib.pyplot as plt
import scipy.io as sio
from jax.scipy.special import logit, expit


def run_stereological_npe():
    key = random.PRNGKey(0)
    prior_samples = get_prior_samples(key, 1)
    true_params = jnp.array([100, 2, -0.1])
    print(os.getcwd())
    # x_mat = sio.loadmat("npe_convergence/data/data_stereo_real.mat")
    # y_obs = jnp.array(x_mat["y"])
    # y_obs = get_summaries(y_obs)
    key, subkey = random.split(key)
    y_obs = stereological(subkey, *true_params, num_samples=1)
    y_obs = get_summaries(y_obs)
    y_obs_original = y_obs.copy()

    num_sims = 100_000

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

    # TODO: NPE INFERENCE
    # TODO: ABC TO VERIFY
    key, sub_key = random.split(key)
    theta_dims = 3
    summary_dims = 4
    flow = CouplingFlow(
        key=sub_key,
        base_dist=Normal(jnp.zeros(theta_dims)),
        # base_dist=Uniform(minval=-3 * jnp.ones(theta_dims), maxval=3 * jnp.ones(theta_dims)),
        transformer=RationalQuadraticSpline(knots=10, interval=5),  # 8 spline segments over [-3, 3].
        cond_dim=summary_dims,
        # flow_layers=8,  # NOTE: changed from 5, default is 8
        # nn_width=50,  # TODO: could experiment with
        # nn_depth=3  # TODO: could experiment with
        )

    key, sub_key = random.split(key)

    flow, losses = fit_to_data(
        key=sub_key,
        dist=flow,
        x=thetas,
        condition=sim_summ_data,
        learning_rate=5e-4,
        max_epochs=1000,
        max_patience=20,
        batch_size=256
    )

    plt.plot(losses['train'], label='train')
    plt.plot(losses['val'], label='val')
    plt.savefig('losses.pdf')
    plt.clf()

    # standardise y_obs
    y_obs = (y_obs - sim_summ_data_mean) / sim_summ_data_std

    num_posterior_samples = 5_000
    posterior_samples = flow.sample(sub_key, sample_shape=(num_posterior_samples,), condition=y_obs)
    posterior_samples = (posterior_samples * thetas_std) + thetas_mean
    posterior_samples = jnp.squeeze(posterior_samples)
    posterior_samples = transform_to_bounded(posterior_samples)
    plt.hist(posterior_samples[:, 0], bins=50)
    # plt.xlim(0, 1)
    plt.axvline(true_params[0], color='red')
    plt.savefig('t1_posterior_stereo_npe_sim.pdf')
    plt.clf()

    plt.hist(posterior_samples[:, 1], bins=50)
    plt.axvline(true_params[1], color='red')
    # plt.xlim(0, 1)
    plt.savefig('t2_posterior_stereo_npe_sim.pdf')
    plt.clf()

    plt.hist(posterior_samples[:, 2], bins=50)
    plt.axvline(true_params[2], color='red')
    # plt.xlim(0, 1)
    plt.savefig('t3_posterior_stereo_npe_sim.pdf')
    plt.clf()

    y_obs_original = jnp.squeeze(y_obs_original)

    ppc_samples = stereological(sub_key, *posterior_samples.T, num_samples=num_posterior_samples)
    ppc_summaries = get_summaries(ppc_samples)
    ppc_summaries = jnp.squeeze(ppc_summaries)
    plt.hist(ppc_summaries[:, 0], bins=50)
    plt.axvline(y_obs_original[0], color='red')
    plt.savefig('num_inclusions_posterior_stereo_npe_sim.pdf')
    plt.clf()

    plt.hist(ppc_summaries[:, 1], bins=50)
    plt.axvline(y_obs_original[1], color='red')
    plt.savefig('min_inclusions_posterior_stereo_npe_sim.pdf')
    plt.clf()

    plt.hist(ppc_summaries[:, 2], bins=50)
    plt.axvline(y_obs_original[2], color='red')
    plt.savefig('mean_inclusions_posterior_stereo_npe_sim.pdf')
    plt.clf()

    plt.hist(ppc_summaries[:, 3], bins=50)
    plt.axvline(y_obs_original[3], color='red')
    plt.savefig('max_inclusions_posterior_stereo_npe_sim.pdf')
    plt.clf()

    pass


if __name__ == "__main__":
    run_stereological_npe()