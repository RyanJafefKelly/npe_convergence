import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np

import os
from npe_convergence.examples.gnk import gnk, ss_robust, run_nuts, pgk, gnk_density, gnk_deriv, ss_octile
from npe_convergence.metrics import kullback_leibler, total_variation, unbiased_mmd

from flowjax.bijections import RationalQuadraticSpline  # type: ignore
import flowjax.bijections as bij
from flowjax.distributions import Normal, StandardNormal, Uniform  # type: ignore
from flowjax.flows import coupling_flow  # type: ignore
from flowjax.train.data_fit import fit_to_data  # type: ignore
import flowjax.bijections as bij
from jax.scipy.special import logit, expit

from numpyro.infer import MCMC, NUTS
import numpyro.handlers as handlers
import numpyro.distributions as dist

import matplotlib.pyplot as plt
import pickle as pkl


def run_gnk(n_obs: int = 1_000, n_sims: int = 100_000):
    dirname = "res/gnk/npe_n_obs_" + str(n_obs) + "_n_sims_" + str(n_sims) + "/"
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    # key = random.PRNGKey(1)
    a, b, g, k = 3.0, 1.0, 2.0, 0.5
    true_params = jnp.array([a, b, g, k])
    key = random.PRNGKey(0)

    z = random.normal(random.PRNGKey(0), shape=(n_obs,))
    x_obs = gnk(z, *true_params)
    x_obs = jnp.atleast_2d(x_obs)
    x_obs = ss_octile(x_obs)
    x_obs = jnp.squeeze(x_obs)
    x_obs_original = x_obs.copy()


    num_prior_pred_samples = 10_000
    x = gnk(z, *true_params)
    # TODO: POOR FOR LOOP
    x_pred = np.empty((num_prior_pred_samples, 7))
    for i in range(num_prior_pred_samples):
        key, subkey = random.split(key)
        z = random.normal(subkey, shape=(n_obs,))
        x = gnk(z, *true_params)
        x = jnp.atleast_2d(x)
        x_pred[i, :] = ss_octile(x).ravel()

    for i in range(7):
        plt.hist(x_pred[:, i], bins=50)
        plt.axvline(x_obs_original[i], color='red')
        plt.savefig(dirname + f"prior_pred_{i}.pdf")
        plt.clf()

    key, subkey = random.split(key)

    # NOTE: first get true thetas
    true_thetas = run_nuts(seed=1, obs=x_obs, n_obs=n_obs)


    posterior_params = ['A', 'B', 'g', 'k']
    for ii, param in enumerate(posterior_params):
        plt.hist(true_thetas[param], bins=50, label=param)
        plt.legend()
        plt.axvline(true_params[ii], color='red')
        plt.savefig(dirname + f"true_samples_{param}.pdf")
        plt.clf()

    # TODO: SAMPLE PRIOR
    key, subkey = random.split(key)
    thetas_bounded = dist.Uniform(0, 10).sample(subkey, (n_sims, 4))
    thetas_unbounded = logit(thetas_bounded / 10)

    # TODO: POOR FOR LOOP
    x_sims = np.empty((n_sims, 7))
    for ii, theta in enumerate(thetas_bounded):
        key, subkey = random.split(key)
        z = random.normal(subkey, shape=(n_obs,))
        x = gnk(z, *theta)
        x = jnp.atleast_2d(x)
        x_sims[ii, :] = ss_octile(x).ravel()

    x_sims = jnp.array(x_sims)

    thetas_mean = thetas_unbounded.mean(axis=0)
    thetas_std = thetas_unbounded.std(axis=0)
    thetas = (thetas_unbounded - thetas_mean) / thetas_std

    # sim_summ_data = x_sims.T
    sim_summ_data = x_sims  # TODO
    sim_summ_data_mean = sim_summ_data.mean(axis=0)
    sim_summ_data_std = sim_summ_data.std(axis=0)
    sim_summ_data = (sim_summ_data - sim_summ_data_mean) / sim_summ_data_std
    x_obs = (x_obs - sim_summ_data_mean) / sim_summ_data_std

    key, sub_key = random.split(key)
    theta_dims = 4
    summary_dims = 7
    flow = coupling_flow(
        key=sub_key,
        base_dist=Normal(jnp.zeros(theta_dims)),
        # base_dist=Uniform(minval=-3 * jnp.ones(theta_dims), maxval=3 * jnp.ones(theta_dims)),
        transformer=RationalQuadraticSpline(knots=10, interval=5),  # 8 spline segments over [-3, 3].
        cond_dim=summary_dims,
        # flow_layers=8,  # NOTE: changed from 5, default is 8
        # nn_width=50,  # TODO: could experiment with
        nn_depth=2  # TODO: could experiment with
        )
    key, sub_key = random.split(key)

    flow, losses = fit_to_data(
        key=sub_key,
        dist=flow,
        x=thetas,
        condition=sim_summ_data,
        learning_rate=5e-5,  # TODO: could experiment with
        max_epochs=2000,
        max_patience=20,
        batch_size=256,
    )

    plt.plot(losses['train'], label='train')
    plt.plot(losses['val'], label='val')
    plt.savefig(f'{dirname}losses.pdf')
    plt.clf()
    key, sub_key = random.split(key)

    num_posterior_samples = 10_000
    posterior_samples = flow.sample(sub_key, sample_shape=(num_posterior_samples,), condition=x_obs)
    posterior_samples = (posterior_samples * thetas_std) + thetas_mean
    posterior_samples = expit(posterior_samples) * 10
    # plt.xlim(0, 1)
    # true_thetas = true_thetas.T  # TODO: ugly
    true_posterior_samples = jnp.zeros((num_posterior_samples, 4)) # TODO: ugly... just make a matrix from start
    for ii, (key, values) in enumerate(true_thetas.items()):
        true_posterior_samples = true_posterior_samples.at[:, ii].set(values)
        _, bins, _ = plt.hist(posterior_samples[:, ii], bins=50, alpha=0.8, label='NPE')
        plt.hist(values, bins=bins, alpha=0.8, label='true')
        plt.legend()
        plt.axvline(true_params[ii], color='black')
        plt.savefig(f'{dirname}posterior_samples_{ii}.pdf')
        plt.clf()

    kl = kullback_leibler(true_posterior_samples, posterior_samples)

    with open(f'{dirname}posterior_samples.pkl', 'wb') as f:
        pkl.dump(posterior_samples, f)

    with open(f'{dirname}true_posterior_samples.pkl', 'wb') as f:
        pkl.dump(true_posterior_samples, f)

    return kl


if __name__ == "__main__":
    run_gnk()
