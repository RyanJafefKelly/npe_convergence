import jax.numpy as jnp
import jax.random as random

import os
from npe_convergence.examples.ma2 import MA2, autocov, CustomPrior_t1, CustomPrior_t2, numpyro_model, sample_autocov_variance
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

import matplotlib.pyplot as plt
import pickle as pkl

def run_ma2_identifiable(n_obs: int = 1000, n_sims: int = 10_000):
    dirname = "res/ma2_npe_n_obs_" + str(n_obs) + "_n_sims_" + str(n_sims) + "/"
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    key = random.PRNGKey(0)
    true_params = jnp.array([0.6, 0.2])
    y_obs = MA2(*true_params, n_obs=n_obs, key=key)
    y_obs = jnp.array([[jnp.var(y_obs)], autocov(y_obs, lag=1), autocov(y_obs, lag=2)]).ravel()

    y_obs_original = y_obs.copy()

    # prior predictive samples
    num_prior_pred_samples = 10_000
    prior_pred_sim_data = MA2(jnp.repeat(true_params[0], num_prior_pred_samples), jnp.repeat(true_params[1], num_prior_pred_samples), batch_size=num_prior_pred_samples, n_obs=n_obs, key=key)
    prior_pred_summ_data = jnp.array((jnp.var(prior_pred_sim_data, axis=1), autocov(prior_pred_sim_data, lag=1), autocov(prior_pred_sim_data, lag=2)))
    print("stdev: ", jnp.std(prior_pred_summ_data, axis=1))
    # test_t1 = CustomPrior_t1.rvs(2., size=(1,), random_state=10)
    # test_t2 = CustomPrior_t2.rvs(test_t1, 1., size=(1,), random_state=10)
    sample_summ_var = sample_autocov_variance(true_params, k=1, n_obs=n_obs, ma_order=2)
    print("sample_summ_std k = 1: ", jnp.sqrt(sample_summ_var))


    key, sub_key = random.split(key)

    num_posterior_samples = 100_000

    nuts_kernel = NUTS(numpyro_model)
    thinning = 1
    mcmc = MCMC(nuts_kernel,
                num_warmup=1_000,
                num_samples=num_posterior_samples * thinning,
                thinning=thinning)
    mcmc.run(random.PRNGKey(1), y_obs_original, n_obs=n_obs)
    mcmc.print_summary()
    samples = mcmc.get_samples()
    true_posterior_samples = jnp.column_stack([samples['t1'], samples['t2']])


    key, sub_key = random.split(key)
    t1_bounded = CustomPrior_t1.rvs(2., size=(n_sims,), key=sub_key)
    t1 = logit((t1_bounded + 2) / 4)

    key, sub_key = random.split(key)
    t2_bounded = CustomPrior_t2.rvs(t1_bounded, 1., size=(n_sims,), key=sub_key)
    t2 = logit((t2_bounded + 1) / 2)

    key, sub_key = random.split(key)
    sim_data = MA2(t1_bounded, t2_bounded, n_obs=n_obs, batch_size=n_sims, key=sub_key)
    # sim_summ_data = sim_data
    sim_summ_data = jnp.array((jnp.var(sim_data, axis=1), autocov(sim_data, lag=1), autocov(sim_data, lag=2)))

    thetas = jnp.column_stack([t1, t2])
    # thetas = jnp.vstack([thetas, true_params])  # TODO: FOR TESTING
    thetas_mean = thetas.mean(axis=0)
    thetas_std = thetas.std(axis=0)
    thetas = (thetas - thetas_mean) / thetas_std

    sim_summ_data = sim_summ_data.T
    # sim_summ_data = jnp.vstack([sim_summ_data, y_obs])  # TODO: FOR TESTING
    sim_summ_data_mean = sim_summ_data.mean(axis=0)
    sim_summ_data_std = sim_summ_data.std(axis=0)
    sim_summ_data = (sim_summ_data - sim_summ_data_mean) / sim_summ_data_std

    y_obs = (y_obs - sim_summ_data_mean) / sim_summ_data_std

    key, sub_key = random.split(sub_key)
    theta_dims = 2
    summary_dims = 3
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

    posterior_samples = flow.sample(sub_key, sample_shape=(num_posterior_samples,), condition=y_obs)
    posterior_samples = (posterior_samples * thetas_std) + thetas_mean
    posterior_samples = posterior_samples.at[:, 0].set(4 * expit(posterior_samples[:, 0]) - 2)
    posterior_samples = posterior_samples.at[:, 1].set(2 * expit(posterior_samples[:, 1]) - 1)
    _, bins, _ = plt.hist(posterior_samples[:, 0], bins=50)
    # plt.xlim(0, 1)
    plt.hist(true_posterior_samples[:, 0], bins=bins)
    plt.axvline(true_params[0], color='red')
    plt.savefig(f'{dirname}t1_posterior_identifiable.pdf')
    plt.clf()

    _, bins, _ = plt.hist(posterior_samples[:, 1], bins=50)
    plt.hist(true_posterior_samples[:, 1], bins=bins)
    plt.axvline(true_params[1], color='red')
    # plt.xlim(0, 1)
    plt.savefig(f'{dirname}t2_posterior_identifiable.pdf')
    plt.clf()

    resolution = 200
    xgrid, ygrid = jnp.meshgrid(jnp.linspace(-3, 3, resolution), jnp.linspace(-3, 3, resolution))
    xy_input = jnp.column_stack([xgrid.ravel(), ygrid.ravel()])
    zgrid = jnp.exp(flow.log_prob(xy_input, condition=y_obs)).reshape(resolution, resolution)
    # restandardise = lambda x: x * thetas_std + thetas_mean
    xgrid = xgrid.ravel() * thetas_std[0] + thetas_mean[0]
    xgrid = xgrid.reshape(resolution, resolution)
    ygrid = ygrid.ravel() * thetas_std[1] + thetas_mean[1]
    ygrid = ygrid.reshape(resolution, resolution)
    plt.axvline(true_params[0], color='red')
    plt.axhline(true_params[1], color='red')
    plt.contourf(xgrid, ygrid, zgrid, levels=50)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig(f'{dirname}contour_identifiable.pdf')
    plt.clf()

    ppc_samples = MA2(posterior_samples[:, 0], posterior_samples[:, 1], n_obs=n_obs, batch_size=num_posterior_samples, key=sub_key)
    ppc_summaries = jnp.array((autocov(ppc_samples, lag=1), autocov(ppc_samples, lag=2)))
    plt.hist(ppc_summaries[0], bins=50)
    plt.axvline(y_obs_original[0], color='red')
    plt.savefig(f'{dirname}ppc_var_identifiable.pdf')
    plt.clf()

    plt.hist(ppc_summaries[-2], bins=50)
    plt.axvline(y_obs_original[-2], color='red')
    plt.savefig(f'{dirname}ppc_ac1_identifiable.pdf')
    plt.clf()

    plt.hist(ppc_summaries[-1], bins=50)
    plt.axvline(y_obs_original[-1], color='red')
    plt.savefig(f'{dirname}ppc_ac2_identifiable.pdf')
    plt.clf()

    kl = kullback_leibler(true_posterior_samples, posterior_samples)

    with open(f'{dirname}posterior_samples.pkl', 'wb') as f:
        pkl.dump(posterior_samples, f)

    with open(f'{dirname}true_posterior_samples.pkl', 'wb') as f:
        pkl.dump(true_posterior_samples, f)

    with open(f'{dirname}kl.txt', 'w') as f:
        f.write(str(kl))

    return kl


if __name__ == '__main__':
    run_ma2_identifiable()
