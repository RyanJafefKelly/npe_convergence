"""Run simple Gauss example."""
import jax.numpy as jnp
import jax.random as random

from npe_convergence.examples.gauss import gauss, get_summaries
from npe_convergence.metrics import kullback_leibler, total_variation, unbiased_mmd

from flowjax.bijections import RationalQuadraticSpline  # type: ignore
from flowjax.distributions import Normal, StandardNormal, Uniform  # type: ignore
from flowjax.flows import CouplingFlow  # type: ignore
from flowjax.train.data_fit import fit_to_data  # type: ignore
import flowjax.bijections as bij
import matplotlib.pyplot as plt


def run_gauss_npe():
    # TODO
    # NOTE: prior
    # TODO: current... 3 dim, unknown mean, identity covariance
    key = random.PRNGKey(0)
    loc = jnp.zeros(10)
    loc = jnp.expand_dims(loc, axis=0)
    scale = jnp.eye(10)
    n_obs = 500
    true_samples = gauss(key, loc=loc, scale=scale, batch_size=n_obs)
    x_obs = get_summaries(true_samples)
    x_obs = x_obs.ravel()

    # TODO: standardise?

    key, subkey = random.split(key)
    prior_scale = 10 * jnp.eye(10)
    num_sims = 25_000
    thetas = random.multivariate_normal(key, loc, prior_scale, shape=(num_sims,))
    x = gauss(subkey, loc=thetas, scale=scale, batch_size=n_obs)
    sim_summ_data = get_summaries(x)

    thetas_mean = thetas.mean(axis=0)
    thetas_std = thetas.std(axis=0)
    thetas = (thetas - thetas_mean) / thetas_std

    sim_summ_data_mean = sim_summ_data.mean(axis=0)
    sim_summ_data_std = sim_summ_data.std(axis=0)
    sim_summ_data = (sim_summ_data - sim_summ_data_mean) / sim_summ_data_std

    x_obs = (x_obs - sim_summ_data_mean) / sim_summ_data_std

    # TODO: NPE stuff...
    key, sub_key = random.split(key)
    theta_dims = 10
    summary_dims = 10
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

    num_posterior_samples = 10_000
    posterior_samples = flow.sample(sub_key, sample_shape=(num_posterior_samples,), condition=x_obs)
    posterior_samples = (posterior_samples * thetas_std) + thetas_mean
    posterior_samples = jnp.squeeze(posterior_samples)
    # posterior_samples = transform_to_bounded(posterior_samples)

    # TODO: iterate plots
    for i in range(len(x_obs)):
        plt.hist(posterior_samples[:, i], bins=50)
        plt.axvline(0, color='red')
        plt.savefig(f't{i}_gauss_npe.pdf')
        plt.clf()

    # TODO: MMD
    # mmd = unbiased_mmd(posterior_samples, )

    # TODO: KL ... could clean this up, should replace inv w/ solve, fine here - but do this if consider finer precision
    true_posterior_mean = jnp.linalg.inv(jnp.linalg.inv(prior_scale) + n_obs * jnp.linalg.inv(scale)) @ (jnp.linalg.inv(prior_scale) @ loc.T + n_obs * jnp.linalg.inv(scale) @ x_obs.reshape((-1, 1)))
    true_posterior_cov = jnp.linalg.inv(jnp.linalg.inv(prior_scale) + n_obs * jnp.linalg.inv(scale))

    true_posterior_samples = random.multivariate_normal(key,
                                                        true_posterior_mean.ravel(),
                                                        true_posterior_cov,
                                                        shape=(num_posterior_samples,))
    kl = kullback_leibler(true_posterior_samples, posterior_samples)
    print(f"KL: {kl}")
    # !TODO! DECIDE ON METRIC ... COULD BE MMD
    # TODO: total variation stuff
    # TODO: need CDF of MVN
    # can use scipy.stats.multivariate_normal.cdf (or logcdf)
    # TODO: CDF of NPE samples (this might be an issue)
    # TODO: grid of points evaluate over
    # TODO: Finally, just get max over grid

    return None


if __name__ == '__main__':
    run_gauss_npe()