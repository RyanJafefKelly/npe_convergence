import os
import pickle as pkl
import argparse

import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
from flowjax.bijections import RationalQuadraticSpline  # type: ignore
from flowjax.distributions import Normal  # type: ignore
from flowjax.flows import coupling_flow  # type: ignore
from flowjax.train.data_fit import fit_to_data  # type: ignore
from jax.scipy.special import expit, logit
from numpyro.infer import MCMC, NUTS  # type: ignore

from npe_convergence.examples.ma2 import (MA2, CustomPrior_t1, CustomPrior_t2,
                                          autocov, numpyro_model,
                                          get_summaries_batches)
from npe_convergence.metrics import (kullback_leibler, median_heuristic,
                                     unbiased_mmd)


def run_ma2_identifiable(*args, **kwargs):
    try:
        seed, n_obs, n_sims = args
    except ValueError:
        args = args[0]
        seed = args.seed
        n_obs = args.n_obs
        n_sims = args.n_sims
    dirname = "res/ma2/npe_n_obs_" + str(n_obs) + "_n_sims_" + str(n_sims) + "_seed_" + str(seed) + "/"
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    true_params = jnp.array([0.6, 0.2])
    key = random.PRNGKey(seed)
    x_obs = MA2(*true_params, n_obs=n_obs, key=key)
    x_obs = jnp.array([[jnp.var(x_obs)], autocov(x_obs, lag=1), autocov(x_obs, lag=2)]).ravel()

    x_obs_original = x_obs.copy()

    key, sub_key = random.split(key)

    num_posterior_samples = 4_000

    nuts_kernel = NUTS(numpyro_model)
    thinning = 10
    mcmc = MCMC(nuts_kernel,
                num_warmup=2_000,
                num_samples=num_posterior_samples * thinning,
                thinning=thinning)
    mcmc.run(random.PRNGKey(1), x_obs_original, n_obs=n_obs)
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
    # sim_data = MA2(t1_bounded, t2_bounded, n_obs=n_obs, batch_size=n_sims, key=sub_key)
    # sim_summ_data = sim_data
    # sim_summ_data = jnp.array((jnp.var(sim_data, axis=1), autocov(sim_data, lag=1), autocov(sim_data, lag=2)))

    batch_size = min(1000, n_sims)
    sim_summ_data = get_summaries_batches(key, t1_bounded, t2_bounded, n_obs, n_sims, batch_size)

    thetas = jnp.column_stack([t1, t2])
    # thetas = jnp.vstack([thetas, true_params])  # TODO: FOR TESTING
    thetas_mean = thetas.mean(axis=0)
    thetas_std = thetas.std(axis=0)
    thetas = (thetas - thetas_mean) / thetas_std

    # sim_summ_data = sim_summ_data.T
    # sim_summ_data = jnp.vstack([sim_summ_data, x_obs])  # TODO: FOR TESTING
    sim_summ_data_mean = sim_summ_data.mean(axis=0)
    sim_summ_data_std = sim_summ_data.std(axis=0)
    sim_summ_data = (sim_summ_data - sim_summ_data_mean) / sim_summ_data_std

    x_obs = (x_obs - sim_summ_data_mean) / sim_summ_data_std

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
        learning_rate=5e-4,  # TODO: could experiment with
        max_epochs=2000,
        max_patience=10,
        batch_size=256,
    )

    plt.plot(losses['train'], label='train')
    plt.plot(losses['val'], label='val')
    plt.savefig(f'{dirname}losses.pdf')
    plt.clf()

    key, sub_key = random.split(key)

    posterior_samples_original = flow.sample(sub_key, sample_shape=(num_posterior_samples,), condition=x_obs)
    posterior_samples = (posterior_samples_original * thetas_std) + thetas_mean
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
    xgrid, ygrid = jnp.meshgrid(jnp.linspace(-3, 3, resolution),
                                jnp.linspace(-3, 3, resolution))
    xy_input = jnp.column_stack([xgrid.ravel(), ygrid.ravel()])
    zgrid = jnp.exp(flow.log_prob(xy_input, condition=x_obs)).reshape(resolution, resolution)
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
    plt.axvline(x_obs_original[0], color='red')
    plt.savefig(f'{dirname}ppc_var_identifiable.pdf')
    plt.clf()

    plt.hist(ppc_summaries[-2], bins=50)
    plt.axvline(x_obs_original[-2], color='red')
    plt.savefig(f'{dirname}ppc_ac1_identifiable.pdf')
    plt.clf()

    plt.hist(ppc_summaries[-1], bins=50)
    plt.axvline(x_obs_original[-1], color='red')
    plt.savefig(f'{dirname}ppc_ac2_identifiable.pdf')
    plt.clf()

    kl = kullback_leibler(true_posterior_samples, posterior_samples)

    lengthscale = median_heuristic(jnp.vstack([true_posterior_samples,
                                               posterior_samples]))
    mmd = unbiased_mmd(true_posterior_samples, posterior_samples, lengthscale)

    with open(f'{dirname}posterior_samples.pkl', 'wb') as f:
        pkl.dump(posterior_samples, f)

    with open(f'{dirname}true_posterior_samples.pkl', 'wb') as f:
        pkl.dump(true_posterior_samples, f)

    with open(f'{dirname}kl.txt', 'w') as f:
        f.write(str(kl))

    with open(f'{dirname}mmd.txt', 'w') as f:
        f.write(str(mmd))

    num_coverage_samples = 100
    coverage_levels = [0.8, 0.9, 0.95]

    # bias/coverage for true parameter
    true_params_unbounded = jnp.array([logit((true_params[0] + 2) / 4),
                                       logit((true_params[1] + 1) / 2)])
    true_params_standardised = (true_params_unbounded - thetas_mean) / thetas_std
    bias = jnp.mean(posterior_samples, axis=0) - true_params
    pdf_posterior_samples = flow.log_prob(posterior_samples_original,
                                          x_obs)
    pdf_posterior_samples = jnp.sort(pdf_posterior_samples.ravel(),
                                     descending=True)
    pdf_theta = flow.log_prob(true_params_standardised, x_obs)
    true_in_credible_interval = [0, 0, 0]
    for i, level in enumerate(coverage_levels):
        coverage_index = int(level * num_posterior_samples)
        pdf_posterior_sample = pdf_posterior_samples[coverage_index]
        if pdf_theta > pdf_posterior_sample:
            true_in_credible_interval[i] = 1

    with open(f"{dirname}true_in_credible_interval.txt", "w") as f:
        f.write(f"{true_in_credible_interval}\n")

    with open(f"{dirname}true_bias.txt", "w") as f:
        f.write(f"{bias}\n")

    coverage_levels_counts = [0, 0, 0]
    biases = jnp.array([])

    for i in range(num_coverage_samples):
        key, sub_key = random.split(key)
        t1_bounded = CustomPrior_t1.rvs(2., size=(1,), key=sub_key)
        t1 = logit((t1_bounded + 2) / 4)

        key, sub_key = random.split(key)
        t2_bounded = CustomPrior_t2.rvs(t1_bounded, 1., size=(1,), key=sub_key)
        t2 = logit((t2_bounded + 1) / 2)
        theta_draw_original = jnp.column_stack([t1_bounded, t2_bounded])
        theta_draw = jnp.column_stack([t1, t2])

        theta_draw = (theta_draw - thetas_mean) / thetas_std

        key, sub_key = random.split(sub_key)
        x_draw = get_summaries_batches(sub_key,
                                       *theta_draw_original.T,
                                       n_obs=n_obs,
                                       n_sims=1, batch_size=1)
        x_draw = (x_draw - sim_summ_data_mean) / sim_summ_data_std

        posterior_samples_original = flow.sample(sub_key,
                                                 sample_shape=(num_posterior_samples,),
                                                 condition=x_draw)
        posterior_samples = (posterior_samples_original * thetas_std) + thetas_mean
        posterior_samples = jnp.squeeze(posterior_samples)
        posterior_samples = posterior_samples.at[:, 0].set(4 * expit(posterior_samples[:, 0]) - 2)
        posterior_samples = posterior_samples.at[:, 1].set(2 * expit(posterior_samples[:, 1]) - 1)
        bias = jnp.mean(posterior_samples, axis=0) - theta_draw_original
        biases = jnp.concatenate((biases, bias.ravel()))
        pdf_posterior_samples = flow.log_prob(posterior_samples_original,
                                              x_draw)
        pdf_posterior_samples = jnp.sort(pdf_posterior_samples.ravel(),
                                         descending=True)
        pdf_theta = flow.log_prob(theta_draw, x_draw)

        for i, level in enumerate(coverage_levels):
            coverage_index = int(level * num_posterior_samples)
            pdf_posterior_sample = pdf_posterior_samples[coverage_index]
            if pdf_theta > pdf_posterior_sample:
                coverage_levels_counts[i] += 1

    print(coverage_levels_counts)
    estimated_coverage = jnp.array(coverage_levels_counts)/num_coverage_samples

    with open(f"{dirname}coverage.txt", "w") as f:
        f.write(f"{estimated_coverage}\n")

    with open(f"{dirname}biases.txt", "w") as f:
        f.write(f"{biases}\n")

    return kl, mmd


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="run_ma2_identifiable.py",
        description="Run MA(2) model.",
        epilog="Example usage: python run_ma2_identifiabl.py"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_obs", type=int, default=1_000)
    parser.add_argument("--n_sims", type=int, default=100_000)
    args = parser.parse_args()
    run_ma2_identifiable(args)
