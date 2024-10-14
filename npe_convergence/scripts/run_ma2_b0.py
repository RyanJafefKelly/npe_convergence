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
                                          autocov_exact, sample_autocov_variance,
                                          autocov, numpyro_model_b0,
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
    dirname = "res/ma2_b0/npe_n_obs_" + str(n_obs) + "_n_sims_" + str(n_sims) + "_seed_" + str(seed) + "/"
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    key = random.PRNGKey(seed)    # x_obs_original = x_obs.copy()

    key, sub_key = random.split(key)

    num_posterior_samples = 4_000  # TODO!

    t1 = random.uniform(sub_key, shape=(n_sims,))
    t1_bounded = 2 * t1 - 1
    t1 = logit(t1)

    t2 = random.uniform(sub_key, shape=(n_sims,))
    t2_bounded = 2 * t2 - 1
    t2 = logit(t2)

    # key, sub_key = random.split(key)
    # t1_bounded = CustomPrior_t1.rvs(2., size=(n_sims,), key=sub_key)
    # t1 = logit((t1_bounded + 1) / 4)

    # key, sub_key = random.split(key)
    # t2_bounded = CustomPrior_t2.rvs(t1_bounded, 1., size=(n_sims,), key=sub_key)
    # t2 = logit((t2_bounded + 1) / 2)

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

    # x_obs = (x_obs - sim_summ_data_mean) / sim_summ_data_std

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

    # NOTE: b0,0 lower
    b_0_0s = [0.01, 0.1, 0.5, 0.99][::-1]
    # TODO! REPEATED RUNS
    for b_0_0 in b_0_0s:
        b_0 = jnp.array([b_0_0, 0.0, 0.0])
        # TODO: exact sampling
        nuts_kernel = NUTS(numpyro_model_b0)
        thinning = 10
        num_chains = 16
        mcmc = MCMC(nuts_kernel,
                    num_warmup=20_000,
                    num_samples=num_posterior_samples*thinning // num_chains,
                    thinning=thinning,
                    num_chains=num_chains
                    )
        mcmc.run(random.PRNGKey(1), obs=b_0,
                #  init_params={'t1': 0., 't2': 0.},
                 n_obs=n_obs)
        mcmc.print_summary()
        samples = mcmc.get_samples()
        true_posterior_samples = jnp.column_stack([samples['t1'], samples['t2']])
        plt.hist(true_posterior_samples[:, 0], bins=50)
        plt.savefig(f'{dirname}t1_posterior_exact_b_0_0_{b_0_0}.pdf')
        plt.clf()

        plt.hist(true_posterior_samples[:, 1], bins=50)
        plt.savefig(f'{dirname}t2_posterior_exact_b_0_0_{b_0_0}.pdf')
        plt.clf()

        # TODO: flow samples
        b_0_standardised = (b_0 - sim_summ_data_mean) / sim_summ_data_std
        key, sub_key = random.split(key)
        posterior_samples_original = flow.sample(sub_key,
                                                 sample_shape=(num_posterior_samples,),
                                                 condition=b_0_standardised)
        posterior_samples = (posterior_samples_original * thetas_std) + thetas_mean
        posterior_samples = posterior_samples.at[:, 0].set(2 * expit(posterior_samples[:, 0]) - 1)
        posterior_samples = posterior_samples.at[:, 1].set(2 * expit(posterior_samples[:, 1]) - 1)
        plt.hist(posterior_samples[:, 0], bins=50)
        plt.savefig(f'{dirname}t1_posterior_flow_b_0_0_{b_0_0}.pdf')
        plt.clf()

        plt.hist(posterior_samples[:, 1], bins=50)
        plt.savefig(f'{dirname}t2_posterior_flow_b_0_0_{b_0_0}.pdf')
        plt.clf()

        kl = kullback_leibler(true_posterior_samples, posterior_samples)
        print(f"KL divergence for b_0 = {b_0_0}: {kl}")
        with open(f'{dirname}kl_{b_0_0}.txt', 'w') as f:
            f.write(str(kl))

    # NOTE: b0, 1
    b_0_1s = [0, 0.1, 0.5, 1.0][::-1]
    for b_0_1 in b_0_1s:
        b_0 = jnp.array([2.0, b_0_1, 0.0])
        nuts_kernel = NUTS(numpyro_model_b0)
        thinning = 10
        mcmc = MCMC(nuts_kernel,
                    num_warmup=2_000,
                    num_samples=num_posterior_samples * thinning,
                    thinning=thinning)
        mcmc.run(random.PRNGKey(1), obs=b_0, init_params={'t1': 0., 't2': 0.}, n_obs=n_obs)
        mcmc.print_summary()
        samples = mcmc.get_samples()
        true_posterior_samples = jnp.column_stack([samples['t1'], samples['t2']])
        plt.hist(true_posterior_samples[:, 0], bins=50)
        plt.savefig(f'{dirname}t1_posterior_exact_b_0_1_{b_0_1}.pdf')
        plt.clf()

        plt.hist(true_posterior_samples[:, 1], bins=50)
        plt.savefig(f'{dirname}t2_posterior_exact_b_0_1_{b_0_1}.pdf')
        plt.clf()

        # TODO: flow samples
        b_0_standardised = (b_0 - sim_summ_data_mean) / sim_summ_data_std
        key, sub_key = random.split(key)
        posterior_samples_original = flow.sample(sub_key,
                                                 sample_shape=(num_posterior_samples,),
                                                 condition=b_0_standardised)
        posterior_samples = (posterior_samples_original * thetas_std) + thetas_mean
        posterior_samples = posterior_samples.at[:, 0].set(2 * expit(posterior_samples[:, 0]) - 1)
        posterior_samples = posterior_samples.at[:, 1].set(2 * expit(posterior_samples[:, 1]) - 1)
        plt.hist(posterior_samples[:, 0], bins=50)
        plt.savefig(f'{dirname}t1_posterior_flow_b_0_1_{b_0_1}.pdf')
        plt.clf()

        plt.hist(posterior_samples[:, 1], bins=50)
        plt.savefig(f'{dirname}t2_posterior_flow_b_0_1_{b_0_1}.pdf')
        plt.clf()

        kl = kullback_leibler(true_posterior_samples, posterior_samples)
        print(f"KL divergence for b_0_1 = {b_0_1}: {kl}")

    # TODO: experiment on compatibility
    # bs = [1.0, 1.5, 1.9, 1.99, 3, 4, 5]
    for i in range(100):
        # test_theta = jnp.array([1.99, 0.999])
        y_obs_synth = jnp.array([
            autocov_exact(true_params, 0, 2),
            autocov_exact(true_params, 1, 2) + i * jnp.sqrt(sample_autocov_variance(true_params, 1, n_obs, 2)),
            autocov_exact(true_params, 2, 2)
            ])
        y_obs_synth = (y_obs_synth - sim_summ_data_mean) / sim_summ_data_std

        # TODO: QUICK ABC CHECK
        distances = jnp.linalg.norm(sim_summ_data - y_obs_synth, axis=1)
        cutoff_index = int(len(distances) * 0.01)
        # closest_distances = np.partition(distances, cutoff_index)[:cutoff_index]
        closest_indices = jnp.argpartition(distances, cutoff_index)[:cutoff_index]
        closest_simulations = sim_summ_data[closest_indices]
        closest_t1 = t1_bounded[closest_indices]

        # y_obs_synth = y_obs_synth.at[1].set()
        key, sub_key = random.split(key)
        posterior_samples_original = flow.sample(sub_key, sample_shape=(num_posterior_samples,), condition=y_obs_synth)
        posterior_samples = (posterior_samples_original * thetas_std) + thetas_mean
        posterior_samples = posterior_samples.at[:, 0].set(4 * expit(posterior_samples[:, 0]) - 2)
        posterior_samples = posterior_samples.at[:, 1].set(2 * expit(posterior_samples[:, 1]) - 1)
        _, bins, _ = plt.hist(posterior_samples[:, 0], bins=50)
        # plt.xlim(0, 1)
        # plt.hist(true_posterior_samples[:, 0], bins=bins)
        # print('')
        plt.axvline(jnp.mean(closest_t1), color='red')
        plt.savefig(f'{dirname}t1_posterior_identifiable_i_{i}.pdf')
        plt.clf()

        plt.hist(posterior_samples[:, 1], bins=50)
        plt.savefig(f'{dirname}t2_posterior_identifiable_i_{i}.pdf')
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
    parser.add_argument("--n_sims", type=int, default=1_000)
    args = parser.parse_args()
    run_ma2_identifiable(args)
