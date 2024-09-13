import argparse
import os
import pickle as pkl

import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpyro  # type: ignore
from flowjax.bijections import RationalQuadraticSpline  # type: ignore
from flowjax.distributions import Normal  # type: ignore
from flowjax.flows import coupling_flow  # type: ignore
from flowjax.train.data_fit import fit_to_data  # type: ignore

from npe_convergence.examples.stereological import (get_prior_samples,
                                                    get_summaries,
                                                    get_summaries_batches,
                                                    stereological,
                                                    transform_to_bounded,
                                                    transform_to_unbounded)


def run_stereological(*args, **kwargs):
    try:
        seed, n_obs, n_sims = args
    except ValueError:
        args = args[0]
        seed = args.seed
        n_obs = args.n_obs
        n_sims = args.n_sims

    dirname = "res/stereological/npe_n_obs_" + str(n_obs) + "_n_sims_" + str(n_sims) + "_seed_" + str(seed) +  "/"
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    key = random.PRNGKey(seed)
    true_params = jnp.array([100, 2, -0.1])  # TODO? fixed here ... doesn't matter
    # x_mat = sio.loadmat("npe_convergence/data/data_stereo_real.mat")
    # y_obs = jnp.array(x_mat["y"])
    # y_obs = get_summaries(y_obs)
    key, subkey = random.split(key)
    y_obs = stereological(subkey, *true_params, num_samples=1, n_obs=n_obs)
    y_obs = get_summaries(y_obs)
    y_obs_original = y_obs.copy()
    print('y_obs: ', y_obs)
    key, subkey = random.split(key)
    thetas = get_prior_samples(subkey, n_sims)

    key, subkey = random.split(key)
    # # TODO: BATCHING
    # sim_data = stereological(subkey, *thetas.T, num_samples=n_sims, n_obs=n_obs)
    # sim_summ_data = get_summaries(sim_data)
    batch_size = 1_000
    sim_summ_data = get_summaries_batches(key, thetas, n_obs, n_sims, batch_size)

    thetas = transform_to_unbounded(thetas)

    thetas_mean = thetas.mean(axis=0)
    thetas_std = thetas.std(axis=0)
    thetas = (thetas - thetas_mean) / thetas_std

    sim_summ_data_mean = jnp.nanmean(sim_summ_data, axis=0)  # TODO: hacky fix
    sim_summ_data_std = jnp.nanstd(sim_summ_data, axis=0)

    sim_summ_data = (sim_summ_data - sim_summ_data_mean) / sim_summ_data_std

    key, sub_key = random.split(key)
    theta_dims = 3
    summary_dims = 4
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
    plt.savefig('losses.pdf')
    plt.clf()

    # standardise y_obs
    y_obs = (y_obs - sim_summ_data_mean) / sim_summ_data_std

    num_posterior_samples = 4_000
    posterior_samples = flow.sample(sub_key,
                                    sample_shape=(num_posterior_samples,),
                                    condition=y_obs)
    posterior_samples = (posterior_samples * thetas_std) + thetas_mean
    posterior_samples = jnp.squeeze(posterior_samples)
    posterior_samples = transform_to_bounded(posterior_samples)
    plt.hist(posterior_samples[:, 0], bins=50)
    # plt.xlim(0, 1)
    plt.axvline(true_params[0], color='red')
    plt.savefig(f'{dirname}t1_posterior.pdf')
    plt.clf()

    plt.hist(posterior_samples[:, 1], bins=50)
    plt.axvline(true_params[1], color='red')
    # plt.xlim(0, 1)
    plt.savefig(f'{dirname}t2_posterior.pdf')
    plt.clf()

    plt.hist(posterior_samples[:, 2], bins=50)
    plt.axvline(true_params[2], color='red')
    # plt.xlim(0, 1)
    plt.savefig(f'{dirname}t3_posterior.pdf')
    plt.clf()

    y_obs_original = jnp.squeeze(y_obs_original)

    with open(f'{dirname}posterior_samples.pkl', 'wb') as f:
        pkl.dump(posterior_samples, f)

    num_coverage_samples = 100
    coverage_levels = [0.8, 0.9, 0.95]
    coverage_levels_counts = [0, 0, 0]
    biases = jnp.array([])

    for i in range(num_coverage_samples):
        key, sub_key = random.split(key)
        theta_draw_original = get_prior_samples(sub_key, 1)

        theta_draw = transform_to_unbounded(theta_draw_original)
        theta_draw = (theta_draw - thetas_mean) / thetas_std

        key, sub_key = random.split(sub_key)
        x_draw_original = stereological(sub_key, *theta_draw_original.T,
                                        num_samples=1)
        x_draw = get_summaries(x_draw_original)
        x_draw = (x_draw - sim_summ_data_mean) / sim_summ_data_std

        posterior_samples_original = flow.sample(sub_key,
                                                 sample_shape=(num_posterior_samples,),
                                                 condition=x_draw)
        posterior_samples = (posterior_samples_original * thetas_std) + thetas_mean
        posterior_samples = jnp.squeeze(posterior_samples)
        posterior_samples = transform_to_bounded(posterior_samples)
        bias = jnp.mean(posterior_samples, axis=0) - theta_draw_original
        biases = jnp.concatenate((biases, bias.ravel()))
        pdf_posterior_samples = flow.log_prob(posterior_samples_original,
                                              x_draw)
        pdf_posterior_samples = jnp.sort(pdf_posterior_samples.ravel())
        pdf_theta = flow.log_prob(theta_draw, x_draw)

        for i, level in enumerate(coverage_levels):
            coverage_index = int(level * num_posterior_samples)
            pdf_posterior_sample = pdf_posterior_samples[coverage_index]
            if pdf_theta < pdf_posterior_sample:
                coverage_levels_counts[i] += 1

    print(coverage_levels_counts)
    estimated_coverage = jnp.array(coverage_levels_counts)/num_coverage_samples

    with open(f"{dirname}coverage.txt", "w") as f:
        f.write(f"{estimated_coverage}\n")

    with open(f"{dirname}biases.txt", "w") as f:
        f.write(f"{biases}\n")

    # ppc_samples = stereological(sub_key, *posterior_samples.T, num_samples=num_posterior_samples)
    # ppc_summaries = get_summaries(ppc_samples)
    # ppc_summaries = jnp.squeeze(ppc_summaries)
    # plt.hist(ppc_summaries[:, 0], bins=50)
    # plt.axvline(y_obs_original[0], color='red')
    # plt.savefig(f'{dirname}t1_posterior_stereo_npe_simnum_inclusions_posterior_stereo_npe_sim.pdf')
    # plt.clf()

    # plt.hist(ppc_summaries[:, 1], bins=50)
    # plt.axvline(y_obs_original[1], color='red')
    # plt.savefig(f'{dirname}t1_posterior_stereo_npe_simmin_inclusions_posterior_stereo_npe_sim.pdf')
    # plt.clf()

    # plt.hist(ppc_summaries[:, 2], bins=50)
    # plt.axvline(y_obs_original[2], color='red')
    # plt.savefig(f'{dirname}t1_posterior_stereo_npe_simmean_inclusions_posterior_stereo_npe_sim.pdf')
    # plt.clf()

    # plt.hist(ppc_summaries[:, 3], bins=50)
    # plt.axvline(y_obs_original[3], color='red')
    # plt.savefig(f'{dirname}t1_posterior_stereo_npe_simmax_inclusions_posterior_stereo_npe_sim.pdf')
    # plt.clf()

    return None, None


if __name__ == "__main__":
    numpyro.set_host_device_count(4)
    parser = argparse.ArgumentParser(
        prog="run_stereological.py",
        description="Run stereological model.",
        epilog="Example usage: python run_stereological.py"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_obs", type=int, default=100)
    parser.add_argument("--n_sims", type=int, default=5_000)
    args = parser.parse_args()

    run_stereological(args)
