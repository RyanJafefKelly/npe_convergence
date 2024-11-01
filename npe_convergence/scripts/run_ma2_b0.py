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
from numpyro.diagnostics import hpdi  # type: ignore
import numpy as np

from npe_convergence.examples.ma2 import (autocov_exact,
                                          sample_autocov_variance,
                                          numpyro_model_b0,
                                          get_summaries_batches)
from npe_convergence.metrics import (kullback_leibler, median_heuristic,
                                     unbiased_mmd)
import numpyro.distributions as dist  # type: ignore
import blackjax  # type: ignore
import blackjax.smc.resampling as resampling  # type: ignore
import jax


def run_ma2_b0(*args, **kwargs):
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
    true_params = jnp.array([0.6, 0.2])
    key = random.key(seed)

    key, sub_key = random.split(key)
    x_obs = get_summaries_batches(sub_key,
                                  jnp.atleast_1d(true_params[0]),
                                  jnp.atleast_1d(true_params[1]),
                                  n_obs,
                                  n_sims,
                                  1)
    x_obs_original = x_obs.copy()

    num_posterior_samples = 10_000

    key, sub_key = random.split(key)
    t1 = random.uniform(sub_key, shape=(n_sims,))
    t1_bounded = 2 * t1 - 1
    t1 = logit(t1)

    key, sub_key = random.split(key)
    t2 = random.uniform(sub_key, shape=(n_sims,))
    t2_bounded = 2 * t2 - 1
    t2 = logit(t2)

    key, sub_key = random.split(key)

    batch_size = min(1000, n_sims)
    sim_summ_data = get_summaries_batches(key, t1_bounded, t2_bounded, n_obs,
                                          n_sims, batch_size)

    thetas = jnp.column_stack([t1, t2])
    thetas_mean = thetas.mean(axis=0)
    thetas_std = thetas.std(axis=0)
    thetas = (thetas - thetas_mean) / thetas_std

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

    nuts_kernel = NUTS(numpyro_model_b0)
    thinning = 10
    mcmc = MCMC(nuts_kernel,
                num_warmup=2_000,
                num_samples=num_posterior_samples * thinning,
                thinning=thinning)
    mcmc.run(random.key(1), obs=x_obs_original,
             init_params={'t1': 0., 't2': 0.}, n_obs=n_obs)
    mcmc.print_summary()
    samples = mcmc.get_samples()
    true_posterior_samples = jnp.column_stack([samples['t1'], samples['t2']])

    key, sub_key = random.split(key)
    posterior_samples_original = flow.sample(sub_key, sample_shape=(num_posterior_samples,), condition=x_obs)
    posterior_samples = (posterior_samples_original * thetas_std) + thetas_mean
    posterior_samples = posterior_samples.at[:, 0].set(2 * expit(posterior_samples[:, 0]) - 1)
    posterior_samples = posterior_samples.at[:, 1].set(2 * expit(posterior_samples[:, 1]) - 1)
    posterior_samples = jnp.squeeze(posterior_samples)

    # Plot t1 exact and flow samples
    _, bins, _ = plt.hist(true_posterior_samples[:, 0], bins=50, alpha=0.5, label='Exact')
    plt.hist(posterior_samples[:, 0], bins=bins, alpha=0.5, label='Flow')
    plt.legend()
    plt.savefig(f'{dirname}t1_posterior.pdf')
    plt.clf()

    # Plot t2 exact and flow samples
    _, bins, _ = plt.hist(true_posterior_samples[:, 1], bins=50, alpha=0.5, label='Exact')
    plt.hist(posterior_samples[:, 1], bins=bins, alpha=0.5, label='Flow')
    plt.legend()
    plt.savefig(f'{dirname}t2_posterior.pdf')
    plt.clf()
    kl = kullback_leibler(true_posterior_samples,
                          posterior_samples)

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
    true_params_unbounded = jnp.array([logit((true_params[0] + 1) / 2),
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

    coverage_levels_counts = np.zeros((2, 3))  # rows - params, cols - coverage levels
    biases = jnp.array([])

    for i in range(num_coverage_samples):
        theta_draw = (true_params - thetas_mean) / thetas_std

        key, sub_key = random.split(sub_key)
        x_draw = get_summaries_batches(sub_key,
                                       jnp.atleast_1d(true_params[0]),
                                       jnp.atleast_1d(true_params[1]),
                                       n_obs=n_obs,
                                       n_sims=1,
                                       batch_size=1)
        x_draw = (x_draw - sim_summ_data_mean) / sim_summ_data_std

        posterior_samples_original = flow.sample(sub_key,
                                                 sample_shape=(num_posterior_samples,),
                                                 condition=x_draw)
        posterior_samples = (posterior_samples_original * thetas_std) + thetas_mean
        posterior_samples = jnp.squeeze(posterior_samples)
        posterior_samples = posterior_samples.at[:, 0].set(2 * expit(posterior_samples[:, 0]) - 1)
        posterior_samples = posterior_samples.at[:, 1].set(2 * expit(posterior_samples[:, 1]) - 1)
        bias = jnp.mean(posterior_samples, axis=0) - true_params
        biases = jnp.concatenate((biases, bias.ravel()))
        pdf_posterior_samples = flow.log_prob(posterior_samples_original,
                                              x_draw)
        pdf_posterior_samples = jnp.sort(pdf_posterior_samples.ravel(),
                                         descending=True)
        pdf_theta = flow.log_prob(theta_draw, x_draw)

        for i in range(2):  # check if true param in credible interval, marginally
            posterior_samples_i = posterior_samples[:, i].ravel()
            for ii, coverage_level in enumerate(coverage_levels):
                lower, upper = hpdi(posterior_samples_i, coverage_level)
                if lower < true_params[i] < upper:
                    coverage_levels_counts[i, ii] += 1

    print(coverage_levels_counts)
    estimated_coverage = jnp.array(coverage_levels_counts)/num_coverage_samples

    np.save(f"{dirname}estimated_coverage.npy", estimated_coverage)
    np.save(f"{dirname}biases.npy", biases)

    def log_det_jacobian(u):
        # Compute the log absolute determinant of the Jacobian
        s = expit(u)
        dt_du = 2.0 * s * (1.0 - s)
        return jnp.sum(jnp.log(jnp.abs(dt_du)), axis=-1)

    def compute_log_prior_single(theta):
        t1, t2 = theta
        t1 = 2 * expit(t1) - 1
        t2 = 2 * expit(t2) - 1
        logp_t1 = dist.Uniform(-1, 1, validate_args=True).log_prob(t1)
        logp_t2 = dist.Uniform(-1, 1, validate_args=True).log_prob(t2)
        # log_det_jacobian_val = log_det_jacobian(theta)
        return logp_t1 + logp_t2  # + log_det_jacobian_val

    def log_prior_fn(params):
        if params.ndim == 1:
            # Unbatched input
            return compute_log_prior_single(params)
        else:
            # Batched input
            compute_log_prior_batch = jax.vmap(compute_log_prior_single)
            return compute_log_prior_batch(params)


    def compute_gamma_h(thetas):
        gamma_h = jnp.zeros(9)  # h from -4 to 4
        h_vals = jnp.arange(-4, 5)
        theta0 = 1.0

        def gamma_fn(h):
            abs_h = jnp.abs(h)
            gamma = jnp.where(
                abs_h == 0,
                theta0 + thetas[0] ** 2 + thetas[1] ** 2,
                jnp.where(
                    abs_h == 1,
                    thetas[0] + thetas[0] * thetas[1],
                    jnp.where(abs_h == 2, thetas[1], 0.0),
                ),
            )
            return gamma

        gamma_h = jnp.array([gamma_fn(h) for h in h_vals])
        return gamma_h

    # Compute the covariance matrix of sample autocovariances
    def compute_covariance_matrix(thetas, n_obs, max_lag=2):
        gamma_h = compute_gamma_h(thetas)

        num_lags = max_lag + 1
        k_vals = jnp.arange(num_lags)
        h_vals = jnp.arange(-2, 3)
        i_vals = jnp.arange(-2, 3)

        K1, K2 = jnp.meshgrid(k_vals, k_vals, indexing="ij")
        delta_k = K1 - K2

        # Compute S1
        h_plus_delta = h_vals[:, None, None] + delta_k[None, :, :]
        valid_S1 = (h_plus_delta >= -2) & (h_plus_delta <= 2)
        h_indices = h_vals[:, None, None] + 4
        h_plus_delta_indices = h_plus_delta + 4

        gamma_h_values = gamma_h[h_indices]
        gamma_h_plus_delta_values = gamma_h[h_plus_delta_indices]
        gamma_products_S1 = gamma_h_values * gamma_h_plus_delta_values * valid_S1
        S1 = jnp.sum(gamma_products_S1, axis=0)

        # Compute S2
        k1_plus_i = K1[None, :, :] + i_vals[:, None, None]
        k2_minus_i = K2[None, :, :] - i_vals[:, None, None]
        valid_S2 = (
            (k1_plus_i >= -2)
            & (k1_plus_i <= 2)
            & (k2_minus_i >= -2)
            & (k2_minus_i <= 2)
        )
        k1_plus_i_indices = k1_plus_i + 4
        k2_minus_i_indices = k2_minus_i + 4

        gamma_k1_i = gamma_h[k1_plus_i_indices]
        gamma_k2_i = gamma_h[k2_minus_i_indices]
        gamma_products_S2 = gamma_k1_i * gamma_k2_i * valid_S2
        S2 = jnp.sum(gamma_products_S2, axis=0)

        cov_matrix = (S1 + S2) / n_obs
        return cov_matrix


    def compute_log_likelihood_single(theta, obs, n_obs=100):
        ma_order = 2
        thetas = 2 * expit(theta) - 1

        mean = jnp.array(
            [autocov_exact(thetas, k, ma_order) for k in range(ma_order + 1)]
        )

        cov_matrix = compute_covariance_matrix(thetas, n_obs, max_lag=ma_order)
        jitter = 1e-6
        cov_matrix += jitter * jnp.eye(ma_order + 1)

        log_lik = dist.MultivariateNormal(mean, cov_matrix).log_prob(obs)
        return log_lik


    def log_likelihood_fn(params, obs, n_obs=100):
        # Check if params is batched or unbatched
        if params.ndim == 1:
            # Unbatched input
            return compute_log_likelihood_single(params, obs, n_obs)
        else:
            # Batched input
            compute_log_likelihood_batch = jax.vmap(compute_log_likelihood_single, in_axes=(0, None, None))
            return compute_log_likelihood_batch(params, obs, n_obs)

    # step_sizes = jnp.linspace(1e-3, 1e-2, num_posterior_samples)  # Varying step sizes
    hmc_parameters = {
        'step_size': jnp.full(num_posterior_samples, 5e-3),
        'inverse_mass_matrix': 0.1*jnp.ones((num_posterior_samples, 2)),
        'num_integration_steps': jnp.full(num_posterior_samples, 100),
    }

    if thetas.shape[0] < num_posterior_samples:
        repeated_thetas = jnp.resize(thetas, (num_posterior_samples, thetas.shape[1]))
        initial_particles = repeated_thetas[:num_posterior_samples, :]
    else:
        initial_particles = thetas[:num_posterior_samples, :]

    def smc_inference_loop(rng_key, smc_kernel, initial_state):
        """Run the temepered SMC algorithm.

        We run the adaptive algorithm until the tempering parameter lambda reaches the value
        lambda=1.

        """

        def cond(carry):
            i, state, _k = carry
            return state.lmbda < 1

        def one_step(carry):
            i, state, k = carry
            k, subk = random.split(k, 2)
            state, _ = smc_kernel(subk, state)
            return i + 1, state, k

        n_iter, final_state, _ = jax.lax.while_loop(
            cond, one_step, (0, initial_state, rng_key)
        )

        return n_iter, final_state

    # def update_inverse_mass_matrix(particles):
    #     # particles: (num_particles, parameter_dim)
    #     # Compute the covariance matrix of the particles
    #     particles_mean = jnp.mean(particles, axis=0)
    #     centered_particles = particles - particles_mean
    #     covariance_matrix = jnp.dot(centered_particles.T, centered_particles) / (particles.shape[0] - 1)
    #     # Regularize the covariance matrix to ensure it's positive definite
    #     covariance_matrix += 1e-6 * jnp.eye(covariance_matrix.shape[0])
    #     # Compute the inverse mass matrix
    #     inverse_mass_matrix = jnp.linalg.inv(covariance_matrix)
    #     return inverse_mass_matrix

    #  NOTE: b0,0 lower
    b_0_0s = [0.01, 0.1, 0.25, 0.5, 0.75, 0.99]

    for b_0_0 in b_0_0s:
        b_0 = jnp.array([b_0_0, 0.0, 0.0])
        tempered_smc = blackjax.adaptive_tempered_smc(
            log_prior_fn,
            lambda params: log_likelihood_fn(params, obs=b_0, n_obs=n_obs),
            blackjax.hmc.build_kernel(),
            blackjax.hmc.init,
            hmc_parameters,
            resampling.systematic,
            0.75,
            num_mcmc_steps=5,
        )
        initial_smc_state = tempered_smc.init(initial_particles)
        key, sub_key = random.split(key)
        n_iter, smc_state = smc_inference_loop(sub_key, tempered_smc.step, initial_smc_state)

        print("Number of steps in the adaptive algorithm: ", n_iter)
        true_posterior_samples = smc_state.particles
        true_posterior_samples = 2 * expit(true_posterior_samples) - 1

        plt.hist(true_posterior_samples[:, 0], bins=50)
        plt.savefig(f'{dirname}t1_posterior_exact_b_0_0_{b_0_0}.pdf')
        plt.clf()

        plt.hist(true_posterior_samples[:, 1], bins=50)
        plt.savefig(f'{dirname}t2_posterior_exact_b_0_0_{b_0_0}.pdf')
        plt.clf()

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

        # t1s = jnp.linspace(-1.0, 1.0, num=100)
        # t2s = jnp.linspace(-1.0, 1.0, num=100)
        # log_pdfs = jnp.zeros((100, 100))
        # for ii, t1 in enumerate(t1s):
        #     for jj, t2 in enumerate(t2s):
        #         log_pdf = 0.0
        #         for i in range(3):
        #             mean = autocov_exact(jnp.array([t1, t2]), i, 2)
        #             y_var = sample_autocov_variance(jnp.array([t1, t2]), i, n_obs, 2)
        #             y_stdev = jnp.sqrt(y_var)
        #             log_pdf += dist.Normal(mean, y_stdev).log_prob(b_0[i])

        #         log_pdfs = log_pdfs.at[ii, jj].set(log_pdf)

        # plt.contourf(t1s, t2s, log_pdfs)
        # plt.savefig(f"{dirname}t1_t2_{b_0_0}_pdfs.pdf")
        # plt.clf()

    return kl, mmd


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="run_ma2_b0.py",
        description="Run MA(2) model.",
        epilog="Example usage: python run_ma2_b0.py"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_obs", type=int, default=5000)
    parser.add_argument("--n_sims", type=int, default=333_333)
    args = parser.parse_args()
    run_ma2_b0(args)
