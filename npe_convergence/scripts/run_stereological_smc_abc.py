import jax.numpy as jnp
import jax.random as random

import os
from npe_convergence.examples.stereological import stereological, get_prior_samples, transform_to_unbounded, transform_to_bounded

import matplotlib.pyplot as plt

import elfi  # type: ignore
import numpy as np
import scipy.stats as ss  # type: ignore
from functools import partial
import argparse
import pickle as pkl


def stereological_sim(poisson_rate, pareto_scale, pareto_shape, n_obs=100, batch_size=1, random_state=None):
    v_0 = 5
    rng = np.random.default_rng(random_state.randint(1 << 31))

    if np.isscalar(poisson_rate):
        poisson_rate = np.full(batch_size, np.array(poisson_rate))

    poisson_rate = np.atleast_1d(poisson_rate)
    pareto_shape = np.atleast_1d(pareto_shape)
    pareto_scale = np.atleast_1d(pareto_scale)

    poisson_samples = rng.poisson(poisson_rate[:, np.newaxis], size=(batch_size, n_obs))
    max_locs = np.max(poisson_samples, axis=0)

    pareto_samples = np.empty((batch_size, n_obs, max_locs.max(), 3))
    for i in range(batch_size):
        for j in range(n_obs):
            pareto_samples[i, j, :int(poisson_samples[i, j]), :] = ss.genpareto.rvs(
                pareto_shape[i], scale=pareto_scale[i], size=(int(poisson_samples[i, j]), 3), random_state=rng)

    pareto_samples = np.max(pareto_samples, axis=-1) + v_0
    unif_samples = rng.uniform(size=(batch_size, n_obs, max_locs.max(), 2))
    V1 = (pareto_samples - v_0) * unif_samples[..., 0] * pareto_samples + v_0
    V2 = (pareto_samples - v_0) * unif_samples[..., 1] * pareto_samples + v_0
    V = np.maximum(V1, V2)
    for i in range(batch_size):
        for j in range(n_obs):
            V[i, j, int(poisson_samples[i, j]):] = np.nan

    return V.reshape((batch_size, n_obs, -1))


def get_summaries(x):
    # Counting the number of non-NaN entries per sample
    num_inclusions = np.sum(~np.isnan(x), axis=-1)

    # Using nanmin, nanmean, and nanmax to ignore NaN values
    min_values = np.nanmin(x, axis=-1)
    mean_values = np.nanmean(x, axis=-1)
    max_values = np.nanmax(x, axis=-1)

    # Computing logarithms and replacing -inf with a large negative value (e.g., -1e20)
    log_min = np.log(min_values)
    log_mean = np.log(mean_values)
    log_max = np.log(max_values)

    log_min[np.isneginf(log_min)] = -1e+20
    log_mean[np.isneginf(log_mean)] = -1e+20
    log_max[np.isneginf(log_max)] = -1e+20

    # Take mean over n_obs
    num_inclusions = np.mean(num_inclusions, axis=-1)
    log_min = np.mean(log_min, axis=-1)
    log_mean = np.mean(log_mean, axis=-1)
    log_max = np.mean(log_max, axis=-1)

    # Stacking the summaries into a single array, transposed for compatibility with ELFI's expectations
    summaries = np.column_stack([num_inclusions, log_min, log_mean, log_max])

    return summaries


def get_model(n_obs=100, true_params=None, seed=None):
    m = elfi.new_model()

    if true_params is None:
        true_params = [100, 2, -0.1]

    poisson_rate = elfi.Prior('uniform', 30, 170, model=m, name='poisson_rate')
    pareto_scale = elfi.Prior('uniform', 0, 15, model=m, name='pareto_scale')
    poisson_rate = elfi.Prior('uniform', -3, 6, model=m, name='pareto_rate')

    # simulation
    fn_simulator = partial(stereological_sim, n_obs=n_obs)

    y_obs = stereological_sim(*true_params, n_obs=n_obs, batch_size=1, random_state=np.random.RandomState(seed))

    elfi.Simulator(fn_simulator, m['poisson_rate'], m['pareto_scale'], m['pareto_rate'], observed=y_obs, name='stereological')

    # summary
    ss = elfi.Summary(get_summaries, m['stereological'], name='ss')
    elfi.Distance('euclidean', ss, name='d')
    return m


def run_stereological_smc_abc(*args, **kwargs):
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
    y_obs_full = stereological(subkey, *true_params, n_obs=n_obs)
    y_obs = get_summaries(y_obs_full)
    # plt.hist(x_obs.ravel())
    # plt.savefig("x_obs.pdf")
    y_obs_original = y_obs.copy()

    m = get_model(n_obs=n_obs, true_params=true_params, seed=seed)
    m.observed['stereological'] = y_obs_full.reshape((1, n_obs, -1))

    np.random.seed(seed)

    max_iter = 5
    num_posterior_samples = 4_000
    adaptive_smc = elfi.AdaptiveThresholdSMC(m['d'],
                                             batch_size=1_000,
                                             seed=seed,
                                             q_threshold=0.99)
    adaptive_smc_samples = adaptive_smc.sample(num_posterior_samples,
                                               max_iter=max_iter)

    print(adaptive_smc_samples)

    for i, pop in enumerate(adaptive_smc_samples.populations):
        s = pop.samples
        for k, v in s.items():
            plt.hist(v, bins=30)
            plt.title(k)
            plt.savefig(dirname + k + "_pop_" + str(i) + ".pdf")
            plt.clf()

    with open(dirname + "adaptive_smc_samples.pkl", "wb") as f:
        pkl.dump(adaptive_smc_samples.samples_array, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run_stereological_smc_abc.py",
        description="Run stereological model with SMC ABC.",
        epilog="Example usage: python run_stereological_smc_abc.py"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_obs", type=int, default=100)
    parser.add_argument("--n_sims", type=int, default=None)
    args = parser.parse_args()
    run_stereological_smc_abc(args)
