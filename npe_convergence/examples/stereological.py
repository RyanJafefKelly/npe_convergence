"""Bortot et al. (2007)"""

import jax.numpy as jnp
import jax.random as random
# import numpy as np
# import scipy.stats as ss  # type: ignore
import jax
from jax import lax
from functools import partial
from jax.scipy.special import expit, logit


def transform_to_unbounded(theta):
    t1 = logit((theta[:, 0] - 30) / 170)
    t2 = logit((theta[:, 1]) / 15)
    t3 = logit((theta[:, 2] + 3) / 6)
    return jnp.column_stack([t1, t2, t3])


def transform_to_bounded(x):
    t1 = expit(x[:, 0]) * 170 + 30
    t2 = expit(x[:, 1]) * 15
    t3 = expit(x[:, 2]) * 6 - 3
    return jnp.column_stack([t1, t2, t3])


def get_prior_samples(key, num_samples: int):
    key, subkey_1, subkey_2, subkey3 = random.split(key, 4)
    poisson_rate_samples = 170 * random.uniform(subkey_1, (num_samples,)) + 30
    pareto_scale_samples = 15 * random.uniform(subkey_2, (num_samples,))
    pareto_shape_samples = 6 * random.uniform(subkey3, (num_samples,)) - 3
    return jnp.column_stack([poisson_rate_samples,
                             pareto_scale_samples,
                             pareto_shape_samples])


# def stereological(key, poisson_rate, pareto_scale, pareto_shape, n_obs=100, num_samples=1):
#     # In 1D real-line, time between arrivals follows exponential distribution
#     v_0 = 5  # CONSTANT
#     key, subkey = random.split(key)
#     poisson_rate = jnp.atleast_1d(poisson_rate)
#     pareto_shape = jnp.atleast_1d(pareto_shape)
#     pareto_scale = jnp.atleast_1d(pareto_scale)

#     number_locs = random.poisson(subkey, poisson_rate[:, np.newaxis], shape=(num_samples, n_obs))
#     numpy_seed = random.randint(subkey, shape=(1,), minval=0, maxval=(1 << 31) - 1)[0]
#     rng = np.random.default_rng(int(numpy_seed))

#     pareto_samples = np.empty((num_samples, n_obs, number_locs.max(), 3))
#     for i in range(num_samples):
#         for j in range(n_obs):
#             pareto_samples[i, j, :int(number_locs[i, j]), :] = ss.genpareto.rvs(
#                 pareto_shape[i], scale=pareto_scale[i], size=(int(number_locs[i, j]), 3), random_state=rng)

#     pareto_samples = jnp.max(pareto_samples, axis=-1)
#     pareto_samples = pareto_samples + v_0  # TODO! CHECK

#     key, subkey = random.split(key)
#     unif_sample = random.uniform(subkey, (num_samples, n_obs, jnp.max(number_locs), 2))
#     V1 = (pareto_samples - v_0) * unif_sample[..., 0] * pareto_samples + v_0
#     V2 = (pareto_samples - v_0) * unif_sample[..., 1] * pareto_samples + v_0
#     V_tmp = np.maximum(V1, V2)
#     for i in range(num_samples):
#         for j in range(n_obs):
#             V_tmp[i, j, int(number_locs[i, j]):] = np.nan
#     return jnp.array(V_tmp)


@partial(jax.jit, static_argnums=(4, 5))
def stereological(key, poisson_rate, pareto_scale, pareto_shape, n_obs=100, num_samples=1):
    v_0 = 5.0
    key, subkey = random.split(key)
    poisson_rate = jnp.atleast_1d(poisson_rate)
    pareto_shape = jnp.atleast_1d(pareto_shape)
    pareto_scale = jnp.atleast_1d(pareto_scale)

    # Simulate the number of locations
    number_locs = random.poisson(subkey, poisson_rate[:, None], shape=(num_samples, n_obs))
    # max_number_locs = jnp.max(number_locs)  # max over dynamic axis, consider revising
    fixed_max_locs = 300

    # Simulate Pareto samples
    key, subkey = random.split(key)
    U_pareto = random.uniform(subkey, shape=(num_samples, n_obs, fixed_max_locs, 3))
    c = pareto_shape[:, None, None, None]
    scale = pareto_scale[:, None, None, None]

    pareto_samples = jnp.where(c != 0,
                               scale * ((1 - U_pareto) ** (-c) - 1) / c,
                               scale * (-jnp.log(1 - U_pareto)))

    locs_mask = jnp.arange(fixed_max_locs)[None, None, :, None] < number_locs[:, :, None, None]
    pareto_samples = jnp.where(locs_mask, pareto_samples, -jnp.inf)
    pareto_samples = jnp.max(pareto_samples, axis=-1) + v_0

    # Generate and compute V1, V2
    unif_sample = random.uniform(key, shape=(num_samples, n_obs, fixed_max_locs, 2))
    V1 = (pareto_samples - v_0) * unif_sample[..., 0] + v_0
    V2 = (pareto_samples - v_0) * unif_sample[..., 1] + v_0
    V_tmp = jnp.maximum(V1, V2)
    mask = locs_mask[..., 0]
    V_tmp = jnp.where(mask, V_tmp, jnp.nan)

    return V_tmp


@jax.jit
def get_summaries(x):
    num_inclusions = jnp.sum(~jnp.isnan(x), axis=-1)
    safe_min = jnp.nanmin(x, axis=-1)
    safe_mean = jnp.nanmean(x, axis=-1)
    safe_max = jnp.nanmax(x, axis=-1)

    ssx = jnp.stack([
        num_inclusions,
        jnp.log(safe_min),
        jnp.log(safe_mean),
        jnp.log(safe_max)
    ], axis=-1)

    # clip_value = 1e+8
    # ssx = ssx.at[jnp.where(jnp.isinf(ssx))].set(-10.0)  # NOTE: lazy fix
    # ssx = ssx.at[jnp.where(ssx > clip_value)].set(-10.0)  # NOTE: lazy fix
    ssx = jnp.mean(ssx, axis=-2)  # assuming 4 in last column
    # if not jnp.all(jnp.isfinite(ssx)):
    #     print("WARNING: non-finite summary statistics")
    return ssx


@partial(jax.jit, static_argnums=(2, 3, 4))
def get_summaries_batches(key, thetas, n_obs, n_sims, batch_size):
    poisson_rate = thetas[:, 0]
    pareto_scale = thetas[:, 1]
    pareto_shape = thetas[:, 2]
    summary_size = 4
    num_full_batches = n_sims // batch_size
    remainder = n_sims % batch_size
    num_batches = num_full_batches + (1 if remainder > 0 else 0)

    pad_size = (batch_size - remainder) % batch_size
    total_size = n_sims + pad_size

    poisson_rate_padded = jnp.pad(poisson_rate, (0, pad_size), mode='edge')
    pareto_scale_padded = jnp.pad(pareto_scale, (0, pad_size), mode='edge')
    pareto_shape_padded = jnp.pad(pareto_shape, (0, pad_size), mode='edge')

    all_summaries = jnp.zeros((total_size, summary_size))
    keys = random.split(key, num_batches)
    start_indices = jnp.arange(num_batches) * batch_size

    def body(carry, xs):
        all_summaries = carry
        key, start_idx = xs
        indices = jnp.array([start_idx])

        poisson_rate_batch = lax.dynamic_slice(poisson_rate_padded, indices, [batch_size])
        pareto_scale_batch = lax.dynamic_slice(pareto_scale_padded, indices, [batch_size])
        pareto_shape_batch = lax.dynamic_slice(pareto_shape_padded, indices, [batch_size])

        sim_data_batch = stereological(key, poisson_rate_batch, pareto_scale_batch, pareto_shape_batch, n_obs=n_obs, num_samples=batch_size)
        sim_summ_data_batch = get_summaries(sim_data_batch)

        all_summaries = lax.dynamic_update_slice(all_summaries, sim_summ_data_batch, [start_idx, 0])

        return all_summaries, None

    xs = (keys, start_indices)
    all_summaries, _ = lax.scan(body, all_summaries, xs)
    all_summaries = all_summaries[:n_sims, :]
    return all_summaries
