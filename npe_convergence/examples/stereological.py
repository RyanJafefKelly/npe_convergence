"""Bortot et al. (2007)"""

import jax.numpy as jnp
import scipy.stats as ss  # type: ignore
import jax.random as random
from jax.scipy.special import logit, expit
import numpy as np


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


def stereological(key, poisson_rate, pareto_scale, pareto_shape, n_obs=100, num_samples=1):
    # In 1D real-line, time between arrivals follows exponential distribution
    v_0 = 5  # CONSTANT
    key, subkey = random.split(key)
    poisson_rate = jnp.atleast_1d(poisson_rate)
    pareto_shape = jnp.atleast_1d(pareto_shape)
    pareto_scale = jnp.atleast_1d(pareto_scale)

    number_locs = random.poisson(subkey, poisson_rate[:, np.newaxis], shape=(num_samples, n_obs))
    numpy_seed = random.randint(subkey, shape=(1,), minval=0, maxval=(1 << 31) - 1)[0]
    rng = np.random.default_rng(int(numpy_seed))

    pareto_samples = np.empty((num_samples, n_obs, number_locs.max(), 3))
    for i in range(num_samples):
        for j in range(n_obs):
            pareto_samples[i, j, :int(number_locs[i, j]), :] = ss.genpareto.rvs(
                pareto_shape[i], scale=pareto_scale[i], size=(int(number_locs[i, j]), 3), random_state=rng)

    pareto_samples = jnp.max(pareto_samples, axis=-1)
    pareto_samples = pareto_samples + v_0  # TODO! CHECK

    key, subkey = random.split(key)
    unif_sample = random.uniform(subkey, (num_samples, n_obs, jnp.max(number_locs), 2))
    V1 = (pareto_samples - v_0) * unif_sample[..., 0] * pareto_samples + v_0
    V2 = (pareto_samples - v_0) * unif_sample[..., 1] * pareto_samples + v_0
    V_tmp = np.maximum(V1, V2)
    for i in range(num_samples):
        for j in range(n_obs):
            V_tmp[i, j, int(number_locs[i, j]):] = np.nan
    return jnp.array(V_tmp)


def get_summaries(x):
    # TODO: few options
    # 4 SUMMARIES: number of inclusions, log(min(S)), log(mean(S)) and log(max(S))
    if x.shape[-1] == 1:
        x = x.T  # NOTE: lazy fix for 1-dim case
    num_inclusions = jnp.sum(~jnp.isnan(x), axis=-1)
    ssx = jnp.array([num_inclusions,
                     jnp.log(jnp.nanmin(x, axis=-1)),
                     jnp.log(jnp.nanmean(x, axis=-1)),
                     jnp.log(jnp.nanmax(x, axis=-1))])
    clip_value = 1e+8
    ssx = ssx.at[jnp.where(jnp.isinf(ssx))].set(-10.0)  # NOTE: lazy fix
    ssx = ssx.at[jnp.where(ssx > clip_value)].set(-10.0)  # NOTE: lazy fix
    ssx = jnp.mean(ssx, axis=-1)
    # if not jnp.all(jnp.isfinite(ssx)):
    #     print("WARNING: non-finite summary statistics")
    return ssx.T
