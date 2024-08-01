"""Bortot et al. (2007)"""

import jax.numpy as jnp
# import jax.scipy.stats as stats
import scipy.stats as ss
import jax.random as random
from jax.scipy.special import logit, expit


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
    return jnp.column_stack([poisson_rate_samples, pareto_scale_samples, pareto_shape_samples])

def stereological(key, poisson_rate, pareto_scale, pareto_shape, num_samples):  # TODO: useful param names
    # TODO: homogenous Poisson
    # In 1D real-line, time between arrivals follows exponential distribution
    v_0 = 5  # CONSTANT
    key, subkey = random.split(key)
    number_locs = random.poisson(subkey, poisson_rate)
    locations = None
    # TODO: should avoid reshapes ...
    pareto_samples = ss.genpareto.rvs(pareto_shape.reshape((-1, 1, 1)),
                                      scale=pareto_scale.reshape((-1, 1, 1)),
                                      size=(num_samples, int(jnp.max(number_locs)), 3))
    pareto_samples = jnp.max(pareto_samples, axis=2)
    pareto_samples = pareto_samples + v_0  # TODO! CHECK

    key, subkey = random.split(key)
    unif_sample = random.uniform(subkey, (num_samples, jnp.max(number_locs), 2))
    V1 = (pareto_samples - v_0) * unif_sample[..., 0] * pareto_samples + v_0
    V2 = (pareto_samples - v_0) * unif_sample[..., 1] * pareto_samples + v_0
    V_tmp = jnp.maximum(V1, V2)
    if num_samples > 1:
        V = jnp.where(jnp.arange(jnp.max(number_locs)) < number_locs[:, None], V_tmp, jnp.nan) # TODO!
    else:
        V = V_tmp
    
    # V = jnp.sort(V, axis=1)
    # V = jnp.where(V < v_0, jnp.nan, V)
    return V

def get_summaries(x):
    # TODO: few options
    # 4 SUMMARIES: number of inclusions, log(min(S)), log(mean(S)) and log(max(S))
    if x.shape[-1] == 1:
        x = x.T  # NOTE: lazy fix for 1-dim case
    num_inclusions = jnp.sum(~jnp.isnan(x), axis=1)
    ssx = jnp.array([num_inclusions,
                     jnp.log(jnp.nanmin(x, axis=1)),
                     jnp.log(jnp.nanmean(x, axis=1)),
                     jnp.log(jnp.nanmax(x, axis=1))])
    return ssx.T
