"""MA(2) model"""

import jax.numpy as jnp
import jax.random as random
import numpyro  # type: ignore
from numpyro import distributions as dist


def autocov_exact(thetas, k, ma_order=1):
    # NOTE: ASSUMPTION - white noise has constant variance one
    res = 0
    # NOTE: assume passing in thetas care about, add in 1 for i=0 case
    thetas = jnp.concatenate((jnp.array([1.0]), thetas))
    for i in range(ma_order + 1):
        if i - k >= 0:
            res += thetas[i] * thetas[i - k]
    return res


def sample_autocov_variance(thetas, k, n_obs, ma_order=1):
    res = 0
    for i in range(-ma_order, ma_order + 1):
        tmp_k = i if i >= 0 else -i
        res += autocov_exact(thetas, tmp_k, ma_order) ** 2
        res = jnp.where(
            jnp.abs(k + i) <= ma_order,
            res + autocov_exact(thetas, k + i, ma_order) * autocov_exact(thetas, k - i, ma_order),
            res
        )

    res = res / n_obs

    return res


def MA2(t1, t2, n_obs=100, batch_size=1, key=None):
    # TODO: could make faster (e.g. use scan)
    # NOTE: adapted from ELFI
    # Make inputs 2d arrays for broadcasting with w
    t1 = jnp.array(t1).reshape((-1, 1))
    t2 = jnp.array(t2).reshape((-1, 1))

    # i.i.d. sequence ~ N(0,1)
    w = random.normal(key, (batch_size, n_obs + 2))
    x = w[:, 2:] + t1 * w[:, 1:-1] + t2 * w[:, :-2]
    return x


def autocov(x, lag=1):
    """Return the autocovariance.

    Assumes a (weak) univariate stationary process with mean 0.
    Realizations are in rows.

    Parameters
    ----------
    x : np.array of size (n, m)
    lag : int, optional

    Returns
    -------
    C : np.array of size (n,)

    """
    x = jnp.atleast_2d(x)
    C = jnp.mean(x[:, lag:] * x[:, :-lag], axis=1)
    return C


def get_summaries(x):
    """Compute summaries: variance and autocovariances at lags 1 and 2.

    Parameters
    ----------
    x : jnp.array of shape (n_samples, n_obs)

    Returns
    -------
    summaries : jnp.array of shape (n_samples, 3)
    """
    var_x = jnp.var(x, axis=1)
    acov1 = autocov(x, lag=1)
    acov2 = autocov(x, lag=2)
    summaries = jnp.stack((var_x, acov1, acov2), axis=1)
    return summaries


def get_summaries_batches(key, t1, t2, n_obs, n_sims, batch_size):
    """Process simulations in batches to save memory.

    Parameters
    ----------
    key : PRNGKey
        JAX random number generator key.
    t1 : jnp.array
        Array of MA(2) parameter t1 values.
    t2 : jnp.array
        Array of MA(2) parameter t2 values.
    n_obs : int
        Number of observations per simulation.
    n_sims : int
        Total number of simulations.
    batch_size : int
        Number of simulations per batch.

    Returns
    -------
    all_summaries : jnp.array of shape (n_sims, 3)
        Summaries for all simulations.
    """
    num_batches = n_sims // batch_size + (n_sims % batch_size != 0)
    all_summaries = []

    for i in range(num_batches):
        # Update key for randomness
        sub_key, key = random.split(key)
        batch_size_i = min(batch_size, n_sims - i * batch_size)

        # Extract batch parameters
        t1_batch = t1[i * batch_size: i * batch_size + batch_size_i]
        t2_batch = t2[i * batch_size: i * batch_size + batch_size_i]

        # Run simulations for the batch
        sim_data_batch = MA2(t1_batch, t2_batch, n_obs=n_obs, batch_size=batch_size_i, key=sub_key)

        # Compute summaries for the batch
        sim_summ_data_batch = get_summaries(sim_data_batch)

        # Collect summaries
        all_summaries.append(sim_summ_data_batch)

    # Concatenate all summaries
    return jnp.concatenate(all_summaries, axis=0)


class CustomPrior_t1:
    """Define prior for t1 in range [-a, a], as in Marin et al., 2012."""

    def rvs(b, size=(1,), key=None):
        """Get random variates."""
        # u = numpyro.sample('u', dist.Uniform(0, 1), sample_shape=size, rng_key=key)
        u = random.uniform(key, shape=size)
        t1 = jnp.where(u < 0.5, jnp.sqrt(2. * u) * b - b, -jnp.sqrt(2. * (1. - u)) * b + b)
        return t1

    def pdf(x, b):
        """Return density at `x`."""
        p = 1. / b - jnp.abs(x) / (b * b)
        # set values outside of [-b, b] to zero
        p = jnp.where((x >= -b) & (x <= b), p, 0.)
        return p


class CustomPrior_t2:
    """Define prior for t2 conditionally on t1 in range [-a, a], as in Marin et al., 2012."""

    def rvs(t1, a, size=(1,), key=None):
        """Get random variates."""
        locs = jnp.maximum(-a - t1, -a + t1)
        scales = a - locs
        # u = dist.Uniform(0, 1).sample(sample_shape=size, key=key)
        u = random.uniform(key, shape=size)

        t2 = locs + scales * u
        return t2

    def pdf(x, t1, a):
        """Return density at `x`."""
        locs = jnp.maximum(-a - t1, -a + t1)
        scales = a - locs
        return (x >= locs) & (x <= locs + scales) * (1 / jnp.where(scales > 0, scales, 1))


def numpyro_model(obs, a=2, n_obs=100):
    ma_order = 2
    t1 = numpyro.sample('t1', dist.Uniform(-a, a))
    locs = jnp.maximum(-a - t1, -a + t1)
    scales = a - locs
    t2 = numpyro.sample('t2', dist.Uniform(locs, locs + scales))
    thetas = jnp.array([t1, t2])
    y_variance = numpyro.deterministic(
        "y_variance",
        jnp.array([sample_autocov_variance(thetas, k, n_obs, ma_order)
                   for k in range(0, ma_order+1)])
    )

    mean = numpyro.deterministic(
        "mean",
        jnp.array([autocov_exact(thetas, i, ma_order) for i in range(0, ma_order+1)])
    )

    stdev = numpyro.deterministic("stdev", jnp.sqrt(y_variance))

    numpyro.sample('obs', dist.Normal(mean, stdev), obs=obs)
