"""MA(2) model"""

import jax.numpy as jnp
import jax.random as random
import numpyro
from numpyro import distributions as dist


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
