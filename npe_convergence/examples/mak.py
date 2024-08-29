"""Moving average of order k model."""

import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro  # type: ignore
from numpyro import distributions as dist
from numpyro.distributions import Distribution  # type: ignore
import numpyro.distributions.constraints as constraints  # type: ignore
from jaxtyping import Array


def autocov_exact(thetas: Array,
                  k: int,
                  ma_order: int = 1
                  ) -> Array:
    """_summary_

    Args:
        thetas (jnp.ndarray): thetas for the MA model
        k (int): autocovariance lag
        ma_order (int, optional): lag of the whole MA model

    Returns:
        jnp.ndarray: exact autocovariance at lag k
    """
    res = jnp.array(0.0)
    # NOTE: assume passing in thetas care about, add in 1 for i=0 case
    thetas = jnp.concatenate((jnp.array([1.0]), thetas))
    for i in range(ma_order + 1):
        if i - k >= 0:
            res += thetas[i] * thetas[i - k]
    return res


def sample_autocov_variance(thetas: Array,
                            k: int,
                            n_obs: int,
                            ma_order: int = 1
                            ) -> Array:
    """Compute the (large sample) autocovariance variance.

    Args:
        thetas (Array): thetas for the MA model
        k (int): autocovariance lag
        n_obs (int): number of observations
        ma_order (int, optional): lag of the whole MA model. Defaults to 1.

    Returns:
        Array: sample autocovariance variance
    """
    res = jnp.array(0.0)
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


def autocov(x: jnp.ndarray,
            lag: int = 1
            ) -> jnp.ndarray:
    """Return the autocovariance.

    Assumes a (weak) univariate stationary process with mean 0.
    Realizations are in rows.

    Parameters
    ----------
    x : jnp.array of size (n, m)
    lag : int, optional

    Returns
    -------
    C : np.array of size (n,)

    """
    x = jnp.atleast_2d(x)
    C = jnp.mean(x[:, lag:] * x[:, :-lag], axis=1)
    return C


def MAK(key: Array,
        thetas: Array,
        n_obs: int = 100,
        batch_size: int = 1
        ) -> Array:
    """Simulate a moving average of order k model.

    Args:
        key (Array): PRNG key
        thetas (Array): thetas (of length k) for the MA model
        n_obs (int, optional):  Defaults to 100.
        batch_size (int, optional):  Defaults to 1.

    Returns:
        Array: time series data
    """
    thetas = jnp.atleast_2d(thetas)
    batch_size, ma_order = thetas.shape  # assume 2d array
    w = random.normal(key, (batch_size, n_obs + ma_order))
    x = w[:, ma_order:]
    for i in range(ma_order):
        x += thetas[:, i].reshape((-1, 1)) * w[:, ma_order - i - 1: -i - 1]

    return x


def get_summaries(sim_data, ma_order):
    batch_size = sim_data.shape[0]
    sim_summ_data = np.empty((batch_size, ma_order + 1))
    sim_summ_data[:, 0] = np.var(sim_data, axis=1)
    for i in range(1, ma_order + 1):
        sim_summ_data[:, i] = autocov(sim_data, lag=i)
    return sim_summ_data


def numpyro_model(obs, a=1, n_obs=100):
    summary_length = len(obs.ravel())
    ma_order = summary_length - 1
    # thetas = numpyro.sample("thetas", MAIdentifiablePrior(ma_order))
    thetas = numpyro.sample('thetas', dist.Uniform(-a, a).expand([ma_order]))

    y_variance = numpyro.deterministic(
        "y_variance",
        jnp.array([sample_autocov_variance(thetas, k, n_obs, ma_order)
                   for k in range(ma_order + 1)])
    )

    mean = numpyro.deterministic(
        "mean",
        jnp.array([autocov_exact(thetas, i, ma_order) for i in range(ma_order + 1)])
    )

    stdev = numpyro.deterministic("stdev", jnp.sqrt(y_variance))

    numpyro.sample('obs', dist.Normal(mean, stdev), obs=obs)


class MAIdentifiablePrior(Distribution):
    support = constraints.real_vector

    def __init__(self, k, a=1.0, intervals=None, validate_args=None):
        self.k = k
        self.a = a
        self.intervals = intervals
        batch_shape, event_shape = (), (k,)
        super(MAIdentifiablePrior, self).__init__(batch_shape=batch_shape, event_shape=event_shape, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        samples = []
        print('key: ', key)
        while len(samples) < np.prod(sample_shape):
            print('c')
            theta = generate_sample(key, self.k, self.intervals)
            if is_valid_sample(theta):
                samples.append(theta)
        return jnp.array(samples).reshape(sample_shape + (self.k,))

    def log_prob(self, value):
        # Compute the log probability assuming the sample is valid
        log_prob_uniform = -self.k * jnp.log(2 * self.a)
        
        # Check sample validity using the is_valid_sample method
        valid = is_valid_sample(value)
        
        # Use jnp.where to conditionally assign log probability
        return jnp.where(valid, log_prob_uniform, -jnp.inf)


def generate_sample(key, k, intervals=None):
    """Generate a sample of theta parameters for an MA(k) model."""
    # Generate random theta values, for example, in the range [-2, 2]
    if intervals is None:
        intervals = [1] * k
    keys = random.split(key, len(intervals))
    samples = jnp.array([random.uniform(k, minval=-a, maxval=a) for k, a in zip(keys, intervals)])
    return samples


def is_valid_sample(theta):
    """Check if the sample is valid (all roots outside the unit circle)."""
    # coeffs =[1] + [-t for t in theta]  #TODO -t or t ?? t matches prior for ELFI, but -t would be my pick mathematically?
    theta = theta.ravel()
    coeffs = [t for t in theta] + [1]
    # print("coeff: ", coeffs)
    # roots = poly.Polynomial(coeffs).roots()
    roots = jnp.roots(jnp.array(coeffs), strip_zeros=False)
    return jnp.all(jnp.abs(roots) > 1)


def generate_valid_samples(key, k, intervals=None, num_samples=1000):
    """Generate valid samples for an MA(k) model using rejection sampling."""
    valid_samples = []
    count = 0
    while len(valid_samples) < num_samples:
        key, sub_key = random.split(key)
        theta = generate_sample(sub_key, k, intervals=intervals)
        count += 1
        if is_valid_sample(theta):
            valid_samples.append(theta)
    return np.array(valid_samples)
