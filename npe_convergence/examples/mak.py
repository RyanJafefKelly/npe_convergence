"""Generic moving average of order k model."""

import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro
from numpyro import distributions as dist
from numpyro.distributions import Distribution
from numpyro.distributions.constraints import real
import numpyro.distributions.constraints as constraints
import numpyro.distributions.util as dist_util


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


def MAK(thetas, n_obs=100, batch_size=1, key=None):
    thetas = jnp.atleast_2d(thetas)
    batch_size, ma_order = thetas.shape  # assume 2d array
    w = random.normal(key, (batch_size, n_obs + ma_order))
    # TODO! NEED TO THINK HOW THETAS COMING IN HERE
    
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
    ma_order = len(obs.ravel()) - 1  # TODO? better...assumes obs has var
    
    # thetas = numpyro.sample("thetas", MAIdentifiablePrior(ma_order))
    thetas = numpyro.sample('thetas', dist.Uniform(-a, a).expand([ma_order]))
    y_variance = [sample_autocov_variance(thetas, k, n_obs, len(obs.ravel())) for k in range(len(obs.ravel()))]

    for i in range(len(y_variance)):
        mean = autocov_exact(thetas, i, len(thetas))
        stdev = jnp.sqrt(y_variance[i])
        numpyro.sample(f'obs_{i}', dist.Normal(mean, stdev), obs=obs.ravel()[i])


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
        while len(samples) < np.prod(sample_shape):
            theta = generate_sample(self.k, self.intervals)
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

def generate_sample(k, intervals=None):
    """Generate a sample of theta parameters for an MA(k) model."""
    # Generate random theta values, for example, in the range [-2, 2]
    if intervals is None:
        intervals = [1] * k
    return np.array([np.random.uniform(-a, a) for a in intervals])


def is_valid_sample(theta):
    """Check if the sample is valid (all roots outside the unit circle)."""
    # coeffs =[1] + [-t for t in theta]  #TODO -t or t ?? t matches prior for ELFI, but -t would be my pick mathematically?
    theta = theta.ravel()
    coeffs =  [t for t in theta] + [1]
    # print("coeff: ", coeffs)
    # roots = poly.Polynomial(coeffs).roots()
    roots = jnp.roots(jnp.array(coeffs), strip_zeros=False)
    return jnp.all(jnp.abs(roots) > 1)

def generate_valid_samples(k, intervals=None, num_samples=1000):
    # TODO! KEY
    """Generate valid samples for an MA(k) model using rejection sampling."""
    valid_samples = []
    count = 0
    while len(valid_samples) < num_samples:
        theta = generate_sample(k, intervals=intervals)
        count += 1
        if is_valid_sample(theta):
            valid_samples.append(theta)
    return np.array(valid_samples)


def log_prob(value, k, a):
    # Compute the log probability assuming the sample is valid
    log_prob_uniform = -k * jnp.log(2 * a)

    # Check sample validity using the is_valid_sample method
    valid = is_valid_sample(value)

    # Use jnp.where to conditionally assign log probability
    return jnp.where(valid, log_prob_uniform, -jnp.inf)
