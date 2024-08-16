import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS


def gnk(z, A, B, g, k, c=0.8):
    """Quantile function for the g-and-k distribution."""
    return A + B * (1 + c * (1 - jnp.exp(-g * z)) / (1 + jnp.exp(-g * z))) * (1 + z**2)**k * z


def ss_robust(y):
    """Compute robust summary statistics as in Drovandi 2011 #TODO."""
    ss_A = _get_ss_A(y)
    ss_B = _get_ss_B(y)
    ss_g = _get_ss_g(y)
    ss_k = _get_ss_k(y)

    # Combine the summary statistics, (batch should be first dim)
    ss_robust = jnp.concatenate([ss_A[:, None], ss_B[:, None], ss_g[:, None], ss_k[:, None]], axis=1)
    return jnp.squeeze(ss_robust)


def _get_ss_A(y):
    """Compute the median as a summary statistic."""
    L2 = jnp.percentile(y, 50, axis=1)
    ss_A = L2
    return ss_A[:, None]


def _get_ss_B(y):
    """Compute the interquartile range."""
    L1, L3 = jnp.percentile(y, jnp.array([25, 75]), axis=1)
    ss_B = L3 - L1
    ss_B = jnp.where(ss_B == 0, jnp.finfo(jnp.float32).eps, ss_B)  # Avoid division by zero.
    return ss_B[:, None]


def _get_ss_g(y):
    """Compute a skewness-like summary statistic."""
    L1, L2, L3 = jnp.percentile(y, jnp.array([25, 50, 75]), axis=1)
    ss_B = _get_ss_B(y).flatten()  # Flatten since we need to use it for division.
    ss_g = (L3 + L1 - 2 * L2) / ss_B
    return ss_g[:, None]

def _get_ss_k(y):
    """Compute a kurtosis-like summary statistic."""
    E1, E3, E5, E7 = jnp.percentile(y, jnp.array([12.5, 37.5, 62.5, 87.5]), axis=1)
    ss_B = _get_ss_B(y).flatten()  # Flatten since we need to use it for division.
    ss_k = (E7 - E5 + E3 - E1) / ss_B
    return ss_k[:, None]


def gnk_model(obs, n_obs):
    """Model for the g-and-k distribution using Numpyro."""
    A = numpyro.sample('A', dist.Uniform(0, 10))
    B = numpyro.sample('B', dist.Uniform(0, 10))
    g = numpyro.sample('g', dist.Uniform(0, 10))
    k = numpyro.sample('k', dist.Uniform(0, 10))

    rng_key = numpyro.prng_key()
    z = random.normal(rng_key, shape=(n_obs,))

    # Compute the quantile function
    y = gnk(z, A, B, g, k)
    y = jnp.atleast_2d(y)
    y_summ = ss_robust(y)
    y_summ = jnp.squeeze(y_summ)

    # Sample y according to the quantile function
    y_variance = 1/jnp.sqrt(n_obs)  # TODO: Is this correct? appeal to CLT?
    numpyro.sample('y', dist.Normal(y_summ, y_variance), obs=obs)  # Can we just assume normality here? Seems correct from prior pred...


def run_nuts(seed, obs, n_obs, num_samples=10_000, num_warmup=2_000):
    """Run the NUTS sampler."""
    rng_key = random.PRNGKey(seed)
    kernel = NUTS(gnk_model)
    thinning = 10
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples*thinning, thinning=thinning)
    # init_params = {'A': 3.0, 'B': 1.0, 'g': 2.0, 'k': 0.5}
    mcmc.run(rng_key=rng_key,
    # init_params=init_params,
    obs=obs, n_obs=n_obs)
    mcmc.print_summary()
    return mcmc.get_samples()
