"""Implementation of the univariate g-and-k model."""

import jax.numpy as jnp
import jax.random as random
import numpyro  # type: ignore
import numpyro.distributions as dist  # type: ignore
from jax import lax
from jax.scipy.special import logit
from jax.scipy.stats import norm
from numpyro.infer import MCMC, NUTS  # type: ignore


# @jit
def gnk(z, A, B, g, k, c=0.8):
    """Quantile function for the g-and-k distribution."""
    return A + B * (1 + c * jnp.tanh(g * z / 2)) * (1 + z**2)**k * z


# @jit
def ss_octile(y):
    """Calculate octiles of the input data."""
    octiles = jnp.linspace(12.5, 87.5, 7)
    return jnp.percentile(y, octiles, axis=-1)


def ss_duodecile(y):
    """Calculate octiles of the input data."""
    duodeciles = jnp.linspace(8.33, 91.67, 11)
    return jnp.percentile(y, duodeciles, axis=-1)


def ss_hexadeciles(y):
    """Calculate hexadeciles of the input data."""
    hexadeciles = jnp.linspace(6.25, 93.75, 15)
    return jnp.percentile(y, hexadeciles, axis=-1)


def gnk_density(x, A, B, g, k, c=0.8):
    """Calculate the density of the g-and-k distribution."""
    z = pgk(x, A, B, g, k, c, zscale=True)
    return norm.pdf(z) / gnk_deriv(z, A, B, g, k, c)


def gnk_density_from_z(z, A, B, g, k, c=0.8):
    """Calculate the density of the g-and-k distribution given z."""
    dQdz = gnk_deriv(z, A, B, g, k, c)
    f_x = (norm.pdf(z) ** 2) / dQdz
    return f_x


def get_summaries_batches(key, A, B, g, k, n_obs, n_sims, batch_size, sum_fn=None):
    if sum_fn is None:
        sum_fn = ss_octile
    num_batches = n_sims // batch_size + (n_sims % batch_size != 0)
    all_octiles = []

    for i in range(num_batches):
        sub_key, key = random.split(key)
        batch_size_i = min(batch_size, n_sims - i * batch_size)

        z_batch = random.normal(sub_key, shape=(n_obs, batch_size_i))

        A_batch = A[i * batch_size: i * batch_size + batch_size_i]
        B_batch = B[i * batch_size: i * batch_size + batch_size_i]
        g_batch = g[i * batch_size: i * batch_size + batch_size_i]
        k_batch = k[i * batch_size: i * batch_size + batch_size_i]

        x_batch = gnk(z_batch, A_batch, B_batch, g_batch, k_batch)
        x_batch = x_batch.T

        octiles_batch = sum_fn(x_batch)
        all_octiles.append(octiles_batch)

    return jnp.concatenate(all_octiles, axis=1)


def gnk_deriv(z, A, B, g, k, c):
    """Calculate the derivative of the g-and-k quantile function."""
    z_squared = z**2
    term1 = (1 + z_squared)**k
    term2 = 1 + c * jnp.tanh(g * z / 2)
    term3 = (1 + (2 * k + 1) * z_squared) / (1 + z_squared)
    term4 = c * g * z / (2 * jnp.cosh(g * z / 2)**2)

    term2 = jnp.where(g == 0, 1.0, term2)
    term4 = jnp.where(g == 0, 0.0, term4)

    term3 = jnp.where(jnp.isinf(z_squared), 2 * k + 1, term3)
    term4 = jnp.where(jnp.isinf(z), 0.0, term4)

    return B * term1 * (term2 * term3 + term4)


def pgk(q, A, B, g, k, c=0.8, zscale=False):
    """Inverse of the g-and-k quantile function."""
    def toroot(p):
        return z2gk(p, A, B, g, k, c) - q

    z = bisection_method(toroot, -5, 5, tol=1e-5, max_iter=100)
    return z if zscale else norm.cdf(z)


def z2gk(p, A, B, g, k, c=0.8):
    """G-and-k quantile function."""
    return A + B * ((1 + c * jnp.tanh(g * p / 2)) * ((1 + p**2)**k) * p)


def bisection_method(f, a, b, tol=1e-5, max_iter=100):
    fa = f(a)

    def body_fun(state):
        a, b, fa, _ = state
        c = (a + b) / 2
        fc = f(c)
        con_zero = jnp.isclose(fc, 0, atol=tol)
        con_tol = (b - a) / 2 < tol
        done = jnp.logical_or(con_zero, con_tol)

        update = jnp.sign(fc) * jnp.sign(fa) > 0
        a_new = jnp.where(update, c, a)
        fa_new = jnp.where(update, fc, fa)
        b_new = jnp.where(update, b, c)

        a_final = jnp.where(done, a, a_new)
        b_final = jnp.where(done, b, b_new)
        fa_final = jnp.where(done, fa, fa_new)

        return a_final, b_final, fa_final, done

    init_state = (a, b, fa, False)
    final_state = lax.while_loop(
        lambda state: jnp.logical_not(state[3]),
        body_fun,
        init_state
    )

    return (final_state[0] + final_state[1]) / 2


def sample_var_fn(p, A, B, g, k, n_obs):
    """Calculate the variance of an order statistic."""
    numerator = p * (1 - p)
    gnk_dens = gnk_density(gnk(norm.ppf(p), A, B, g, k), A, B, g, k)
    denominator = n_obs * gnk_dens ** 2
    res = numerator/denominator
    return res


def compute_covariance_matrix(A, B, g, k, quantiles, n_obs, c=0.8):
    """Compute the covariance matrix of the order statistics (quantiles) for the g-and-k distribution."""
    m = len(quantiles)  # Number of quantiles
    cov_matrix = jnp.zeros((m, m))

    z = norm.ppf(quantiles)

    f_x = gnk_density_from_z(z, A, B, g, k, c)

    for i in range(m):
        p_i = quantiles[i]
        f_i = f_x[i]

        var_i = p_i * (1 - p_i) / (n_obs * f_i ** 2)
        cov_matrix = cov_matrix.at[i, i].set(var_i)

        for j in range(i + 1, m):
            p_j = quantiles[j]
            f_j = f_x[j]

            cov_ij = (jnp.minimum(p_i, p_j) - p_i * p_j) / (n_obs * f_i * f_j)
            cov_matrix = cov_matrix.at[i, j].set(cov_ij)
            cov_matrix = cov_matrix.at[j, i].set(cov_ij)  # Symmetric matrix

    return cov_matrix


def gnk_model(obs, n_obs):
    """Model for the g-and-k distribution using Numpyro with a multivariate normal."""
    # Sample parameters
    A = numpyro.sample('A', dist.Uniform(0, 10))
    B = numpyro.sample('B', dist.Uniform(0, 10))
    g = numpyro.sample('g', dist.Uniform(0, 10))
    k = numpyro.sample('k', dist.Uniform(0, 10))

    quantile_length = 1 / (len(obs) + 1)
    quantiles = jnp.linspace(quantile_length, 1 - quantile_length, len(obs))
    z = norm.ppf(quantiles)
    expected_summaries = gnk(z, A, B, g, k)

    cov_matrix = compute_covariance_matrix(A, B, g, k, quantiles, n_obs)

    jitter = 1e-6
    cov_matrix += jitter * jnp.eye(len(obs))

    numpyro.sample('obs', dist.MultivariateNormal(expected_summaries, cov_matrix), obs=obs)


# def gnk_model(obs, n_obs):
#     """Model for the g-and-k distribution using Numpyro."""
#     A = numpyro.sample('A', dist.Uniform(0, 10))
#     B = numpyro.sample('B', dist.Uniform(0, 10))
#     g = numpyro.sample('g', dist.Uniform(0, 10))
#     k = numpyro.sample('k', dist.Uniform(0, 10))

#     # octiles = jnp.linspace(12.5, 87.5, 7) / 100
#     quantile_length = 100*(1/len(obs))
#     quantiles = jnp.linspace(quantile_length, 100-quantile_length, len(obs)) / 100
#     norm_quantiles = norm.ppf(quantiles)
#     expected_summaries = gnk(norm_quantiles, A, B, g, k)

#     y_variance = [sample_var_fn(p, A, B, g, k, n_obs) for p in quantiles]
#     for i in range(len(obs)):
#         numpyro.sample(f'y_{i}',
#                        dist.Normal(expected_summaries[i],
#                                    jnp.sqrt(y_variance[i])),
#                        obs=obs[i])


def gnk_model_narrow_prior(obs, n_obs):
    """Model for the g-and-k distribution using Numpyro."""
    A = numpyro.sample('A', dist.Uniform(0, 5))
    B = numpyro.sample('B', dist.Uniform(0, 5))
    g = numpyro.sample('g', dist.Uniform(0, 5))
    k = numpyro.sample('k', dist.Uniform(0, 5))

    octiles = jnp.linspace(12.5, 87.5, 7) / 100
    norm_quantiles = norm.ppf(octiles)
    expected_summaries = gnk(norm_quantiles, A, B, g, k)

    y_variance = [sample_var_fn(p, A, B, g, k, n_obs) for p in octiles]
    for i in range(len(obs)):
        numpyro.sample(f'y_{i}',
                       dist.Normal(expected_summaries[i],
                                   jnp.sqrt(y_variance[i])),
                       obs=obs[i])


def run_nuts(seed, obs, n_obs, num_samples=10_000, num_warmup=10_000):
    """Run the NUTS sampler."""
    rng_key = random.key(seed)
    kernel = NUTS(gnk_model)
    thinning = 10
    num_chains = 4

    mcmc = MCMC(kernel,
                num_warmup=num_warmup,
                num_samples=num_samples*thinning // num_chains,
                thinning=thinning,
                num_chains=num_chains,
                )

    # NOTE: need to transform initial parameters to unbounded space
    def init_param_to_unbounded(value, num_chains, subkey):
        param_arr = jnp.repeat(logit(jnp.array([value])/10), num_chains)
        noise = random.normal(subkey, (num_chains,)) * 0.05

        return param_arr + noise

    rng_key, *subkeys = random.split(rng_key, 5)

    init_params = {
        'A': init_param_to_unbounded(3.0, num_chains, subkeys[0]),
        'B': init_param_to_unbounded(1.0, num_chains, subkeys[1]),
        'g': init_param_to_unbounded(2.0, num_chains, subkeys[2]),
        'k': init_param_to_unbounded(0.5, num_chains, subkeys[3])
    }

    mcmc.run(rng_key=rng_key,
             init_params=init_params,
             obs=obs,
             n_obs=n_obs)

    return mcmc


def run_nuts_narrow_prior(seed, obs, n_obs, num_samples=10_000, num_warmup=10_000):
    """Run the NUTS sampler."""
    rng_key = random.PRNGKey(seed)
    kernel = NUTS(gnk_model_narrow_prior)
    thinning = 10
    num_chains = 4

    mcmc = MCMC(kernel,
                num_warmup=num_warmup,
                num_samples=num_samples*thinning // num_chains,
                thinning=thinning,
                num_chains=num_chains,
                )

    # NOTE: need to transform initial parameters to unbounded space
    def init_param_to_unbounded(value, num_chains, subkey):
        param_arr = jnp.repeat(logit(jnp.array([value])/5), num_chains)
        noise = random.normal(subkey, (num_chains,)) * 0.05

        return param_arr + noise

    rng_key, *subkeys = random.split(rng_key, 5)

    init_params = {
        'A': init_param_to_unbounded(3.0, num_chains, subkeys[0]),
        'B': init_param_to_unbounded(1.0, num_chains, subkeys[1]),
        'g': init_param_to_unbounded(2.0, num_chains, subkeys[2]),
        'k': init_param_to_unbounded(0.5, num_chains, subkeys[3])
    }

    mcmc.run(rng_key=rng_key,
             init_params=init_params,
             obs=obs,
             n_obs=n_obs)

    return mcmc


def ss_robust(y):
    """Compute robust summary statistics as in Drovandi 2011 #TODO."""
    ss_A = _get_ss_A(y)
    ss_B = _get_ss_B(y)
    ss_g = _get_ss_g(y)
    ss_k = _get_ss_k(y)

    # Combine the summary statistics, (batch should be first dim)
    ss_robust = jnp.concatenate([ss_A[:, None], ss_B[:, None],
                                 ss_g[:, None], ss_k[:, None]], axis=1)
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
    ss_B = jnp.where(ss_B == 0, jnp.finfo(jnp.float32).eps, ss_B)
    return ss_B[:, None]


def _get_ss_g(y):
    """Compute a skewness-like summary statistic."""
    L1, L2, L3 = jnp.percentile(y, jnp.array([25, 50, 75]), axis=1)
    ss_B = _get_ss_B(y).flatten()
    ss_g = (L3 + L1 - 2 * L2) / ss_B
    return ss_g[:, None]


def _get_ss_k(y):
    """Compute a kurtosis-like summary statistic."""
    E1, E3, E5, E7 = jnp.percentile(y, jnp.array([12.5, 37.5, 62.5, 87.5]),
                                    axis=1)
    ss_B = _get_ss_B(y).flatten()
    ss_k = (E7 - E5 + E3 - E1) / ss_B
    return ss_k[:, None]
