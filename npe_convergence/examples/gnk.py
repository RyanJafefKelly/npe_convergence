"""gnk model."""

import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist

import numpy as np

from jax import vmap, custom_vjp, grad

from numpyro.infer import MCMC, NUTS, ESS, AIES
from jax.scipy.stats import norm
from scipy.optimize import root_scalar
from jax.scipy.optimize import minimize
import arviz as az
import matplotlib.pyplot as plt
from jax.scipy.special import logit, expit

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


def ss_octile(y):
    octiles = jnp.linspace(12.5, 87.5, 7)
    return jnp.percentile(y, octiles, axis=-1)


def gnk_density(x, A, B, g, k, c=0.8):
    z = pgk(x, A, B, g, k, c, zscale=True)
    return norm.pdf(z) / gnk_deriv(z, A, B, g, k, c)


def gnk_deriv(z, A, B, g, k, c, getR=False):
    # TODO: REMOVE getR?
    z_squared = z**2
    term1 = jnp.where(getR, 1.0, (1 + z_squared)**k)
    term2 = 1 + c * jnp.tanh(g * z / 2)
    term3 = (1 + (2 * k + 1) * z_squared) / (1 + z_squared)
    term4 = c * g * z / (2 * jnp.cosh(g * z / 2)**2)

    gzero = (g == 0)
    term2 = jnp.where(gzero, 1.0, term2)
    term4 = jnp.where(gzero, 0.0, term4)

    zbig = jnp.isinf(z_squared)
    # if not getR:  # TODO: MAY NEED TO CONSIDER?
    #     if zbig.any():
    #         term1 = jnp.abs(z)**(2 * k)
        # term1 = jnp.where(zbig, jnp.abs(z[zbig])**(2 * k), term1)
    term3 = jnp.where(zbig, 2 * k + 1, term3)
    term4 = jnp.where(jnp.isinf(z), 0.0, term4)

    if getR:
        return term2 * term3 + term4
    else:
        return B * term1 * (term2 * term3 + term4)


def pgk(q, A, B, g, k, c=0.8, zscale=False):
    return pgk_scalar(q, A, B, g, k, c, zscale)


def z2gk(p, A, B, g, k, c=0.8):
    res = A + B * ((1 + c * jnp.tanh(g * p / 2)) * ((1 + p**2)**k) * p)
    return res


def bisection_method(f, a, b, tol=1e-5, max_iter=100):
    fa = f(a)
    c = a
    for _ in range(max_iter):
        c = (a + b) / 2
        fc = f(c)
        con_zero = jnp.isclose(fc, 0, atol=tol)
        con_tol = (b - a) / 2 < tol
        done = jnp.logical_or(con_zero, con_tol)

        update = jnp.sign(fc) * jnp.sign(fa) > 0
        a = jnp.where(update, c, a)
        fa = jnp.where(update, fc, fa)
        b = jnp.where(update, b, c)

        a = jnp.where(done, a, a)
        b = jnp.where(done, b, b)
        c = jnp.where(done, c, c)

    return c


def pgk_scalar(q, A, B, g, k, c=0.8, zscale=False):
    def toroot(p):
        res = z2gk(p, A, B, g, k, c) - q
        return res
    z = bisection_method(toroot, -5, 5, tol=1e-5, max_iter=100)

    if zscale:
        return z
    else:
        return norm.cdf(z)


def sample_var_fn(p, A, B, g, k, n_obs):
    res = (p*(1-p))/(n_obs * gnk_density(gnk(norm.ppf(p), A, B, g, k),  A, B, g, k) ** 2)
    return res


def gnk_model(obs, n_obs):
    """Model for the g-and-k distribution using Numpyro."""
    A = numpyro.sample('A', dist.Uniform(0, 10))
    B = numpyro.sample('B', dist.Uniform(0, 10))
    g = numpyro.sample('g', dist.Uniform(0, 10))
    k = numpyro.sample('k', dist.Uniform(0, 10))

    octiles = jnp.linspace(12.5, 87.5, 7) / 100
    norm_quantiles = norm.ppf(octiles)
    expected_summaries = gnk(norm_quantiles, A, B, g, k)

    y_variance = [sample_var_fn(p, A, B, g, k, n_obs) for p in octiles]
    for i in range(7):
        numpyro.sample(f'y_{i}', dist.Normal(expected_summaries[i], jnp.sqrt(y_variance[i])), obs=obs[i])


def run_nuts(seed, obs, n_obs, num_samples=10_000, num_warmup=10_000):
    """Run the NUTS sampler."""
    rng_key = random.PRNGKey(seed)
    kernel = NUTS(gnk_model)
    aies_kernel = AIES(gnk_model, moves={AIES.DEMove() : 0.5,
                           AIES.StretchMove() : 0.5},
                    #    init_strategy='init_to_value'
                       )
    thinning = 1
    # num_chains = 2 * len(obs)
    num_chains = 4

    A = jnp.array([3.0])
    B = jnp.array([1.0])
    B2 = jnp.array([0.39])
    g = jnp.array([2.0])
    k = jnp.array([0.5])
    k2 = jnp.array([9.9])

    norm_quantiles = norm.ppf(jnp.array([0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]))
    expected_summaries = gnk(norm_quantiles, A, B, g, k)  # TODO: UGLY CODE
    expected_summaries2 = gnk(norm_quantiles, A, B2, g, k2)  # TODO: UGLY CODE

    # Sample y according to the quantile function
    octiles = jnp.linspace(12.5, 87.5, 7) / 100
    y_variance = [sample_var_fn(p, A, B, g, k, n_obs) for p in octiles]
    y_variance2 = [sample_var_fn(p, A, B2, g, k2, n_obs) for p in octiles]
    y_stdev = [jnp.sqrt(var) for var in y_variance]
    y_stdev2 = [jnp.sqrt(var) for var in y_variance2]

    diff = obs - expected_summaries
    diff_norm = [diff[i] / y_stdev[i] for i in range(7)]
    diff2 = obs - expected_summaries2
    diff_norm2 = [diff2[i] / y_stdev2[i] for i in range(7)]

    mcmc = MCMC(kernel,
                num_warmup=num_warmup,
                num_samples=num_samples*thinning // num_chains,
                thinning=thinning,
                num_chains=num_chains,
                # chain_method='vectorized'
                )
    init_params = {
        'A': jnp.repeat(logit(jnp.array([3.0])/10), num_chains),
        'B': jnp.repeat(logit(jnp.array([1.0])/10), num_chains),
        'g': jnp.repeat(logit(jnp.array([2.0])/10), num_chains),
        'k': jnp.repeat(logit(jnp.array([0.5])/10), num_chains)
    }
    mcmc.run(rng_key=rng_key,
    init_params=init_params,  # NOTE: just cheat, want to be sampling exactly anyway
    obs=obs, n_obs=n_obs)
    mcmc.print_summary()
    inference_data = az.from_numpyro(mcmc)
    dirname = ""
    az.plot_trace(inference_data, compact=False)
    plt.savefig(f"{dirname}traceplots.png")
    plt.close()
    az.plot_ess(inference_data, kind="evolution")
    plt.savefig(f"{dirname}ess_plots.png")
    plt.close()
    az.plot_autocorr(inference_data)
    plt.savefig(f"{dirname}autocorr.png")
    plt.close()

    return mcmc.get_samples()
