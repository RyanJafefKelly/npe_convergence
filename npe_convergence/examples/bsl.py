"""Small Bayesian synthetic likelihood helpers."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import jax.random as random
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import expit, logit

from npe_convergence.examples.gnk import gnk, ss_octile


def theta_to_eta(theta: jax.Array) -> jax.Array:
    """Map GNK parameters from (0, 10)^4 to R^4."""
    return logit(theta / 10.0)


def eta_to_theta(eta: jax.Array) -> jax.Array:
    """Map GNK parameters from R^4 to (0, 10)^4."""
    return expit(eta) * 10.0


def log_uniform_prior_jacobian_eta(eta: jax.Array) -> jax.Array:
    """Log Jacobian term for a Uniform(0, 10)^4 prior in eta coordinates."""
    return jnp.sum(-jax.nn.softplus(-eta) - jax.nn.softplus(eta))


def _chol_ok(chol: jax.Array) -> jax.Array:
    diag = jnp.diag(chol)
    return jnp.all(jnp.isfinite(chol)) & jnp.all(diag > 0)


def _mvn_logpdf_from_chol(x: jax.Array, mean: jax.Array, chol: jax.Array) -> jax.Array:
    diff = x - mean
    solved = solve_triangular(chol, diff, lower=True)
    quad = jnp.dot(solved, solved)
    log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))
    dim = x.shape[0]
    return -0.5 * (dim * jnp.log(2.0 * jnp.pi) + log_det + quad)


def stable_mvn_logpdf(
    x: jax.Array,
    mean: jax.Array,
    cov: jax.Array,
    *,
    jitter: float = 1e-6,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate an MVN logpdf, adding jitter only when the raw Cholesky fails."""
    dim = x.shape[0]
    eye = jnp.eye(dim, dtype=cov.dtype)
    cov_is_finite = jnp.all(jnp.isfinite(cov)) & jnp.all(jnp.isfinite(mean))

    chol = jnp.linalg.cholesky(cov)
    chol_jitter = jnp.linalg.cholesky(cov + jitter * eye)
    raw_ok = cov_is_finite & _chol_ok(chol)
    jitter_ok = cov_is_finite & _chol_ok(chol_jitter)

    selected_chol = jnp.where(raw_ok, chol, chol_jitter)
    logpdf = _mvn_logpdf_from_chol(x, mean, selected_chol)
    rank_deficient = ~raw_ok & ~jitter_ok
    logpdf = jnp.where(rank_deficient, -jnp.inf, logpdf)
    jitter_used = ~raw_ok & jitter_ok
    return logpdf, jitter_used, rank_deficient


@partial(jax.jit, static_argnames=("n_obs", "m"))
def gnk_bsl_loglikelihood(
    key: jax.Array,
    theta: jax.Array,
    x_obs: jax.Array,
    *,
    n_obs: int,
    m: int,
    jitter: float = 1e-6,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Synthetic Gaussian log likelihood for GNK octile summaries.

    This is the plain Gaussian synthetic likelihood. The Price et al. (2018)
    unbiased correction is not applied.
    """
    z = random.normal(key, shape=(m, n_obs))
    summaries = ss_octile(gnk(z, *theta)).T
    mean = jnp.mean(summaries, axis=0)
    centered = summaries - mean
    cov = centered.T @ centered / jnp.maximum(m - 1, 1)
    return stable_mvn_logpdf(x_obs, mean, cov, jitter=jitter)


@partial(jax.jit, static_argnames=("n_obs", "m"))
def gnk_bsl_logtarget_eta(
    key: jax.Array,
    eta: jax.Array,
    x_obs: jax.Array,
    *,
    n_obs: int,
    m: int,
    jitter: float = 1e-6,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """BSL log target in eta coordinates, up to an additive constant."""
    theta = eta_to_theta(eta)
    loglik, jitter_used, rank_deficient = gnk_bsl_loglikelihood(
        key,
        theta,
        x_obs,
        n_obs=n_obs,
        m=m,
        jitter=jitter,
    )
    logtarget = loglik + log_uniform_prior_jacobian_eta(eta)
    return logtarget, jitter_used, rank_deficient
