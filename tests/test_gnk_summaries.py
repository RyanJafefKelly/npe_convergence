import jax.numpy as jnp

from npe_convergence.examples.gnk import (
    ss_duodecile,
    ss_even_quantiles,
    ss_hexadeciles,
    ss_octile,
    ss_sextile,
    ss_vigintile,
)


def test_even_quantile_summary_dims() -> None:
    y = jnp.arange(1000.0)
    helpers = [
        (5, ss_sextile),
        (7, ss_octile),
        (11, ss_duodecile),
        (15, ss_hexadeciles),
        (19, ss_vigintile),
    ]
    for d_s, helper in helpers:
        assert helper(y).shape == (d_s,)
        assert ss_even_quantiles(y, d_s).shape == (d_s,)


def test_even_quantile_batch_shape() -> None:
    y = jnp.arange(2000.0).reshape(2, 1000)
    assert ss_sextile(y).shape == (5, 2)
    assert ss_vigintile(y).shape == (19, 2)
