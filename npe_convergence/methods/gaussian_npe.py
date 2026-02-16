# ruff: noqa: F722
"""Conditional Gaussian approximation for neural posterior estimation.

Replaces the normalising flow in standard NPE with a neural network that
outputs the mean and Cholesky factor of a multivariate Gaussian posterior
approximation, conditioned on summary statistics. This is the k=1 case of
the Gaussian mixture of experts class referenced in the paper (Appendix A).

The loss is the forward KL (equivalently, Gaussian NLL) over simulated
(theta, S) pairs — identical to the standard NPE objective but with
Q restricted to the family of multivariate Gaussians.
"""

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jaxtyping import Array, Float, PRNGKeyArray


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class ConditionalGaussianNPE(eqx.Module):
    """MLP mapping summary statistics to Gaussian posterior parameters.

    Architecture: shared hidden layers → two heads (mean, Cholesky factor).
    The Cholesky diagonal is enforced positive via softplus + floor.
    """

    _shared: list
    _mu_head: eqx.nn.Linear
    _chol_head: eqx.nn.Linear
    d_theta: int = eqx.field(static=True)

    def __init__(
        self,
        d_summary: int,
        d_theta: int,
        hidden_dims: tuple[int, ...] = (128, 128),
        *,
        key: PRNGKeyArray,
    ):
        n_chol = d_theta * (d_theta + 1) // 2
        keys = jr.split(key, len(hidden_dims) + 2)

        layers: list[eqx.nn.Linear] = []
        d_in = d_summary
        for i, d_out in enumerate(hidden_dims):
            layers.append(eqx.nn.Linear(d_in, d_out, key=keys[i]))
            d_in = d_out

        self._shared = layers
        self._mu_head = eqx.nn.Linear(d_in, d_theta, key=keys[-2])
        self._chol_head = eqx.nn.Linear(d_in, n_chol, key=keys[-1])
        self.d_theta = d_theta

    def __call__(
        self, s: Float[Array, " d_s"]
    ) -> tuple[Float[Array, " d_theta"], Float[Array, " d_theta d_theta"]]:
        """Forward pass: S → (mu, L) where Sigma = LL^T."""
        h = s
        for layer in self._shared:
            h = jax.nn.relu(layer(h))
        mu = self._mu_head(h)
        chol_raw = self._chol_head(h)
        L = _build_cholesky(chol_raw, self.d_theta)
        return mu, L


def _build_cholesky(
    entries: Float[Array, " n_chol"], d: int, eps: float = 1e-6
) -> Float[Array, " d d"]:
    """Assemble lower-triangular L from a flat vector, softplus on diagonal."""
    L = jnp.zeros((d, d)).at[jnp.tril_indices(d)].set(entries)
    diag = jnp.diag_indices(d)
    L = L.at[diag].set(jax.nn.softplus(L[diag]) + eps)
    return L


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def gaussian_nll(
    model: ConditionalGaussianNPE,
    theta: Float[Array, " d_theta"],
    s: Float[Array, " d_s"],
) -> Float[Array, ""]:
    r"""Negative log-likelihood of $\theta$ under $\mathcal{N}(\mu(S), L(S)L(S)^\top)$.

    Uses the Cholesky factor directly:
    $$-\log p = \tfrac{d}{2}\log(2\pi) + \sum_i \log L_{ii}
                + \tfrac{1}{2}\|L^{-1}(\theta - \mu)\|^2.$$
    """
    mu, L = model(s)
    d = mu.shape[0]
    diff = theta - mu
    z = jax.scipy.linalg.solve_triangular(L, diff, lower=True)
    log_det = jnp.sum(jnp.log(jnp.diag(L)))
    return 0.5 * d * jnp.log(2.0 * jnp.pi) + log_det + 0.5 * jnp.dot(z, z)


def _batch_loss(
    model: ConditionalGaussianNPE,
    thetas: Float[Array, "n d_theta"],
    summaries: Float[Array, "n d_s"],
) -> Float[Array, ""]:
    """Mean NLL over a batch."""
    return jax.vmap(lambda t, s: gaussian_nll(model, t, s))(thetas, summaries).mean()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class TrainConfig(NamedTuple):
    """Training hyperparameters (sensible defaults for low-dim problems)."""

    lr: float = 5e-4
    batch_size: int = 256
    max_epochs: int = 2000
    patience: int = 10
    val_frac: float = 0.1


def fit(
    model: ConditionalGaussianNPE,
    thetas: Float[Array, "n d_theta"],
    summaries: Float[Array, "n d_s"],
    *,
    key: PRNGKeyArray,
    config: TrainConfig = TrainConfig(),
) -> tuple[ConditionalGaussianNPE, dict[str, list[float]]]:
    """Train the conditional Gaussian via Adam + early stopping.

    Returns the best model (by validation loss) and loss history dict.
    """
    n = thetas.shape[0]
    n_val = max(1, int(n * config.val_frac))

    key, subkey = jr.split(key)
    perm = jr.permutation(subkey, n)
    t_train, s_train = thetas[perm[n_val:]], summaries[perm[n_val:]]
    t_val, s_val = thetas[perm[:n_val]], summaries[perm[:n_val]]

    opt = optax.adam(config.lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(model, opt_state, t_batch, s_batch):
        loss, grads = eqx.filter_value_and_grad(_batch_loss)(model, t_batch, s_batch)
        updates, opt_state = opt.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    @eqx.filter_jit
    def eval_loss(model, t, s):
        return _batch_loss(model, t, s)

    best_val_loss = jnp.inf
    best_model = model
    wait = 0
    train_losses: list[float] = []
    val_losses: list[float] = []

    n_train = t_train.shape[0]
    n_batches = max(1, n_train // config.batch_size)

    for epoch in range(config.max_epochs):
        key, subkey = jr.split(key)
        idx_perm = jr.permutation(subkey, n_train)
        epoch_loss = 0.0

        for b in range(n_batches):
            idx = idx_perm[b * config.batch_size : (b + 1) * config.batch_size]
            model, opt_state, loss = step(model, opt_state, t_train[idx], s_train[idx])
            epoch_loss += float(loss)

        train_losses.append(epoch_loss / n_batches)
        vl = float(eval_loss(model, t_val, s_val))
        val_losses.append(vl)

        if vl < best_val_loss:
            best_val_loss = vl
            best_model = model
            wait = 0
        else:
            wait += 1
        if wait >= config.patience:
            break

    return best_model, {"train": train_losses, "val": val_losses}


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample(
    model: ConditionalGaussianNPE,
    s: Float[Array, " d_s"],
    n_samples: int,
    *,
    key: PRNGKeyArray,
) -> Float[Array, "n d_theta"]:
    r"""Draw from $\mathcal{N}(\mu_\phi(S),\, L_\phi(S) L_\phi(S)^\top)$."""
    mu, L = model(s)
    z = jr.normal(key, (n_samples, mu.shape[0]))
    return mu + z @ L.T
