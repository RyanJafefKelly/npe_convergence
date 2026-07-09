"""Assess whether the NUTS posterior is approximately Gaussian.

Tests multivariate and marginal normality of the exact (NUTS) posterior
samples across different n_obs values. This helps determine whether BvM
has kicked in, which is the regime where Gaussian NPE should shine.

Run locally (uses available cached files) or on HPC (full seed range).

Usage:
    python assess_bvm.py                  # use all available seeds
    python assess_bvm.py --seeds 0 5      # seeds 0..4
    python assess_bvm.py --plot           # save Q-Q plots
"""

import argparse
import io
import os
import pickle as pkl

import numpy as np
from scipy import stats


class _JaxStubUnpickler(pkl.Unpickler):
    """Unpickler that handles JAX arrays without requiring JAX installed.

    When pickle encounters a JAX DeviceArray/ArrayImpl, this redirects
    reconstruction to produce a plain numpy array instead.
    """

    def find_class(self, module, name):
        # Intercept any JAX array class and map to a helper that
        # just returns the numpy data stored in the pickle.
        if "jax" in module:
            return _jax_array_from_numpy
        return super().find_class(module, name)


def _jax_array_from_numpy(*args, **kwargs):
    """Fallback reconstructor for JAX arrays — returns numpy."""
    # JAX arrays are pickled as numpy arrays under the hood, so
    # the reduce tuple typically contains the numpy data.
    # If called with a single ndarray arg, just return it.
    if len(args) == 1 and isinstance(args[0], np.ndarray):
        return args[0]
    return np.array(args[0]) if args else np.array([])


def _load_pickle_robust(path):
    """Load pickle, falling back to stub unpickler if JAX is unavailable."""
    with open(path, "rb") as f:
        try:
            return pkl.load(f)
        except ModuleNotFoundError:
            pass
    # Retry with stub unpickler
    with open(path, "rb") as f:
        return _JaxStubUnpickler(f).load()


def load_nuts_samples(n_obs, seed, convention="gaussian"):
    """Load NUTS posterior samples for a (n_obs, seed, convention) cell.

    Resolution order:
    1. Canonical v3 reference:
       ``res/gnk_v3_refs/nuts_n_obs_{n}_seed_{s}_conv_{convention}.pkl``.
       This is a fingerprinted dict; we extract and flatten the
       ``samples`` array of shape ``(num_chains, samples_per_chain, 4)``.
    2. Legacy v2 cache:
       ``res/gnk/nuts_cache_v2{flow_suffix}_n_obs_{n}_seed_{s}.pkl``
       where ``flow_suffix`` is ``"_flow"`` if convention=="flow" else "".
    3. Pre-v2 legacy cache (returned only if v3 and v2 both miss):
       ``res/gnk/nuts_cache_n_obs_{n}_seed_{s}.pkl``.

    Returns ``(samples, source)`` where ``samples`` is ``(n_total, 4)``
    and ``source`` describes which artefact was loaded, or ``None`` if
    nothing matches.
    """
    v3_path = (
        f"res/gnk_v3_refs/nuts_n_obs_{n_obs}_seed_{seed}_conv_{convention}.pkl"
    )
    if os.path.isfile(v3_path):
        fingerprint = _load_pickle_robust(v3_path)
        if isinstance(fingerprint, dict) and "samples" in fingerprint:
            grouped = np.asarray(fingerprint["samples"])
            flat = grouped.reshape(-1, grouped.shape[-1])
            return flat, v3_path
        return None
    flow_suffix = "_flow" if convention == "flow" else ""
    legacy_v2 = f"res/gnk/nuts_cache_v2{flow_suffix}_n_obs_{n_obs}_seed_{seed}.pkl"
    if os.path.isfile(legacy_v2):
        return np.array(_load_pickle_robust(legacy_v2)), legacy_v2
    pre_v2 = f"res/gnk/nuts_cache_n_obs_{n_obs}_seed_{seed}.pkl"
    if os.path.isfile(pre_v2):
        return np.array(_load_pickle_robust(pre_v2)), pre_v2
    return None


def marginal_normality_tests(samples, param_names):
    """Run Shapiro-Wilk on each marginal. Returns dict of p-values."""
    results = {}
    for j, name in enumerate(param_names):
        x = samples[:, j]
        # Shapiro-Wilk limited to 5000 samples; subsample if needed
        if len(x) > 5000:
            rng = np.random.default_rng(42)
            x = rng.choice(x, 5000, replace=False)
        stat, pval = stats.shapiro(x)
        results[name] = {"shapiro_stat": stat, "shapiro_p": pval}

        # Also compute skewness and kurtosis
        results[name]["skewness"] = float(stats.skew(x))
        results[name]["excess_kurtosis"] = float(stats.kurtosis(x))
    return results


def mardia_tests(samples):
    """Mardia's multivariate skewness and kurtosis tests."""
    n, p = samples.shape
    mean = samples.mean(axis=0)
    centered = samples - mean
    cov = np.cov(centered, rowvar=False)
    cov_inv = np.linalg.inv(cov)

    # Mahalanobis-like inner products: D_ij = (x_i - mu)' Sigma^{-1} (x_j - mu)
    D = centered @ cov_inv @ centered.T

    # Mardia skewness: (1/n^2) * sum D_ij^3
    skewness = (D ** 3).mean()
    # Under H0: n * skewness / 6 ~ chi2(p*(p+1)*(p+2)/6)
    skew_stat = n * skewness / 6
    skew_df = p * (p + 1) * (p + 2) / 6
    skew_p = 1 - stats.chi2.cdf(skew_stat, skew_df)

    # Mardia kurtosis: (1/n) * sum D_ii^2
    kurtosis = (np.diag(D) ** 2).mean()
    # Under H0: kurtosis ~ N(p*(p+2), 8*p*(p+2)/n)
    kurt_expected = p * (p + 2)
    kurt_std = np.sqrt(8 * p * (p + 2) / n)
    kurt_z = (kurtosis - kurt_expected) / kurt_std
    kurt_p = 2 * (1 - stats.norm.cdf(abs(kurt_z)))

    return {
        "mardia_skewness": skewness,
        "mardia_skew_stat": skew_stat,
        "mardia_skew_p": skew_p,
        "mardia_kurtosis": kurtosis,
        "mardia_kurt_z": kurt_z,
        "mardia_kurt_p": kurt_p,
    }


def kl_to_fitted_gaussian(samples):
    """Estimate KL(NUTS || fitted Gaussian) using log-likelihood ratio.

    Fits a Gaussian to the samples and computes the average difference
    between the empirical log-density (via KDE) and the Gaussian log-density.
    Also returns a simpler measure: the Gaussian log-likelihood (higher = more Gaussian).
    """
    n, d = samples.shape
    mean = samples.mean(axis=0)
    cov = np.cov(samples, rowvar=False)

    # Gaussian log-likelihood per sample
    rv = stats.multivariate_normal(mean=mean, cov=cov)
    gauss_ll = rv.logpdf(samples).mean()

    return {"gauss_mean_loglik": gauss_ll}


def make_qq_plots(samples, param_names, n_obs, seed, outdir="plots/bvm"):
    """Save Q-Q plots for each parameter."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    fig, axes = plt.subplots(1, len(param_names), figsize=(4 * len(param_names), 4))
    for j, (ax, name) in enumerate(zip(axes, param_names)):
        x = np.sort(samples[:, j])
        # Standardize
        x_std = (x - x.mean()) / x.std()
        theoretical = stats.norm.ppf(np.linspace(0.001, 0.999, len(x_std)))
        ax.scatter(theoretical, np.sort(x_std), s=1, alpha=0.3)
        lims = [min(theoretical.min(), x_std.min()), max(theoretical.max(), x_std.max())]
        ax.plot(lims, lims, "r--", linewidth=1)
        ax.set_xlabel("Theoretical quantiles")
        ax.set_ylabel("Sample quantiles")
        ax.set_title(f"{name} (n_obs={n_obs})")
    plt.tight_layout()
    path = f"{outdir}/qq_n_obs_{n_obs}_seed_{seed}.pdf"
    plt.savefig(path)
    plt.close()
    return path


def main():
    parser = argparse.ArgumentParser(description="Assess BvM for NUTS posteriors")
    parser.add_argument("--seeds", type=int, nargs=2, default=None,
                        help="Seed range [start, end). Default: use all available.")
    parser.add_argument("--plot", action="store_true", help="Generate Q-Q plots")
    parser.add_argument(
        "--convention",
        type=str,
        default="gaussian",
        choices=("flow", "gaussian"),
        help=(
            "x_obs convention for resolving canonical v3 references. The two "
            "pipelines condition NUTS on different data realisations; see "
            "docs/meeting_2026_05_18 for context. Default: gaussian."
        ),
    )
    args = parser.parse_args()

    n_obs_list = [100, 500, 1000, 5000]
    param_names = ["A", "B", "g", "k"]

    if args.seeds:
        seed_range = range(args.seeds[0], args.seeds[1])
    else:
        # Auto-detect available seeds
        seed_range = range(101)

    print("=" * 80)
    print("BvM Assessment: Is the NUTS posterior approximately Gaussian?")
    print("=" * 80)

    for n_obs in n_obs_list:
        # Collect results across seeds
        all_marginal = {p: {"skewness": [], "excess_kurtosis": [], "shapiro_p": []}
                        for p in param_names}
        all_mardia_skew_p = []
        all_mardia_kurt_p = []
        n_loaded = 0

        for seed in seed_range:
            loaded = load_nuts_samples(n_obs, seed, convention=args.convention)
            if loaded is None:
                continue
            samples, source = loaded
            n_loaded += 1
            if n_loaded == 1:
                print(f"  source ({n_obs}, seed={seed}): {source}")

            # Marginal tests
            marg = marginal_normality_tests(samples, param_names)
            for p in param_names:
                all_marginal[p]["skewness"].append(marg[p]["skewness"])
                all_marginal[p]["excess_kurtosis"].append(marg[p]["excess_kurtosis"])
                all_marginal[p]["shapiro_p"].append(marg[p]["shapiro_p"])

            # Multivariate tests
            mardia = mardia_tests(samples)
            all_mardia_skew_p.append(mardia["mardia_skew_p"])
            all_mardia_kurt_p.append(mardia["mardia_kurt_p"])

            # Q-Q plots (only for first seed found)
            if args.plot and n_loaded == 1:
                path = make_qq_plots(samples, param_names, n_obs, seed)
                print(f"  Q-Q plot saved: {path}")

        if n_loaded == 0:
            print(f"\nn_obs={n_obs}: no cached NUTS files found, skipping.")
            continue

        print(f"\nn_obs={n_obs} ({n_loaded} seeds)")
        print("-" * 60)

        # Marginal summary
        print(f"  {'Param':<6} {'Skewness':>12} {'Ex. Kurtosis':>14} {'Shapiro p':>12}")
        for p in param_names:
            skew = np.array(all_marginal[p]["skewness"])
            kurt = np.array(all_marginal[p]["excess_kurtosis"])
            shap = np.array(all_marginal[p]["shapiro_p"])
            print(f"  {p:<6} {np.mean(skew):>8.3f} ({np.std(skew):.3f})"
                  f" {np.mean(kurt):>10.3f} ({np.std(kurt):.3f})"
                  f" {np.median(shap):>8.4f}")

        # Multivariate summary
        skew_p = np.array(all_mardia_skew_p)
        kurt_p = np.array(all_mardia_kurt_p)
        print(f"\n  Mardia skewness test:  median p = {np.median(skew_p):.4f}"
              f"  (reject at 0.05: {(skew_p < 0.05).mean()*100:.0f}% of seeds)")
        print(f"  Mardia kurtosis test: median p = {np.median(kurt_p):.4f}"
              f"  (reject at 0.05: {(kurt_p < 0.05).mean()*100:.0f}% of seeds)")

    print()


if __name__ == "__main__":
    main()
