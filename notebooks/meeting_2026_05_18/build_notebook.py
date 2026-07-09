"""Build the meeting notebook (empirical_results_summary.ipynb).

Run from the repo root with the project virtualenv:

    .venv/bin/python notebooks/meeting_2026_05_18/build_notebook.py
    .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
        notebooks/meeting_2026_05_18/empirical_results_summary.ipynb
    .venv/bin/jupyter nbconvert --to pdf --no-input \
        notebooks/meeting_2026_05_18/empirical_results_summary.ipynb
"""

import json
import pickle
import re
import shutil
from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
REPO = HERE.parents[1]
POST = REPO / "docs/paper_empirical_push_2026_05_13/postprocessing_20260513"
STEREO_DIR = POST / "stereological_current_cache_aquarius_20260513"

SOURCES = {
    "gnk_kl_flow_vs_gaussian.csv":
        REPO / "notebooks/plots/gnk_task2_20260526_v3/raw_theta_kl_summary_comparable.csv",
    "gnk_kl_paired_per_seed.csv":
        REPO / "notebooks/plots/gnk_task2_20260526_v3/raw_theta_kl_paired_per_seed.csv",
    "gnk_theta_oracle_by_n.csv":
        REPO / "notebooks/plots/gnk_task2_20260526_v3/theta_oracle_by_n.csv",
    "gnk_hexadecile_gaussian.csv":
        POST / "gnk_hexadecile_gaussian/gnk_hexadecile_group_summary.csv",
    "gnk_robust_scaling_overlay.csv":
        DATA / "gnk_robust_scaling_overlay.csv",
    "gnk_robust_scaling_summary.json":
        DATA / "gnk_robust_scaling_summary.json",
    "gnk_bsl_diagnostic.csv":
        DATA / "gnk_bsl_diagnostic.csv",
    "gnk_robust_scaling_n5000_2M_summary.json":
        DATA / "gnk_robust_scaling_n5000_2M_summary.json",
    "gnk_rejection_abc_summary.json":
        DATA / "gnk_rejection_abc_summary.json",
    "dim_scaling_pilot_kl_by_d.csv":
        DATA / "dim_scaling_pilot_kl_by_d.csv",
    "stereological_coverage.csv": STEREO_DIR / "coverage_all_params.csv",
    "stereological_bias_by_seed.csv": STEREO_DIR / "bias_boxplot_by_seed.csv",
    "stereological_posterior_overlay.csv":
        STEREO_DIR / "posterior_overlay_density_n1000_seed1_with_local_abc.csv",
    "ma2_compatibility.csv":
        POST / "ma2_b0_compat/ma2_b0_grouped_finite_kl_mmd_summary.csv",
    "ma2_compatibility_flow.csv":
        DATA / "ma2_compatibility_flow.csv",
    "ma2_compatibility_gaussian.csv":
        DATA / "ma2_compatibility_gaussian.csv",
    "ma2_delta1_refresh.csv":
        DATA / "ma2_delta1_refresh.csv",
    "ma2_delta1_refresh_audit.csv":
        DATA / "ma2_delta1_refresh_audit.csv",
    "ma2_b0_flow_current_reference_per_seed.csv":
        DATA / "ma2_b0_flow_current_reference_per_seed.csv",
    "ma2_b0_flow_current_reference_summary.csv":
        DATA / "ma2_b0_flow_current_reference_summary.csv",
    "ma2_b0_gaussian_current_reference_per_seed.csv":
        DATA / "ma2_b0_gaussian_current_reference_per_seed.csv",
    "ma2_b0_reference_audit.csv":
        DATA / "ma2_b0_reference_audit.csv",
    "ma2_posterior_overlay_seed_22.csv":
        DATA / "ma2_posterior_overlay_seed_22.csv",
    "ma2_posterior_overlay_seed_22.png":
        DATA / "ma2_posterior_overlay_seed_22.png",
}

GNK_OVERLAY_CELLS = [
    ("n=1000, N=n^(3/2)", 1000, 31622, 36),
    ("n=1000, N=n^2", 1000, 1000000, 36),
    ("n=5000, N=n^(3/2)", 5000, 353553, 50),
    ("n=5000, N=n^2", 5000, 25000000, 50),
]
BUDGETS = ["N=n", "N=n log(n)", "N=n^(3/2)", "N=n^2"]
N_OBS = [100, 500, 1000, 5000]
GNK_PARAMS = ["A", "B", "g", "k"]


def _read_kl(path):
    try:
        v = float(Path(path).read_text().strip())
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def _read_json(path):
    with open(path, "r") as fh:
        return json.load(fh)


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _copy_if_needed(src, dst):
    if not src.exists():
        if not dst.exists():
            print("missing source and staged file:", src)
        return
    if src.resolve() == dst.resolve():
        return
    shutil.copy(src, dst)


def stage_inputs():
    DATA.mkdir(exist_ok=True)
    for name, src in SOURCES.items():
        _copy_if_needed(Path(src), DATA / name)
    for stale in ("gnk_oracle_floor_by_n.csv", "stereological_bias.csv",
                  "ma2_bivariate.csv"):
        (DATA / stale).unlink(missing_ok=True)
    print("staged or checked", len(SOURCES), "input files")


def _load_v3_reference(n_obs, seed, convention):
    path = (
        REPO
        / "res"
        / "gnk_v3_refs"
        / f"nuts_n_obs_{n_obs}_seed_{seed}_conv_{convention}.pkl"
    )
    with open(path, "rb") as fh:
        fingerprint = pickle.load(fh)
    grouped = np.asarray(fingerprint["samples"], dtype=float)
    return grouped.reshape(-1, grouped.shape[-1])


def build_gnk_overlay():
    """KDE overlays of the g-and-k posteriors for selected cells."""
    rows = []
    for label, n, n_sims, seed in GNK_OVERLAY_CELLS:
        cell = f"n_obs_{n}_n_sims_{n_sims}_seed_{seed}"
        flow_post_path = REPO / f"res/gnk/npe_{cell}/posterior_samples.pkl"
        gauss_post_path = REPO / f"res/gnk/gaussian_npe_{cell}/posterior_samples.pkl"
        with open(flow_post_path, "rb") as fh:
            flow_post = np.asarray(pickle.load(fh), dtype=float)
        with open(gauss_post_path, "rb") as fh:
            gauss_post = np.asarray(pickle.load(fh), dtype=float)

        flow_ref = _load_v3_reference(n, seed, "flow")
        gauss_ref = _load_v3_reference(n, seed, "gaussian")
        sources = {
            "Reference (flow convention)": flow_ref,
            "Reference (Gaussian convention)": gauss_ref,
            "flow-NPE": flow_post,
            "Gaussian-NPE": gauss_post,
        }
        for j, param in enumerate(GNK_PARAMS):
            spans = {m: np.percentile(arr[:, j], [1.0, 99.0])
                     for m, arr in sources.items()}
            lo = float(min(v[0] for v in spans.values()))
            hi = float(max(v[1] for v in spans.values()))
            pad = 0.08 * (hi - lo)
            xs = np.linspace(lo - pad, hi + pad, 200)
            for name, arr in sources.items():
                dens = gaussian_kde(arr[:, j])(xs)
                for x, d in zip(xs, dens):
                    rows.append({
                        "cell": label,
                        "method": name,
                        "param": param,
                        "x": float(x),
                        "density": float(d),
                    })
    pd.DataFrame(rows).to_csv(DATA / "gnk_posterior_overlay.csv", index=False)
    print("wrote gnk_posterior_overlay.csv")


def aggregate_flow_hexadecile():
    rows = []
    base = REPO / "res/gnk_hexadeciles"
    for d in sorted(base.glob("npe_n_obs_*_n_sims_*_seed_*")):
        parts = d.name.split("_")
        n_obs = int(parts[parts.index("obs") + 1])
        n_sims = int(parts[parts.index("sims") + 1])
        rows.append({"n_obs": n_obs, "n_sims": n_sims,
                     "kl": _read_kl(d / "kl.txt")})
    if not rows:
        return
    df = pd.DataFrame(rows)
    out = (df.groupby(["n_obs", "n_sims"])["kl"]
             .agg(kl_median=lambda s: np.nanmedian(s),
                  n_seeds=lambda s: int(np.isfinite(s).sum()))
             .reset_index())
    out.to_csv(DATA / "gnk_hexadecile_flow.csv", index=False)
    print("wrote gnk_hexadecile_flow.csv with", len(out), "cells")


def bootstrap_kl_ci(n_resamples=1000, seed=20260528):
    src = DATA / "gnk_kl_paired_per_seed.csv"
    df = pd.read_csv(src)
    df = df[df.standard_paper_grid == 1].copy()
    rng = np.random.default_rng(seed)
    rows = []
    for (n, N, label), g in df.groupby(["n", "N", "N_label"], sort=True):
        g = g.dropna(subset=["flow_theta_kl", "gaussian_theta_kl"])
        if g.empty:
            continue
        flow = g.flow_theta_kl.to_numpy(float)
        gauss = g.gaussian_theta_kl.to_numpy(float)
        diff = gauss - flow
        m = len(g)
        boot_flow = []
        boot_gauss = []
        boot_diff = []
        for _ in range(n_resamples):
            idx = rng.integers(0, m, size=m)
            boot_flow.append(float(np.median(flow[idx])))
            boot_gauss.append(float(np.median(gauss[idx])))
            boot_diff.append(float(np.median(diff[idx])))
        rows.append({
            "n": int(n),
            "N": int(N),
            "N_label": label,
            "seed_count": int(m),
            "flow_median": float(np.median(flow)),
            "flow_ci05": float(np.quantile(boot_flow, 0.05)),
            "flow_ci95": float(np.quantile(boot_flow, 0.95)),
            "gaussian_median": float(np.median(gauss)),
            "gaussian_ci05": float(np.quantile(boot_gauss, 0.05)),
            "gaussian_ci95": float(np.quantile(boot_gauss, 0.95)),
            "gaussian_minus_flow_median": float(np.median(diff)),
            "gaussian_minus_flow_ci05": float(np.quantile(boot_diff, 0.05)),
            "gaussian_minus_flow_ci95": float(np.quantile(boot_diff, 0.95)),
        })
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "gnk_paper_grid_bootstrap_ci.csv", index=False)
    print("wrote gnk_paper_grid_bootstrap_ci.csv with", len(out), "rows")


def build_bsl_diagnostic_table():
    base = REPO / "res/gnk_bsl"
    candidates = sorted(base.glob("bsl_n_obs_1000_seed_0_M_500_*"))
    latest = None
    for path in candidates:
        diag = path / "diagnostics.json"
        if not diag.exists():
            continue
        payload = _read_json(diag)
        gate = payload.get("acceptance_gate", {})
        kl = gate.get("kl", {})
        if "KL_BSL_unique_to_NUTS" in kl:
            latest = path
    if latest is None and candidates:
        latest = candidates[-1]
    if latest is None:
        pd.DataFrame([{"status": "pending"}]).to_csv(
            DATA / "gnk_bsl_diagnostic.csv", index=False)
        return

    diagnostics = _read_json(latest / "diagnostics.json")
    gate = diagnostics.get("acceptance_gate", {})
    kl = gate.get("kl", {})
    accept = float(np.mean(diagnostics.get("sample_acceptance_rate", [np.nan])))
    rows = []
    for param in GNK_PARAMS:
        per = diagnostics["per_parameter"][param]
        rows.append({
            "parameter": param,
            "NUTS median": gate["nuts_median"][param],
            "NUTS sd": gate["nuts_std"][param],
            "BSL median": gate["bsl_median"][param],
            "BSL sd": per["std"],
            "R-hat": per["r_hat"],
            "ESS": per["n_eff"],
            "acceptance rate": accept,
            "KL_BSL_unique_to_NUTS": kl.get("KL_BSL_unique_to_NUTS", np.nan),
            "KL_NUTS_to_BSL_unique": kl.get("KL_NUTS_to_BSL_unique", np.nan),
        })
    pd.DataFrame(rows).to_csv(DATA / "gnk_bsl_diagnostic.csv", index=False)
    print("wrote gnk_bsl_diagnostic.csv")


def _median_shift(post, ref):
    shifts = {}
    for j, param in enumerate(GNK_PARAMS):
        ref_sd = float(np.std(ref[:, j], ddof=1))
        shifts[param] = float((np.median(post[:, j]) - np.median(ref[:, j])) / ref_sd)
    shifts["max_abs"] = float(max(abs(v) for v in shifts.values()))
    return shifts


def build_robust_scaling_panel():
    source_png = (
        REPO
        / "res/gnk_robust_scale/"
        / "gaussian_npe_n_obs_5000_n_sims_2000000_seed_50_transform_asinh/"
        / "gnk_robust_scaling_overlay_n5000_2M.png"
    )
    if source_png.exists():
        shutil.copy(source_png, DATA / "gnk_robust_scaling_n5000_2M.png")

    metrics_path = (
        REPO
        / "res/gnk_robust_scale/"
        / "gaussian_npe_n_obs_5000_n_sims_2000000_seed_50_transform_asinh/"
        / "metrics.json"
    )
    standard_path = (
        REPO
        / "res/gnk/gaussian_npe_n_obs_5000_n_sims_25000000_seed_50/"
        / "metrics_v3.json"
    )
    post_path = metrics_path.parent / "posterior_samples.pkl"
    ref_path = REPO / "res/gnk_v3_refs/nuts_n_obs_5000_seed_50_conv_gaussian.pkl"
    if not metrics_path.exists():
        _write_json(DATA / "gnk_robust_scaling_n5000_2M_summary.json",
                    {"status": "pending"})
        return
    metrics = _read_json(metrics_path)
    standard = _read_json(standard_path) if standard_path.exists() else {}
    shifts = {}
    if post_path.exists() and ref_path.exists():
        with open(post_path, "rb") as fh:
            post = np.asarray(pickle.load(fh), dtype=float)
        ref = _load_v3_reference(5000, 50, "gaussian")
        shifts = _median_shift(post, ref)
    payload = {
        "status": "complete",
        "n_obs": metrics.get("n_obs"),
        "seed": metrics.get("seed"),
        "n_sims": metrics.get("n_sims"),
        "kl_value": metrics.get("kl_value"),
        "mmd_value": metrics.get("mmd_value"),
        "standard_zscore_n_sims": standard.get("n_sims"),
        "standard_zscore_kl_value": standard.get("kl_value"),
        "standard_zscore_mmd_value": standard.get("mmd_value"),
        "train_epochs": metrics.get("train_epochs"),
        "stop_reason": metrics.get("training_info", {}).get("stop_reason"),
        "marginal_shifts_in_nuts_sd": shifts,
        "png": "gnk_robust_scaling_n5000_2M.png" if source_png.exists() else None,
    }
    _write_json(DATA / "gnk_robust_scaling_n5000_2M_summary.json", payload)
    print("wrote gnk_robust_scaling_n5000_2M_summary.json")


def build_rejection_abc_panel():
    root = (
        REPO
        / "res/gnk_rejection_abc/"
        / "gaussian_npe_n_obs_1000_n_pool_10000000_seed_0_acc_0.01"
    )
    metrics_path = root / "metrics.json"
    if not metrics_path.exists():
        _write_json(DATA / "gnk_rejection_abc_summary.json",
                    {"status": "pending"})
        return
    metrics = _read_json(metrics_path)
    bsl = pd.read_csv(DATA / "gnk_bsl_diagnostic.csv")
    bsl_kl = float(bsl["KL_NUTS_to_BSL_unique"].dropna().iloc[0])
    payload = {
        "status": "complete",
        "n_obs": metrics.get("n_obs"),
        "seed": metrics.get("seed"),
        "n_pool": metrics.get("n_pool"),
        "acceptance": metrics.get("acceptance"),
        "n_keep": metrics.get("n_keep"),
        "kl_reference_to_npe_value": metrics.get("kl_reference_to_npe_value"),
        "kl_npe_to_reference_value": metrics.get("kl_npe_to_reference_value"),
        "mmd_value": metrics.get("mmd_value"),
        "train_epochs": metrics.get("train_epochs"),
        "stop_reason": metrics.get("training_info", {}).get("stop_reason"),
        "runtime_seconds": metrics.get("runtime_seconds"),
        "simulation_seconds": metrics.get("simulation_seconds"),
        "selection_seconds": metrics.get("selection_seconds"),
        "training_seconds": metrics.get("training_seconds"),
        "max_marginal_shift": max(
            abs(v["shift"]) for v in metrics.get("marginal_shifts", {}).values()
        ),
        "max_shift_parameter": max(
            metrics.get("marginal_shifts", {}),
            key=lambda p: abs(metrics["marginal_shifts"][p]["shift"]),
        ),
        "vanilla_canonical_kl": metrics.get("comparators", {}).get("vanilla_canonical_kl"),
        "robust_canonical_kl": metrics.get("comparators", {}).get("robust_canonical_kl"),
        "bsl_unique_kl": bsl_kl,
    }
    _write_json(DATA / "gnk_rejection_abc_summary.json", payload)
    print("wrote gnk_rejection_abc_summary.json")


def build_dim_scaling_panel():
    manifest = (
        REPO
        / "res/gnk_dim_scaling/manifests/"
        / "gnk_dim_scaling_cells_manifest_20260526T000000Z.csv"
    )
    if not manifest.exists():
        pd.DataFrame([{"status": "pending"}]).to_csv(
            DATA / "dim_scaling_pilot_kl_by_d.csv", index=False)
        return
    rows = []
    cells = pd.read_csv(manifest)
    for _, row in cells.iterrows():
        method = row["method"]
        d_s = int(row["d_s"])
        n_obs = int(row["n_obs"])
        n_sims = int(row["n_sims"])
        seed = int(row["seed"])
        path = (
            REPO
            / "res/gnk_dim_scaling"
            / f"gnk_{method}_d_s_{d_s}_n_obs_{n_obs}_n_sims_{n_sims}_seed_{seed}"
            / "metrics.json"
        )
        theta_kl = np.nan
        mmd = np.nan
        status = "missing_metrics"
        if path.exists():
            metrics = _read_json(path)
            theta_kl = float(metrics.get("kl_theta_knn_2000", np.nan))
            mmd = float(metrics.get("mmd_theta_2000", np.nan))
            status = "finite_metric" if np.isfinite(theta_kl) else "nonfinite_metric"
        d_total = int(row["d"])
        rows.append({
            "method": method,
            "seed": seed,
            "n_obs": n_obs,
            "d_s": d_s,
            "d": d_total,
            "N_sims": n_sims,
            "N_over_d2n": float(n_sims / (d_total ** 2 * n_obs)),
            "theta_kl": theta_kl,
            "mmd": mmd,
            "status": status,
        })
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "dim_scaling_pilot_kl_by_d.csv", index=False)
    print("wrote dim_scaling_pilot_kl_by_d.csv with", len(out), "rows")


def build_ma2_audit_caveat():
    src = REPO / "res/ma2_b0/audit_n_obs_1000_pilot_metrics.json"
    if not src.exists():
        pd.DataFrame([{"status": "pending"}]).to_csv(
            DATA / "ma2_b0_reference_audit.csv", index=False)
        return
    payload = _read_json(src)
    rows = []
    for row in payload:
        r = dict(row)
        r["status"] = "complete"
        rows.append(r)
    pd.DataFrame(rows).to_csv(DATA / "ma2_b0_reference_audit.csv", index=False)
    print("wrote ma2_b0_reference_audit.csv")


def aggregate_ma2_b0_kl():
    rows = []
    base = REPO / "res/ma2_b0"
    for prefix, method in [("npe", "flow_npe"), ("gaussian_npe", "gaussian_npe")]:
        for d in sorted(base.glob(f"{prefix}_n_obs_*_n_sims_*_seed_*")):
            if d.name.endswith(("_audit", "_overlay")):
                continue
            parts = d.name.split("_")
            n_obs = int(parts[parts.index("obs") + 1])
            n_sims = int(parts[parts.index("sims") + 1])
            rows.append({"method": method, "n_obs": n_obs, "n_sims": n_sims,
                         "kl": _read_kl(d / "kl.txt")})
    df = pd.DataFrame(rows)
    out = (df.groupby(["method", "n_obs", "n_sims"])["kl"]
             .agg(kl_median=lambda s: np.nanmedian(s),
                  n_seeds=lambda s: int(np.isfinite(s).sum()))
             .reset_index())
    out["current_reference_kl_median"] = np.nan
    out["current_reference_seed_count"] = 0
    flow_current_path = DATA / "ma2_b0_flow_current_reference_per_seed.csv"
    if flow_current_path.exists():
        flow_current = pd.read_csv(flow_current_path)["current_flow_kl"].dropna()
        mask = (
            (out.method == "flow_npe")
            & (out.n_obs == 1000)
            & (out.n_sims == 1000000)
        )
        out.loc[mask, "current_reference_kl_median"] = float(np.median(flow_current))
        out.loc[mask, "current_reference_seed_count"] = int(len(flow_current))

    gaussian_current_path = DATA / "ma2_b0_gaussian_current_reference_per_seed.csv"
    if gaussian_current_path.exists():
        gauss = pd.read_csv(gaussian_current_path)
        gauss_current = gauss["gauss_kl_current_recomputed"].dropna()
        if len(gauss_current):
            mask = (
                (out.method == "gaussian_npe")
                & (out.n_obs == 1000)
                & (out.n_sims == 1000000)
            )
            out.loc[mask, "current_reference_kl_median"] = float(np.median(gauss_current))
            out.loc[mask, "current_reference_seed_count"] = int(len(gauss_current))

    audit_path = DATA / "ma2_b0_reference_audit.csv"
    if audit_path.exists():
        audit = pd.read_csv(audit_path)
        if "flow_kl_current" in audit.columns and not flow_current_path.exists():
            flow_current = audit["flow_kl_current"].dropna()
            mask = (
                (out.method == "flow_npe")
                & (out.n_obs == 1000)
                & (out.n_sims == 1000000)
            )
            out.loc[mask, "current_reference_kl_median"] = float(np.median(flow_current))
            out.loc[mask, "current_reference_seed_count"] = int(len(flow_current))
        if "gauss_kl_current" in audit.columns and not gaussian_current_path.exists():
            gauss_current = audit["gauss_kl_current"].dropna()
            if len(gauss_current):
                mask = (
                    (out.method == "gaussian_npe")
                    & (out.n_obs == 1000)
                    & (out.n_sims == 1000000)
                )
                out.loc[mask, "current_reference_kl_median"] = float(np.median(gauss_current))
                out.loc[mask, "current_reference_seed_count"] = int(len(gauss_current))
    out.to_csv(DATA / "ma2_b0_kl.csv", index=False)
    print("wrote ma2_b0_kl.csv with", len(out), "rows")


def write_sources():
    text = (
        "# Data sources\n\n"
        "Staged or generated by build_notebook.py so the notebook is "
        "self-contained.\n\n"
        "- gnk_kl_flow_vs_gaussian.csv: copied from the g-and-k v3 summary.\n"
        "- gnk_kl_paired_per_seed.csv and gnk_paper_grid_bootstrap_ci.csv: "
        "paired-seed g-and-k KL values and bootstrap summaries.\n"
        "- gnk_theta_oracle_by_n.csv: moment-matched Gaussian oracle row.\n"
        "- gnk_hexadecile_gaussian.csv and gnk_hexadecile_flow.csv: "
        "hexadecile summary-dimension comparison.\n"
        "- gnk_posterior_overlay.csv: KDE of selected g-and-k posterior draws.\n"
        "- gnk_bsl_diagnostic.csv: dense-proposal BSL diagnostic table.\n"
        "- gnk_robust_scaling_overlay.csv and gnk_robust_scaling_summary.json: "
        "n=1000 robust standardisation diagnostic.\n"
        "- gnk_robust_scaling_n5000_2M_summary.json and "
        "gnk_robust_scaling_n5000_2M.png: n=5000 robust standardisation pilot.\n"
        "- gnk_rejection_abc_summary.json: rejection-ABC Gaussian-NPE pilot.\n"
        "- dim_scaling_pilot_kl_by_d.csv: partial dimension-scaling aggregate "
        "from available metrics under res/gnk_dim_scaling/.\n"
        "- stereological_coverage.csv, stereological_bias_by_seed.csv, and "
        "stereological_posterior_overlay.csv: stereological summaries copied "
        "from the 2026-05-13 postprocessing cache.\n"
        "- ma2_b0_kl.csv: stored MA(2) compatible-case KL medians, with "
        "current-reference entries from the latest refreshes where available.\n"
        "- ma2_b0_flow_current_reference_per_seed.csv and "
        "ma2_b0_flow_current_reference_summary.csv: 80-seed MA(2) b0 "
        "flow-NPE current-reference refresh.\n"
        "- ma2_b0_gaussian_current_reference_per_seed.csv: five-seed MA(2) "
        "b0 Gaussian-NPE current-reference audit.\n"
        "- ma2_b0_reference_audit.csv: pilot MA(2) b0 reference-consistency "
        "audit used for seed-level comparison.\n"
        "- ma2_delta1_refresh.csv and ma2_delta1_refresh_audit.csv: exact "
        "compatible MA(2) refresh summaries.\n"
        "- ma2_compatibility_flow.csv and ma2_compatibility_gaussian.csv: "
        "delta0 compatibility sweep inputs.\n"
        "- ma2_posterior_overlay_seed_22.csv and .png: seed-22 MA(2) overlay.\n"
    )
    (DATA / "SOURCES.md").write_text(text)
    print("wrote data/SOURCES.md")


def build_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    def md(s):
        cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))

    def code(s):
        cells.append(nbf.v4.new_code_cell(s.strip("\n")))

    # ---- framing ----------------------------------------------------------
    md(r"""
These are the updated empirical results for the paper, for Chris, David, and
David. This is an interim update. It includes paper-ready results, checks that
are still in progress, and a few diagnostics that I would keep out of the paper
unless they survive follow-up.

The paper asks how large the simulation budget $N$ must be, relative to the
number of observations $n$ and the problem dimension $d$ (summaries plus
parameters), for NPE to recover the posterior. The general smooth-posterior
case gives a dimension dependence that is too costly for the examples here. The
Bernstein-von Mises case gives the practical scaling, $N \gg d^2 n$. I have
kept that framing in the plots below.

The empirical setup compares two NPE families. Flow-NPE uses a normalising-flow
conditional density estimator. Gaussian-NPE uses a single conditional Gaussian
(mean and covariance both neural-parametrised), which is the $k=1$ case of the
Gaussian-mixture-of-experts family the paper's appendix analyses. In the BvM
regime the posterior is asymptotically Gaussian, so Gaussian-NPE has the
matching inductive bias, and a residual gap to the reference reads as finite-$N$
approximation error rather than a family mismatch. This is the framing I am
using to interpret the GNK and MA(2) results below.
""")

    code(r"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display

DATA = Path("data")
plt.rcParams.update({
    "figure.dpi": 110, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
})
C_FLOW, C_GAUSS, C_REF = "#1f77b4", "#d62728", "#444444"
C_BSL, C_ROBUST, C_REJ = "#2ca02c", "#9467bd", "#ff7f0e"
C_DELTA1, C_TIMEOUT = "#2ca02c", "#7f7f7f"
BUDGETS = ["N=n", "N=n log(n)", "N=n^(3/2)", "N=n^2"]
BUDGET_MATH = ["n", r"n\log n", r"n^{3/2}", r"n^2"]
N_OBS = [100, 500, 1000, 5000]
METHOD_NAME = {"flow_npe": "flow-NPE", "gaussian_npe": "Gaussian-NPE"}
GNK_TRUE = {"A": 3.0, "B": 1.0, "g": 2.0, "k": 0.5}
PARAM_MATH = {"A": "A", "B": "B", "g": "g", "k": "k",
              "lambda": r"\lambda", "sigma": r"\sigma", "xi": r"\xi"}
""")

    # ---- stereological ----------------------------------------------------
    md(r"""
# Stereological

The stereological model stays first because it is the motivating example in the
paper. It has three parameters $(\lambda, \sigma, \xi)$ and four summaries, so
$d=7$. The reference is SMC-ABC. With this low dimension, the budget asked for
by the $d^2 n$ scaling is modest, and $N$ at or above $n^{3/2}$ is already
enough for good coverage.
""")

    md(r"""
**Coverage.** Monte Carlo coverage of the 95% credible intervals, computed over
100 simulated datasets and averaged over the three parameters. The
$n=5000,\ N=n^2$ cell is not run because the simulation cost is too high.
""")

    code(r"""
cov = pd.read_csv(DATA / "stereological_coverage.csv")
cov = cov[cov.N_label.isin(BUDGETS)].copy()
cov["method"] = cov["method"].map(METHOD_NAME)
cov_avg = (cov.groupby(["method", "N_label", "n"])["coverage_95_mean"]
              .mean().reset_index())
cov_avg["N_label"] = pd.Categorical(cov_avg["N_label"], categories=BUDGETS,
                                    ordered=True)
cov_table = cov_avg.pivot_table(index=["method", "N_label"], columns="n",
                                values="coverage_95_mean", observed=True)
cov_table = cov_table.sort_index()[N_OBS].round(3)
cov_table = cov_table.where(cov_table.notna(), "")
cov_table.index.names = ["method", "budget"]
cov_table.columns.name = "n"
cov_table
""")

    md(r"""
**Posterior-mean bias.** Boxplots of the per-seed posterior-mean bias, by
parameter and observation count, with flow-NPE and Gaussian-NPE side by side at
each budget. The dashed line is zero bias.
""")

    code(r"""
bias = pd.read_csv(DATA / "stereological_bias_by_seed.csv")
bias = bias[bias.N_label.isin(BUDGETS)]
params = ["lambda", "sigma", "xi"]
fig, axes = plt.subplots(3, 4, figsize=(13, 8.2))
for r, param in enumerate(params):
    for c, n in enumerate(N_OBS):
        ax = axes[r, c]
        for b, bud in enumerate(BUDGETS):
            for k, (raw, color) in enumerate([("flow_npe", C_FLOW),
                                              ("gaussian_npe", C_GAUSS)]):
                vals = bias[(bias["param"] == param) & (bias.n == n)
                            & (bias.N_label == bud)
                            & (bias.method == raw)]["mean_bias"].dropna().values
                if len(vals) == 0:
                    continue
                bp = ax.boxplot([vals], positions=[b * 2.6 + k], widths=0.9,
                                patch_artist=True,
                                flierprops=dict(marker=".", markersize=2,
                                                markerfacecolor=color,
                                                markeredgecolor=color))
                for patch in bp["boxes"]:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.55)
                for med in bp["medians"]:
                    med.set_color("black")
        ax.axhline(0, color="black", ls="--", lw=0.8, alpha=0.5)
        ax.set_xticks([b * 2.6 + 0.5 for b in range(4)])
        ax.set_xticklabels([f"${b}$" for b in BUDGET_MATH], fontsize=8)
        if r == 0:
            ax.set_title(f"$n = {n}$")
        if c == 0:
            ax.set_ylabel(f"${PARAM_MATH[param]}$\nposterior-mean bias")
axes[0, 0].plot([], [], "s", color=C_FLOW, label="flow-NPE")
axes[0, 0].plot([], [], "s", color=C_GAUSS, label="Gaussian-NPE")
axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
plt.show()
""")

    md(r"""
**Posterior overlay for $\lambda$** (a single simulated dataset, $n=1000$,
seed 1). The NPE posterior at each budget is shown against the SMC-ABC
benchmark. The dashed line is the true $\lambda$.
""")

    code(r"""
ster_ov = pd.read_csv(DATA / "stereological_posterior_overlay.csv")
lam = ster_ov[ster_ov.param == "lambda"]
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharex=True)
specs = [("flow_npe", "flow-NPE", plt.cm.Blues),
         ("gaussian_npe", "Gaussian-NPE", plt.cm.Reds)]
for ax, (raw, mname, cmap) in zip(axes, specs):
    shades = cmap(np.linspace(0.4, 0.95, len(BUDGETS)))
    for bud, shade, lab in zip(BUDGETS, shades, BUDGET_MATH):
        s = lam[(lam.method == raw) & (lam.N_label == bud)].sort_values("x")
        ax.plot(s.x, s.density, color=shade, label=f"$N={lab}$")
    a = lam[lam.method == "abc_smc"].sort_values("x")
    ax.plot(a.x, a.density, color="black", lw=1.4, label="SMC-ABC")
    ax.axvline(100.0, color="black", ls="--", lw=1, alpha=0.6)
    ax.set_title(mname)
    ax.set_xlabel(r"$\lambda$")
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=8)
axes[0].set_ylabel("posterior density")
fig.tight_layout()
plt.show()
""")

    md(r"""
Coverage is near or above the nominal level. Bias falls as the budget grows for
both methods. flow-NPE reaches small bias at a lower budget than Gaussian-NPE,
which still shows visible $\sigma$ and $\xi$ bias at smaller budgets for the
larger sample sizes. In the $\lambda$ overlay both methods concentrate onto the
true value as $N$ grows.

A note on the reference. The stereological reference is the SMC-ABC posterior. I
am considering an upgrade for the next iteration, exploiting the exact Poisson
count factorisation to condition $\lambda$ on $K$ analytically (truncated
Gamma), then run SMC-ABC only on $(\sigma, \xi)$. Together with a
Mahalanobis-distance schedule, a tighter tolerance ladder, and at least three
independent replicates per cell, this would give a stronger reference for the
KL claim. Not done yet.
""")

    # ---- g-and-k ----------------------------------------------------------
    md(r"""
# g-and-k

The g-and-k model has four parameters $(A, B, g, k)$ and is summarised by seven
octiles, so $d=11$. The reference posterior is from NUTS, conditioned on the
same seven octiles. This section now separates the paper-grid result from the
diagnostics prompted by the concern that the GNK NPE posteriors were not close
enough to the reference at the headline cell.
""")

    md(r"""
**KL from the reference posterior.** Median over seeds, shown as
flow-NPE / Gaussian-NPE, across observation count $n$ and budget $N$. Lower is
better. The moment-matched Gaussian row measures how close the reference
posterior is to Gaussian.
""")

    code(r"""
gnk = pd.read_csv(DATA / "gnk_kl_flow_vs_gaussian.csv")
grid = gnk[gnk.standard_paper_grid == 1].copy()
grid["entry"] = grid.apply(
    lambda r: f"{r.flow_theta_kl_median:.2f} / {r.gaussian_theta_kl_median:.2f}",
    axis=1)
kl_table = grid.pivot(index="N_label", columns="n", values="entry")
kl_table = kl_table.reindex(BUDGETS)[N_OBS]

oracle = pd.read_csv(DATA / "gnk_theta_oracle_by_n.csv").set_index("n")
kl_table.loc["moment-matched Gaussian"] = [
    f"{oracle.loc[n, 'K_theta_star_median']:.4f}" for n in N_OBS]
kl_table.index.name = "budget"
kl_table.columns.name = "n"
display(kl_table)

ci = pd.read_csv(DATA / "gnk_paper_grid_bootstrap_ci.csv")
ci = ci[ci.N_label.isin(BUDGETS)].copy()
ci["budget"] = pd.Categorical(ci.N_label, categories=BUDGETS, ordered=True)
ci["flow median (90% CI)"] = ci.apply(
    lambda r: f"{r.flow_median:.2f} [{r.flow_ci05:.2f}, {r.flow_ci95:.2f}]",
    axis=1)
ci["Gaussian median (90% CI)"] = ci.apply(
    lambda r: f"{r.gaussian_median:.2f} [{r.gaussian_ci05:.2f}, {r.gaussian_ci95:.2f}]",
    axis=1)
ci["paired Gaussian-flow median (90% CI)"] = ci.apply(
    lambda r: (f"{r.gaussian_minus_flow_median:.2f} "
               f"[{r.gaussian_minus_flow_ci05:.2f}, "
               f"{r.gaussian_minus_flow_ci95:.2f}]"),
    axis=1)
ci = ci.sort_values(["n", "budget"]).reset_index(drop=True)
ci_table = ci[["n", "N_label", "seed_count", "flow median (90% CI)",
               "Gaussian median (90% CI)",
               "paired Gaussian-flow median (90% CI)"]]
ci_table
""")

    md(r"""
The bootstrap intervals above use paired seed resampling. The paired difference
is useful because the two estimators share the same simulated observation at a
given seed.
""")

    md(r"""
**KL versus simulation budget.** Median over seeds, with the interquartile
band, one panel per $n$.
""")

    code(r"""
fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.3), sharey=True)
for ax, n in zip(axes, N_OBS):
    sub = grid[grid.n == n].sort_values("N")
    ax.plot(sub.N, sub.flow_theta_kl_median, "o-", color=C_FLOW, label="flow-NPE")
    ax.fill_between(sub.N, sub.flow_theta_kl_q25, sub.flow_theta_kl_q75,
                    color=C_FLOW, alpha=0.15)
    ax.plot(sub.N, sub.gaussian_theta_kl_median, "s-", color=C_GAUSS,
            label="Gaussian-NPE")
    ax.fill_between(sub.N, sub.gaussian_theta_kl_q25, sub.gaussian_theta_kl_q75,
                    color=C_GAUSS, alpha=0.15)
    ax.set_xscale("log")
    ax.set_title(f"$n = {n}$")
    ax.set_xlabel(r"simulation budget $N$")
axes[0].set_ylabel("KL from reference posterior")
axes[0].legend(frameon=False)
fig.tight_layout()
plt.show()
""")

    md(r"""
**Octile versus hexadecile summaries.** The same g-and-k experiment with seven
octile summaries ($d_s=7$) and 15 hexadecile summaries ($d_s=15$), for the two
sample sizes where the hexadecile runs exist. Only the summary dimension
changes.
""")

    code(r"""
oct_df = grid
hex_g = pd.read_csv(DATA / "gnk_hexadecile_gaussian.csv")
hex_f = pd.read_csv(DATA / "gnk_hexadecile_flow.csv")
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for ax, n in zip(axes, [100, 1000]):
    o = oct_df[oct_df.n == n].sort_values("N")
    ax.plot(o.N, o.gaussian_theta_kl_median, "o-", color=C_GAUSS,
            label=r"Gaussian octiles ($d_s=7$)")
    h = hex_g[hex_g.n_obs == n].sort_values("n_sims")
    ax.plot(h.n_sims, h.finite_kl_median, "s--", color=C_GAUSS,
            label=r"Gaussian hexadeciles ($d_s=15$)")
    hf = hex_f[hex_f.n_obs == n].sort_values("n_sims")
    if not hf.empty:
        ax.plot(hf.n_sims, hf.kl_median, "o--", color=C_FLOW,
                label=r"flow hexadeciles ($d_s=15$)")
    ax.set_xscale("log")
    ax.set_title(f"$n = {n}$")
    ax.set_xlabel(r"simulation budget $N$")
    ax.legend(frameon=False, fontsize=8)
axes[0].set_ylabel("KL from reference posterior")
fig.tight_layout()
plt.show()
""")

    md(r"""
**Posterior overlay across the scaling grid.** Each row is one $(n,N)$ cell,
with $n$ and $N$ increasing down the rows, and the columns are the four
parameters. Dashed lines mark the true values. The $n=1000$ rows use seed 36
and the $n=5000$ rows use seed 50.
""")

    code(r"""
gnk_ov = pd.read_csv(DATA / "gnk_posterior_overlay.csv")
cells = ["n=1000, N=n^(3/2)", "n=1000, N=n^2",
         "n=5000, N=n^(3/2)", "n=5000, N=n^2"]
gnk_params = ["A", "B", "g", "k"]
overlay = [
    ("Reference (flow convention)", "#777777", "-", "reference, flow obs."),
    ("Reference (Gaussian convention)", "#222222", "--", "reference, Gaussian obs."),
    ("flow-NPE", C_FLOW, "-", "flow-NPE"),
    ("Gaussian-NPE", C_GAUSS, "-", "Gaussian-NPE"),
]
fig, axes = plt.subplots(4, 4, figsize=(12, 10.5))
for i, cell in enumerate(cells):
    for j, param in enumerate(gnk_params):
        ax = axes[i, j]
        for method, color, ls, label in overlay:
            s = gnk_ov[(gnk_ov.cell == cell) & (gnk_ov.method == method)
                       & (gnk_ov.param == param)].sort_values("x")
            if not s.empty:
                ax.plot(s.x, s.density, color=color, linestyle=ls, label=label)
        ax.axvline(GNK_TRUE[param], color="black", ls=":", lw=1, alpha=0.6)
        ax.set_yticks([])
        if i == 0:
            ax.set_title(f"${PARAM_MATH[param]}$")
        if i == 3:
            ax.set_xlabel(f"${PARAM_MATH[param]}$")
        if j == 0:
            ax.set_ylabel(cell, fontsize=9)
axes[0, 0].legend(frameon=False, fontsize=7)
fig.tight_layout()
plt.show()
""")

    md(r"""
**BSL diagnostic at $n=1000$, seed 0.** Chris suggested running BSL against the
GNK reference as a cross-check, since BSL uses model simulations rather than the
asymptotic summary likelihood that NUTS conditions on. At $n=1000$, seed 0, the
BSL marginal medians and standard deviations agree with the NUTS reference
within rounding, and the KL between BSL and NUTS is around 0.025 in both
directions after removing duplicate retained rows from the random-walk chain.
This supports the asymptotic summary likelihood as a reasonable stand-in for
the simulator-based partial posterior at this cell.
""")

    code(r"""
bsl = pd.read_csv(DATA / "gnk_bsl_diagnostic.csv")
bsl_table = bsl[["parameter", "NUTS median", "NUTS sd", "BSL median",
                 "BSL sd", "R-hat", "ESS", "acceptance rate"]].copy()
bsl_table[["NUTS median", "NUTS sd", "BSL median", "BSL sd",
           "R-hat", "acceptance rate"]] = (
    bsl_table[["NUTS median", "NUTS sd", "BSL median", "BSL sd",
               "R-hat", "acceptance rate"]].round(4)
)
bsl_table["ESS"] = bsl_table["ESS"].round(1)
display(bsl_table)
print("KL(BSL unique rows || NUTS):",
      round(float(bsl["KL_BSL_unique_to_NUTS"].iloc[0]), 4))
print("KL(NUTS || BSL unique rows):",
      round(float(bsl["KL_NUTS_to_BSL_unique"].iloc[0]), 4))
""")

    md(r"""
**Robust standardisation diagnostic.** One thing I noticed in the GNK paper-grid
Gaussian-NPE results is that the prior-predictive simulated summaries span
several orders of magnitude under the broad $U(0,10)^4$ prior. The default
z-score standardisation puts most of the numerical scale on a few extreme
summaries and squashes the region around the observed summary. As a diagnostic,
I replaced z-score with asinh plus median/IQR. At $n=1000$, seed 0, this
dropped the joint 4-D KL from 4.70 to 1.37 nats without other changes. At
$n=5000$, seed 50, an early-stopping pilot at $N=2M$ reached KL 1.88, lower
than the same-seed standard z-score result at $N=25M$ (2.72). The shift is
consistent across parameters. The largest marginal median shift at $n=5000$ is
in $B$ at about 0.3 posterior-sd units.

I want to be careful with the framing here. This is an internal diagnostic on
the training pipeline, not a paper claim. The theory does not say anything
about standardisation, so I am treating this as a finite-$N$ implementation
question, not a method contribution.
""")

    code(r"""
with open(DATA / "gnk_robust_scaling_summary.json") as fh:
    robust1000 = json.load(fh)
with open(DATA / "gnk_robust_scaling_n5000_2M_summary.json") as fh:
    robust5000 = json.load(fh)

robust_rows = [
    {
        "cell": "n=1000, seed=0",
        "standard z-score KL": robust1000["metrics"]["standard_metrics_v3"]["kl_value"],
        "robust KL": robust1000["metrics"]["robust_metrics"]["kl_value"],
        "BSL KL": robust1000["metrics"]["bsl"]["kl_value"],
        "max median shift": robust1000["median_shifts_in_nuts_sd"]["Gaussian-NPE (robust asinh)"]["max_abs"],
    },
    {
        "cell": "n=5000, seed=50",
        "standard z-score KL": robust5000.get("standard_zscore_kl_value"),
        "robust KL": robust5000.get("kl_value"),
        "BSL KL": np.nan,
        "max median shift": robust5000.get("marginal_shifts_in_nuts_sd", {}).get("max_abs"),
    },
]
display(pd.DataFrame(robust_rows).round(3))

ov = pd.read_csv(DATA / "gnk_robust_scaling_overlay.csv")
fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.1), sharey=False)
colors = {"NUTS reference": C_REF, "BSL": C_BSL,
          "Gaussian-NPE (z-score)": C_GAUSS,
          "Gaussian-NPE (robust asinh)": C_ROBUST}
for ax, param in zip(axes, ["A", "B", "g", "k"]):
    for method, color in colors.items():
        s = ov[(ov.param == param) & (ov.method == method)].sort_values("x")
        ax.plot(s.x, s.density, color=color, label=method)
    ax.set_title(param)
    ax.set_yticks([])
axes[0].set_ylabel("density")
axes[0].legend(frameon=False, fontsize=7)
fig.suptitle("GNK robust standardisation, n=1000 seed=0", y=1.04)
fig.tight_layout()
plt.show()

if (DATA / "gnk_robust_scaling_n5000_2M.png").exists():
    display(Image(filename=str(DATA / "gnk_robust_scaling_n5000_2M.png"), width=780))
""")

    md(r"""
**Rejection ABC pilot.** A second diagnostic checks the same hypothesis. If
extreme prior-predictive summaries are diluting Gaussian-NPE training, then
filtering the training set to keep only simulations whose summaries are close
to the observed summary should reach a similar improvement, regardless of how
the inputs are standardised. I ran rejection ABC over a 10M prior-predictive
pool at $n=1000$, seed 0, kept the top 1% by distance to the observed summary,
then trained Gaussian-NPE on the kept 100k pairs with standard z-score
standardisation. KL dropped from 4.70 to 0.81, closing 83% of the gap to BSL at
the same cell, slightly more than the 71% closed by the robust standardisation
diagnostic. Two routes point to the same diagnosis. The residual gap to BSL is
still about 30x in KL, so this is not at the floor. This is also an internal
diagnostic, not in the paper.
""")

    code(r"""
with open(DATA / "gnk_rejection_abc_summary.json") as fh:
    rej = json.load(fh)

if rej.get("status") == "pending":
    print("Rejection ABC pilot pending.")
else:
    comp = pd.DataFrame([
        {"pipeline": "Vanilla Gaussian-NPE, standard z-score",
         "KL": rej["vanilla_canonical_kl"],
         "training set": "1M",
         "simulator budget": "1M"},
        {"pipeline": "Robust-scaling Gaussian-NPE (asinh + median/IQR)",
         "KL": rej["robust_canonical_kl"],
         "training set": "1M",
         "simulator budget": "1M"},
        {"pipeline": "Rejection ABC + standard z-score",
         "KL": rej["kl_reference_to_npe_value"],
         "training set": "100k accepted",
         "simulator budget": "10M (1% kept)"},
        {"pipeline": "BSL (dense proposal, M=500)",
         "KL": rej["bsl_unique_kl"],
         "training set": "",
         "simulator budget": ""},
    ])
    display(comp.round({"KL": 3}))
    wall = pd.DataFrame([{
        "total seconds": rej["runtime_seconds"],
        "simulation seconds": rej["simulation_seconds"],
        "selection seconds": rej["selection_seconds"],
        "training seconds": rej["training_seconds"],
        "epochs": rej["train_epochs"],
        "stop reason": rej["stop_reason"],
        "max marginal shift": rej["max_marginal_shift"],
        "shift parameter": rej["max_shift_parameter"],
    }])
    display(wall.round(3))
""")

    md(r"""
**Dimension-scaling pilot.** The g-and-k octile ($d_s=7$) versus hexadecile
($d_s=15$) comparison above is a two-point version of the $d^2$ scaling. I have
a pilot running that extends this to a wider range of $d_s$ at fixed $n$,
holding $N/(d^2 n)$ constant. The aim is to test directly whether KL is
approximately stable across $d$ under the predicted scaling. The pilot is
partial as of this writing. The numbers below are preliminary, 187 of 300 cells
have finite metrics in the local aggregate, and 12 finished cells without
metrics still need retry or diagnosis. I am flagging this here because if it
lands cleanly, the dim-scaling result is probably the strongest single
empirical anchor for the paper's headline $N \gg d^2 n$ result.
""")

    code(r"""
dim = pd.read_csv(DATA / "dim_scaling_pilot_kl_by_d.csv")
finite = dim[(dim.status == "finite_metric") & np.isfinite(dim.theta_kl)].copy()
status_counts = dim.status.value_counts().rename_axis("status").reset_index(name="rows")
display(status_counts)

dim_summary = (finite.groupby(["method", "d_s", "d", "N_sims", "N_over_d2n"])
                    .theta_kl
                    .agg(seed_count="count", median="median",
                         q25=lambda s: np.quantile(s, 0.25),
                         q75=lambda s: np.quantile(s, 0.75))
                    .reset_index())
dim_summary["method_label"] = dim_summary.method.map(METHOD_NAME)
dim_summary["entry"] = dim_summary.apply(
    lambda r: f"{r['median']:.2f} ({int(r.seed_count)})", axis=1)
dim_table = dim_summary.pivot(index="method_label", columns="d_s",
                              values="entry")
dim_table.index.name = "method"
dim_table.columns.name = "summary dimension"
display(dim_table)

fig, ax = plt.subplots(figsize=(7.2, 4.0))
for method, color, label in [("flow_npe", C_FLOW, "flow-NPE"),
                             ("gaussian_npe", C_GAUSS, "Gaussian-NPE")]:
    s = dim_summary[dim_summary.method == method].sort_values("d_s")
    ax.plot(s.d_s, s["median"], "o-", color=color, label=label)
    ax.fill_between(s.d_s, s.q25, s.q75, color=color, alpha=0.15)
    for _, row in s.iterrows():
        ax.text(row.d_s, row["median"], f"{int(row.seed_count)}",
                fontsize=7, ha="center", va="bottom", color=color)
ax.set_xlabel(r"summary dimension $d_s$")
ax.set_ylabel("KL from reference posterior")
ax.set_title(r"GNK dim-scaling pilot, fixed $N/(d^2 n)=5$")
ax.legend(frameon=False)
fig.tight_layout()
plt.show()
""")

    md(r"""
The main GNK story is now clearer. The paper-grid result still supports the
$N$-scaling claim, but the poor headline Gaussian-NPE fit is not simply a
failure of the Gaussian posterior family. BSL is close to the NUTS reference,
and both robust standardisation and rejection filtering move Gaussian-NPE much
closer to that target. I would present those two checks as internal diagnostics
for now, then decide later whether a short appendix note is useful.
""")

    # ---- MA(2) ------------------------------------------------------------
    md(r"""
# MA(2)

Before the MA(2) results, a correction. I found that the NUTS reference samples
I had been comparing against for the compatible MA(2) b0 cell were generated
under the older diagonal independent summary likelihood, while the current
driver uses the joint-covariance MVN reference (analytic autocovariance mean,
Bartlett covariance for the sample autocovariances). These are different
references. Under fresh joint-covariance NUTS for all 80 flow seeds at this
cell, the flow median KL drops from 0.667 to 0.019, with q75 0.035 and max
0.131. The headline I was previously carrying ("flow does not converge on MA(2)
compatible") is therefore not supported by current-reference data, and I am
rewriting this section accordingly. The same reference convention applies to
the $\delta_0$ sweep (verified separately), so those numbers are unchanged.

All neural estimators here condition on the three sample autocovariances. So I
compare them to a summary-based reference, not to the exact full-data posterior,
which conditions on information the neural estimators do not see. The reference
is the posterior under a Gaussian summary likelihood with analytic mean (the
population autocovariances at $\theta$) and Bartlett joint covariance for the
sample autocovariances. This is an asymptotic summary-likelihood reference, not
the exact finite-sample posterior for the summaries. For $n \ge 1000$ the
asymptotic and finite-sample summary moments agree closely (max relative
covariance error under 0.5%, mean bias under 0.02 summary-sd units), so the
reference is usable.
""")

    code(r"""
ma2_audit = pd.read_csv(DATA / "ma2_b0_reference_audit.csv")
flow_current = pd.read_csv(DATA / "ma2_b0_flow_current_reference_per_seed.csv")
flow_current_summary = pd.read_csv(DATA / "ma2_b0_flow_current_reference_summary.csv")
gauss_current = pd.read_csv(DATA / "ma2_b0_gaussian_current_reference_per_seed.csv")

flow_summary_table = flow_current_summary[
    flow_current_summary.metric.isin(["stored_flow_kl", "current_flow_kl"])
].copy()
display(flow_summary_table[["metric", "n", "median", "q75", "max"]].round(4))

audit_cols = ["seed", "flow_kl_stored", "flow_kl_current",
              "gauss_kl_stored", "gauss_kl_current", "ref_mean_shift_l2"]
display(ma2_audit[audit_cols].round(4))
""")

    md(r"""
**Post-timeout status.** The training-pipeline completion issue is separate
from the reference audit above. The exact compatible refresh at $\delta_0=1$
produced complete files for every Gaussian-NPE row, while the largest flow-NPE
cell timed out after producing setup artifacts and no metrics. The 25M flow row
is therefore an operational timeout, not a scientific KL value, and it is not
needed for the current paper-facing MA(2) claim. For Gaussian-NPE, finite KL
medians and `Inf` KL counts are reported separately. MMD stays finite across
the completed rows.
""")

    code(r"""
delta1 = pd.read_csv(DATA / "ma2_delta1_refresh.csv")
delta1_audit = pd.read_csv(DATA / "ma2_delta1_refresh_audit.csv")
delta1["budget"] = pd.Categorical(delta1["budget_label"], categories=BUDGETS,
                                  ordered=True)

status_cols = [
    "status_complete_rows", "nonfinite_kl_status_rows", "missing_rows",
    "timed_out_rows", "partial_training_only_rows", "bad_shape_rows",
]
for col in status_cols:
    if col not in delta1.columns:
        delta1[col] = 0

status_overview = (
    delta1_audit.groupby(["method", "status"]).size()
    .unstack(fill_value=0)
    .reindex(["flow_npe", "gaussian_npe"])
)
status_overview.index = status_overview.index.map(METHOD_NAME)
display(status_overview)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
for ax, method in zip(axes, ["flow_npe", "gaussian_npe"]):
    s = delta1[delta1.method == method].copy()
    mat = s.pivot(index="n_obs", columns="budget", values="complete_rows")
    mat = mat.reindex(index=N_OBS, columns=BUDGETS).fillna(0)
    im = ax.imshow(mat.values, vmin=0, vmax=101, cmap="Greens", aspect="auto")
    ax.set_title(METHOD_NAME[method])
    ax.set_xticks(range(len(BUDGETS)), BUDGET_MATH, rotation=30, ha="right")
    ax.set_yticks(range(len(N_OBS)), [str(n) for n in N_OBS])
    ax.set_xlabel(r"budget $N$")
    if ax is axes[0]:
        ax.set_ylabel(r"$n$")
    for i, n in enumerate(N_OBS):
        for j, budget in enumerate(BUDGETS):
            row = s[(s.n_obs == n) & (s.budget_label == budget)]
            if row.empty:
                label = "0/101\nmissing"
            else:
                row = row.iloc[0]
                label = f"{int(row.complete_rows)}/101"
                extra = []
                if int(row.nonfinite_kl_status_rows):
                    extra.append(f"Inf KL {int(row.infinite_kl_rows)}")
                if int(row.partial_training_only_rows):
                    extra.append("partial")
                if int(row.timed_out_rows):
                    extra.append("timeout")
                if int(row.missing_rows):
                    extra.append(f"missing {int(row.missing_rows)}")
                if extra:
                    label += "\n" + ", ".join(extra[:2])
            ax.text(j, i, label, ha="center", va="center", fontsize=7,
                    color="black")
fig.colorbar(im, ax=axes, shrink=0.8, label="complete file/shape rows")
plt.show()
""")

    md(r"""
**Compatible-case current-reference results.** Flow-NPE now has 80 seeds under
the current joint-covariance reference. Gaussian-NPE has five audit seeds (7,
22, 58, 63, 69). That coverage is asymmetric. A full Gaussian-NPE refresh would
take about five hours locally at roughly 300 seconds per rerun and is queued as
a follow-up. At the audit seeds, flow-NPE has lower current-reference KL than
Gaussian-NPE for all five seeds, but both are small.
""")

    code(r"""
ma2kl = pd.read_csv(DATA / "ma2_b0_kl.csv")
target = ma2kl[(ma2kl.n_obs == 1000) & (ma2kl.n_sims == 1000000)].copy()
before_after = []
for method in ["flow_npe", "gaussian_npe"]:
    row = target[target.method == method].iloc[0]
    current = row.current_reference_kl_median
    n_current = int(row.current_reference_seed_count)
    if method == "flow_npe":
        implication = "converges cleanly, old flow stalls headline was a reference artefact"
    else:
        implication = "converges, small change from stored, but only five audit seeds"
    before_after.append({
        "method": METHOD_NAME[method],
        "n_obs": int(row.n_obs),
        "n_sims": int(row.n_sims),
        "stored median KL": row.kl_median,
        "current-reference median KL": current,
        "seeds covered": n_current,
        "implication": implication,
    })
display(pd.DataFrame(before_after).round(3))

flow_current = pd.read_csv(DATA / "ma2_b0_flow_current_reference_per_seed.csv")
gauss_current = pd.read_csv(DATA / "ma2_b0_gaussian_current_reference_per_seed.csv")
audit_join = gauss_current.merge(
    flow_current[["seed", "current_flow_kl"]],
    on="seed",
    how="left",
)
audit_join = audit_join.rename(columns={
    "gauss_kl_stored": "stored Gaussian KL",
    "gauss_kl_current_recomputed": "current Gaussian KL",
    "current_flow_kl": "current flow KL",
})
display(audit_join[["seed", "current flow KL", "stored Gaussian KL",
                    "current Gaussian KL"]].round(4))

ma2kl_plot = pd.read_csv(DATA / "ma2_b0_kl.csv")
ma2kl_plot = ma2kl_plot.sort_values(["method", "n_obs", "n_sims"])
fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.3), sharey=True)
for ax, n in zip(axes, N_OBS):
    for raw, color, mlabel in [("flow_npe", C_FLOW, "flow-NPE, stored ref."),
                               ("gaussian_npe", C_GAUSS, "Gaussian-NPE, stored ref.")]:
        s = ma2kl_plot[(ma2kl_plot.method == raw) & (ma2kl_plot.n_obs == n)]
        s = s.sort_values("n_sims")
        done = s[(s.n_seeds > 0) & s.kl_median.notna()]
        ax.plot(done.n_sims, done.kl_median, "o-", color=color, alpha=0.55,
                label=mlabel)
        audited = done[(done.current_reference_seed_count > 0)
                       & done.current_reference_kl_median.notna()]
        if not audited.empty:
            ax.plot(audited.n_sims, audited.current_reference_kl_median,
                    "s", color=color, markersize=8,
                    label=METHOD_NAME[raw] + ", current audit")
    ax.set_xscale("log")
    ax.set_title(f"$n = {n}$")
    ax.set_xlabel(r"simulation budget $N$")
axes[0].set_ylabel("KL from reference posterior")
axes[0].legend(frameon=False, fontsize=7)
fig.tight_layout()
plt.show()
""")

    md(r"""
**Exact compatible refresh, KL and MMD versus budget.** This uses the
$\delta_0=1$ refresh outputs. It is a completion and stability view, not a
replacement for the b0 reference audit above. Cells with no finite KL or MMD
are explicitly marked at their budget position.
""")

    code(r"""
fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.2), sharex="col")
for col, n in enumerate(N_OBS):
    for method, color, label in [("flow_npe", C_FLOW, "flow-NPE"),
                                 ("gaussian_npe", C_GAUSS, "Gaussian-NPE")]:
        s = delta1[(delta1.method == method) & (delta1.n_obs == n)].sort_values("n_sims")
        kl_done = s[(s.finite_kl_rows > 0) & s.finite_kl_median.notna()]
        mmd_done = s[(s.finite_mmd_rows > 0) & s.mmd_median.notna()]
        axes[0, col].plot(kl_done.n_sims, kl_done.finite_kl_median, "o-",
                          color=color, label=label)
        axes[1, col].plot(mmd_done.n_sims, mmd_done.mmd_median, "o-",
                          color=color, label=label)
        missing = s[s.finite_kl_rows == 0]
        for _, row in missing.iterrows():
            axes[0, col].text(row.n_sims, 0.95, "no\nmetric", color=color,
                              fontsize=7, rotation=90, va="top", ha="right",
                              transform=axes[0, col].get_xaxis_transform())
            axes[1, col].text(row.n_sims, 0.95, "no\nmetric", color=color,
                              fontsize=7, rotation=90, va="top", ha="right",
                              transform=axes[1, col].get_xaxis_transform())
        inf_kl = s[s.infinite_kl_rows > 0]
        for _, row in inf_kl.iterrows():
            axes[0, col].annotate(f"Inf KL {int(row.infinite_kl_rows)}",
                                  xy=(row.n_sims, row.finite_kl_median),
                                  xytext=(0, 8), textcoords="offset points",
                                  color=color, fontsize=7, ha="center")
    axes[0, col].set_title(f"$n = {n}$")
    axes[1, col].set_xlabel(r"simulation budget $N$")
    for r in range(2):
        axes[r, col].set_xscale("log")
axes[0, 0].set_ylabel("finite KL median")
axes[1, 0].set_ylabel("MMD median")
axes[0, 0].legend(frameon=False, fontsize=8)
fig.tight_layout()
plt.show()
""")

    md(r"""
**Near-compatible lower-boundary summary perturbation.** The $\delta_0$ sweep
is the David Warne meeting follow-up. The meeting concern was that flow-NPE
looked worse than Gaussian-NPE under a near-compatible lower-boundary summary
perturbation. With the staged aggregate below, that exact reading is not
supported. At $\delta_0=0.99$, flow-NPE KL decreases with budget, from about
0.37 at $N=n$ to about 0.012 at $N=n^2$. Gaussian-NPE also decreases, from
about 0.26 to about 0.041. Gaussian-NPE is lower at the two smaller budgets,
while flow-NPE is lower at the two larger budgets.

The reference check still matters. I checked separately that the $\delta_0$
reference convention is the current joint-covariance MVN summary likelihood,
not the older diagonal independent one that caused the compatible case audit
issue, so these numbers are reliable for this reference question. The Gaussian
sweep currently covers only the two endpoints $\{0.01, 0.99\}$, while the flow
sweep covers $\delta_0 \in \{0.01, 0.1, 0.25, 0.5, 0.75, 0.99\}$. Completing
the Gaussian sweep to the four missing $\delta_0$ values is queued.
""")

    code(r"""
flow_sweep = pd.read_csv(DATA / "ma2_compatibility_flow.csv").copy()
flow_sweep["method"] = "flow_npe"

gauss_raw = pd.read_csv(DATA / "ma2_compatibility_gaussian.csv")
gauss_rows = []
for (delta0, n_obs, n_sims), g in gauss_raw.groupby(["delta0", "n_obs", "n_sims"]):
    vals = g.kl.to_numpy(float)
    if int(n_sims) == 1000:
        budget = "N=n"
    elif int(n_sims) == 6907:
        budget = "N=n log(n)"
    elif int(n_sims) == 31622:
        budget = "N=n^(3/2)"
    else:
        budget = "N=n^2"
    gauss_rows.append({
        "n_obs": int(n_obs),
        "n_sims": int(n_sims),
        "budget_label": budget,
        "delta0": float(delta0),
        "rows": len(vals),
        "finite_kl_rows": int(np.isfinite(vals).sum()),
        "infinite_kl_rows": int(np.isinf(vals).sum()),
        "finite_kl_median": float(np.nanmedian(vals[np.isfinite(vals)])),
        "mmd_median": np.nan,
        "method": "gaussian_npe",
    })
gauss_sweep = pd.DataFrame(gauss_rows)

sweep = pd.concat([flow_sweep, gauss_sweep], ignore_index=True, sort=False)
sweep = sweep[(sweep.n_obs == 1000) & sweep.budget_label.isin(BUDGETS)].copy()
sweep["budget"] = pd.Categorical(sweep.budget_label, categories=BUDGETS,
                                 ordered=True)

mild = sweep[np.isclose(sweep.delta0, 0.99)].copy()
mild_table = mild.pivot_table(index="budget", columns="method",
                              values="finite_kl_median", observed=True)
mild_table = mild_table.reindex(BUDGETS)
mild_table.columns = [METHOD_NAME[c] for c in mild_table.columns]
display(mild_table.round(3))

fig, ax = plt.subplots(figsize=(7.2, 4.0))
for method, color, marker in [("flow_npe", C_FLOW, "o"),
                              ("gaussian_npe", C_GAUSS, "s")]:
    for budget, ls in zip(BUDGETS, ["-", "--", "-.", ":"]):
        s = sweep[(sweep.method == method) & (sweep.budget_label == budget)]
        s = s.sort_values("delta0")
        if s.empty:
            continue
        label = f"{METHOD_NAME[method]}, {budget}"
        ax.plot(s.delta0, s.finite_kl_median, marker=marker, linestyle=ls,
                color=color, alpha=0.75, label=label)
ax.set_xlabel(r"forced summary perturbation $\delta_0$")
ax.set_ylabel("finite KL median")
ax.set_title(r"MA(2) $\delta_0$ sweep, joint-covariance summary reference")
ax.legend(frameon=False, fontsize=7, ncol=2)
fig.tight_layout()
plt.show()
""")

    md(r"""
**Severe incompatibility, $\delta_0=0.01$.** The major-incompatibility rows are
still useful as a failure-mode check. Increasing $N$ does not repair a target
that is far outside the model-compatible summary region. This is a different
question from the compatible-case reference audit.
""")

    code(r"""
severe = sweep[np.isclose(sweep.delta0, 0.01)].copy()
sev_table = severe.pivot_table(index="budget", columns="method",
                               values="finite_kl_median", observed=True)
sev_table = sev_table.reindex(BUDGETS)
sev_table.columns = [METHOD_NAME[c] for c in sev_table.columns]
display(sev_table.round(3))

fig, ax = plt.subplots(figsize=(6.6, 3.8))
for method, color, label in [("flow_npe", C_FLOW, "flow-NPE"),
                             ("gaussian_npe", C_GAUSS, "Gaussian-NPE")]:
    s = severe[severe.method == method].sort_values("n_sims")
    ax.plot(s.n_sims, s.finite_kl_median, "o-", color=color, label=label)
ax.set_xscale("log")
ax.set_xlabel(r"simulation budget $N$")
ax.set_ylabel("finite KL median")
ax.set_title(r"MA(2), severe incompatibility $\delta_0=0.01$")
ax.legend(frameon=False)
fig.tight_layout()
plt.show()
""")

    md(r"""
**Posterior overlay at seed 22.** Compatible case, $n=1000$, $N=n^2=1M$.
Flow-NPE closely overlaps the current joint-covariance reference. Gaussian-NPE
is close but slightly shifted in $\theta_1$ and $\theta_2$ with a different
joint correlation. Diagnostic KL values from a 2000-draw subsample at this seed
are 0.016 for flow-NPE and 0.115 for Gaussian-NPE against the current
reference. Compare to the stored `kl.txt` values for the same seed, 0.664 for
flow and 0.091 for Gaussian, measured against the older diagonal-independent
reference. The relative ordering flips.
""")

    code(r"""
if (DATA / "ma2_posterior_overlay_seed_22.png").exists():
    display(Image(filename=str(DATA / "ma2_posterior_overlay_seed_22.png"), width=720))
else:
    ma2_ov = pd.read_csv(DATA / "ma2_posterior_overlay_seed_22.csv")
    fig, ax = plt.subplots(figsize=(5, 5))
    for method, color in [("true posterior", C_REF), ("flow-NPE", C_FLOW),
                          ("Gaussian-NPE", C_GAUSS)]:
        s = ma2_ov[ma2_ov.method == method]
        ax.scatter(s.t1, s.t2, s=8, alpha=0.25, color=color, label=method)
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()
""")

    md(r"""
Reading the MA(2) results together. The compatible-case headline I had been
carrying ("flow stalls, Gaussian converges cleanly") is not supported under the
current joint-covariance reference. Both methods converge cleanly on compatible
MA(2) b0. Flow-NPE has a slightly lower median KL than Gaussian-NPE at the five
audit seeds where I have an apples-to-apples comparison, but both are small and
the broader 80-seed flow distribution is also small. Under the checked
$\delta_0=0.99$ endpoint, the current staged aggregate does not support a simple
"flow worse than Gaussian" reading. Gaussian-NPE is lower at the smaller
budgets, flow-NPE is lower at the larger budgets, and the missing Gaussian
middle $\delta_0$ values are still queued. The "Gaussian wins on compatible"
headline was an artefact of comparing against a stale diagonal reference. It
does not survive the audit and I am dropping that framing.
""")

    # ---- where to go from here -------------------------------------------
    md(r"""
# Where to go from here

- MA(2) current-reference coverage. The 80-seed flow refresh is done. The
  Gaussian-NPE current-reference audit covers five seeds, and a full refresh is
  queued as follow-up.
- Flow-NPE coverage at the GNK headline cell. The paired-seed coverage is 65 of
  101 at the moment. A staged-pipeline training recovery is running on the
  remaining 36 seeds and will be evaluated when complete.
- Robust standardisation. This is an internal diagnostic only at the moment. If
  the result holds and the theoretical justification clears, it could appear as
  a brief diagnostic in an appendix.
- Dimension-scaling pilot. The aim is to test $N \gg d^2 n$ directly. The
  result is preliminary.
- Stereological reference upgrade. I am considering the count-factorisation plus
  finer SMC-ABC plan above.
- Repo and data cleanup. Not done yet. HPC access ends around early June.
""")

    md(r"""
# Summary

- Stereological remains the clean opening example. Coverage is near nominal at
  practical budgets, with the reference-upgrade plan still open.
- GNK now has two useful diagnostics. BSL agrees with the NUTS reference at the
  checked cell, and both robust standardisation and rejection filtering move
  Gaussian-NPE much closer to that reference.
- MA(2) now has the corrected reference story. The compatible-case flow failure
  does not survive the current-reference audit, and the checked $\delta_0$
  endpoint no longer supports a simple flow-worse reading.
""")

    nb.cells = cells
    nb.metadata = {
        "title": "Empirical updates, meeting 18 May 2026",
        "date": "",
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python"},
    }
    out = HERE / "empirical_results_summary.ipynb"
    nbf.write(nb, out)
    print("wrote", out.name, "with", len(cells), "cells")


if __name__ == "__main__":
    stage_inputs()
    build_gnk_overlay()
    aggregate_flow_hexadecile()
    bootstrap_kl_ci()
    build_bsl_diagnostic_table()
    build_robust_scaling_panel()
    build_rejection_abc_panel()
    build_dim_scaling_panel()
    build_ma2_audit_caveat()
    aggregate_ma2_b0_kl()
    write_sources()
    build_notebook()
