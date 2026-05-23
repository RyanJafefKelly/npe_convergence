"""Aggregate exact MA(2) delta0=1.0 refresh outputs.

Reads the dry-run manifest and any completed files under
``res/ma2_b0_delta1_refresh``. It writes a method-aware group summary, a
per-row audit table, and a short note that can be rerun as jobs complete.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "notebooks" / "meeting_2026_05_18" / "data"
DOC_DIR = REPO_ROOT / "docs" / "meeting_2026_05_18"
MANIFEST_PATH = (
    DOC_DIR
    / "ma2_delta1_refresh_plan"
    / "ma2_delta1_refresh_dry_run_manifest.csv"
)
SUMMARY_PATH = DATA_DIR / "ma2_delta1_refresh.csv"
AUDIT_PATH = DATA_DIR / "ma2_delta1_refresh_audit.csv"
NOTE_PATH = DOC_DIR / "ma2_delta1_refresh_note.md"

COMPAT_FLOW_PATH = DATA_DIR / "ma2_compatibility_flow.csv"
B0_SUMMARY_PATH = DATA_DIR / "ma2_b0_per_seed_summary.csv"

COMPAT_COLUMNS = [
    "method",
    "n_obs",
    "n_sims",
    "budget_label",
    "delta0",
    "rows",
    "complete_rows",
    "shape_ok_rows",
    "finite_kl_rows",
    "infinite_kl_rows",
    "nan_kl_rows",
    "finite_kl_fraction",
    "infinite_kl_fraction",
    "finite_kl_min",
    "finite_kl_q25",
    "finite_kl_median",
    "finite_kl_q75",
    "finite_kl_max",
    "finite_mmd_rows",
    "mmd_min",
    "mmd_q25",
    "mmd_median",
    "mmd_q75",
    "mmd_max",
]


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_float(path: Path) -> float:
    try:
        return float(path.read_text().strip())
    except Exception:
        return math.nan


def finite_quantiles(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "min": math.nan,
            "q25": math.nan,
            "median": math.nan,
            "q75": math.nan,
            "max": math.nan,
        }
    q25, median, q75 = np.quantile(arr, [0.25, 0.50, 0.75])
    return {
        "min": float(np.min(arr)),
        "q25": float(q25),
        "median": float(median),
        "q75": float(q75),
        "max": float(np.max(arr)),
    }


def load_shape(path: Path) -> tuple[int, int] | None:
    if not path.is_file():
        return None
    try:
        data = np.load(path)
        samples = data["samples"]
    except Exception:
        return None
    if samples.ndim != 2:
        return None
    return int(samples.shape[0]), int(samples.shape[1])


def metric_key(method: str, metric: str) -> str:
    if method == "flow_npe" and metric == "kl":
        return "kl_reference_to_flow_npe"
    if method == "flow_npe" and metric == "mmd":
        return "mmd_reference_vs_flow_npe"
    if method == "gaussian_npe" and metric == "kl":
        return "kl_reference_to_gaussian_npe"
    if method == "gaussian_npe" and metric == "mmd":
        return "mmd_reference_vs_gaussian_npe"
    raise ValueError(f"unknown method/metric: {method}/{metric}")


def load_metrics(path: Path, method: str) -> tuple[float, float, str]:
    if not path.is_file():
        return math.nan, math.nan, "missing"
    try:
        metrics = json.loads(path.read_text())
        kl = float(metrics.get(metric_key(method, "kl"), math.nan))
        mmd = float(metrics.get(metric_key(method, "mmd"), math.nan))
    except Exception:
        return math.nan, math.nan, "unreadable"
    return kl, mmd, "ok"


def audit_rows(manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in manifest.iterrows():
        method = str(row["method"])
        kl_path = REPO_ROOT / str(row["kl_txt"])
        mmd_path = REPO_ROOT / str(row["mmd_txt"])
        metrics_path = REPO_ROOT / str(row["metrics_json"])
        reference_path = REPO_ROOT / str(row["reference_samples_npz"])
        posterior_path = REPO_ROOT / str(row["posterior_samples_npz"])

        metrics_kl, metrics_mmd, metrics_status = load_metrics(metrics_path, method)
        kl = read_float(kl_path) if kl_path.is_file() else metrics_kl
        mmd = read_float(mmd_path) if mmd_path.is_file() else metrics_mmd
        ref_shape = load_shape(reference_path)
        post_shape = load_shape(posterior_path)
        shape_ok = (
            ref_shape is not None
            and post_shape is not None
            and ref_shape[1] == 2
            and post_shape[1] == 2
        )
        required_paths = [kl_path, mmd_path, metrics_path, reference_path, posterior_path]
        missing = [rel(path) for path in required_paths if not path.is_file()]
        complete = not missing and shape_ok

        if not missing and not shape_ok:
            status = "bad_shape"
        elif missing:
            status = "missing"
        elif not np.isfinite(float(kl)):
            status = "nonfinite_kl"
        elif not np.isfinite(float(mmd)):
            status = "nonfinite_mmd"
        else:
            status = "complete"

        rows.append(
            {
                "method": method,
                "n_obs": int(row["n_obs"]),
                "n_sims": int(row["n_sims"]),
                "budget_label": row["budget_label"],
                "seed": int(row["seed"]),
                "delta0": float(row["delta0"]),
                "output_dir": row["output_dir"],
                "delta_dir": row["delta_dir"],
                "complete": complete,
                "status": status,
                "missing_files": ";".join(missing),
                "kl": float(kl),
                "mmd": float(mmd),
                "kl_finite": bool(np.isfinite(float(kl))),
                "mmd_finite": bool(np.isfinite(float(mmd))),
                "metrics_status": metrics_status,
                "reference_shape": "" if ref_shape is None else str(ref_shape),
                "posterior_shape": "" if post_shape is None else str(post_shape),
                "shape_ok": shape_ok,
                "metrics_json": row["metrics_json"],
                "kl_txt": row["kl_txt"],
                "mmd_txt": row["mmd_txt"],
                "reference_samples_npz": row["reference_samples_npz"],
                "posterior_samples_npz": row["posterior_samples_npz"],
            }
        )
    return pd.DataFrame(rows)


def group_summary(audit: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["method", "n_obs", "n_sims", "budget_label", "delta0"]
    for key, group in audit.groupby(keys, sort=True):
        method, n_obs, n_sims, budget_label, delta0 = key
        kl = pd.to_numeric(group["kl"], errors="coerce").to_numpy(dtype=float)
        mmd = pd.to_numeric(group["mmd"], errors="coerce").to_numpy(dtype=float)
        finite_kl = kl[np.isfinite(kl)]
        finite_mmd = mmd[np.isfinite(mmd)]
        inf_kl = kl[np.isinf(kl)]
        nan_kl = kl[np.isnan(kl)]
        kl_stats = finite_quantiles(finite_kl)
        mmd_stats = finite_quantiles(finite_mmd)
        total = int(len(group))
        rows.append(
            {
                "method": method,
                "n_obs": int(n_obs),
                "n_sims": int(n_sims),
                "budget_label": budget_label,
                "delta0": float(delta0),
                "rows": total,
                "complete_rows": int(group["complete"].astype(bool).sum()),
                "shape_ok_rows": int(group["shape_ok"].astype(bool).sum()),
                "finite_kl_rows": int(finite_kl.size),
                "infinite_kl_rows": int(inf_kl.size),
                "nan_kl_rows": int(nan_kl.size),
                "finite_kl_fraction": float(finite_kl.size / total) if total else math.nan,
                "infinite_kl_fraction": float(inf_kl.size / total) if total else math.nan,
                "finite_kl_min": kl_stats["min"],
                "finite_kl_q25": kl_stats["q25"],
                "finite_kl_median": kl_stats["median"],
                "finite_kl_q75": kl_stats["q75"],
                "finite_kl_max": kl_stats["max"],
                "finite_mmd_rows": int(finite_mmd.size),
                "mmd_min": mmd_stats["min"],
                "mmd_q25": mmd_stats["q25"],
                "mmd_median": mmd_stats["median"],
                "mmd_q75": mmd_stats["q75"],
                "mmd_max": mmd_stats["max"],
            }
        )
    return pd.DataFrame(rows, columns=COMPAT_COLUMNS)


def comparison_note(summary: pd.DataFrame, audit: pd.DataFrame) -> str:
    complete = int(audit["complete"].astype(bool).sum())
    total = int(len(audit))
    lines = [
        f"The exact delta0=1.0 refresh currently has {complete} complete rows out of {total} manifest rows.",
    ]
    if complete < total:
        lines.append(
            "The exact-compatible comparison should be treated as incomplete until the full PBS grid or selected repair runs finish."
        )
    if complete:
        sub = summary[
            (summary["n_obs"] == 100)
            & (summary["budget_label"] == "N=n")
            & (summary["finite_kl_rows"] > 0)
        ].sort_values("method")
        parts = [
            f"{row.method} KL median {row.finite_kl_median:.3f}"
            for row in sub.itertuples()
        ]
        if parts:
            lines.append("In the completed pilot cell group, " + ", ".join(parts) + ".")

    if COMPAT_FLOW_PATH.is_file():
        flow = pd.read_csv(COMPAT_FLOW_PATH)
        f099 = flow[(flow["delta0"] == 0.99) & (flow["n_obs"] == 1000)]
        if not f099.empty:
            vals = ", ".join(
                f"{row.budget_label} {row.finite_kl_median:.2f}"
                for row in f099.sort_values("n_sims").itertuples()
                if np.isfinite(row.finite_kl_median)
            )
            lines.append(f"For context, existing flow-NPE delta0=0.99 medians at n_obs=1000 are {vals}.")

    if B0_SUMMARY_PATH.is_file():
        b0 = pd.read_csv(B0_SUMMARY_PATH)
        fb0 = b0[(b0["method"] == "flow_npe") & (b0["n_obs"] == 1000)]
        if not fb0.empty:
            vals = ", ".join(
                f"{row.budget_label} {row.median:.2f}"
                for row in fb0.sort_values("n_sims").itertuples()
                if np.isfinite(row.median)
            )
            lines.append(f"The ordinary b0 compatible flow-NPE medians at n_obs=1000 are {vals}.")
    return "\n".join(lines) + "\n"


def main() -> None:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"missing manifest: {MANIFEST_PATH}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MANIFEST_PATH)
    audit = audit_rows(manifest)
    summary = group_summary(audit)
    audit.to_csv(AUDIT_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    NOTE_PATH.write_text(comparison_note(summary, audit))

    print("MA(2) exact delta0=1.0 aggregation complete")
    print(f"wrote: {rel(SUMMARY_PATH)} ({len(summary)} rows)")
    print(f"wrote: {rel(AUDIT_PATH)} ({len(audit)} rows)")
    print(f"wrote: {rel(NOTE_PATH)}")
    print(f"complete rows: {int(audit['complete'].astype(bool).sum())}/{len(audit)}")
    print(audit["status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
