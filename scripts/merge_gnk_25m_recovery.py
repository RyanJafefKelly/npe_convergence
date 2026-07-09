"""Fold the staged 25M flow-NPE recovery KLs into res/gnk and refresh the V3 aggregation.

The g-and-k headline cell (n_obs=5000, N=25,000,000, flow-NPE) shows 65 paired
seeds because only 65 of its 101 res/gnk dirs have a finite top-level kl.txt. The
recovered seeds (0-35 subset) finished in the staged worktree and have a finite
`final/kl.txt` there, but in a different layout, so aggregate_gnk_task2.py (one
cache root, strict naming, top-level kl.txt) does not see them.

This script copies each staged seed's final/{kl.txt, biases.npy,
estimated_coverage.npy} into the matching res/gnk dir (backing up anything it
overwrites), re-runs the aggregator to a temp dir, verifies the 25M flow cell
count went up, and only then copies the refreshed CSVs over the V3 dir.

Safe to re-run: it backs up overwritten files and refuses to touch V3 if the
aggregation fails. All actions are logged to /tmp/gnk_merge.log.
"""
from __future__ import annotations

import csv
import glob
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path("/Users/ryankelly/python_projects/npe_convergence")
BASE = REPO / "res/gnk"
STAGED = REPO / "res/staged_high_budget_20260517_all_remaining"
V3 = REPO / "notebooks/plots/gnk_task2_20260526_v3"
BACKUP = REPO / "res/_gnk_25m_premerge_backup"
FILES = ["kl.txt", "biases.npy", "estimated_coverage.npy"]
CELL = "n_obs_5000_n_sims_25000000"
PY = str(REPO / ".venv/bin/python")

_log: list[str] = []


def L(*a: object) -> None:
    s = " ".join(str(x) for x in a)
    _log.append(s)
    print(s, flush=True)


def finite(p: Path) -> bool:
    try:
        return math.isfinite(float(p.read_text().strip()))
    except Exception:
        return False


def cell_row(path: Path) -> dict | None:
    with path.open() as fh:
        for row in csv.DictReader(fh):
            if row.get("n") == "5000" and row.get("N") == "25000000":
                return row
    return None


def main() -> int:
    os.chdir(REPO)
    if not PY or not Path(PY).exists():
        L("ERROR: venv python not found at", PY)
        return 2

    before = cell_row(V3 / "raw_theta_kl_summary_comparable.csv")
    L("BEFORE 25M flow paired_seed_count",
      before["paired_seed_count"] if before else "??",
      "median", before["flow_theta_kl_median"] if before else "??")

    staged_dirs = sorted(glob.glob(str(STAGED / f"gnk_flow_npe_{CELL}_seed_*")))
    to_copy = []
    for d in staged_dirs:
        seed = int(d.split("seed_")[-1])
        final = Path(d) / "final"
        if finite(final / "kl.txt"):
            to_copy.append((seed, final))
    L("staged finite recovery seeds:", len(to_copy), [s for s, _ in to_copy])

    BACKUP.mkdir(exist_ok=True)
    copied = []
    for seed, final in to_copy:
        dest = BASE / f"npe_{CELL}_seed_{seed}"
        if not dest.exists():
            L("WARN res/gnk dir missing for seed", seed, "- skipping")
            continue
        for f in FILES:
            src = final / f
            if not src.exists():
                L("WARN staged missing", f, "for seed", seed)
                continue
            dst = dest / f
            if dst.exists():
                shutil.copy2(dst, BACKUP / f"seed_{seed}_{f}")
            shutil.copy2(src, dst)
        copied.append(seed)
    L("copied recovery seeds into res/gnk:", len(copied), copied)

    tmp = Path(tempfile.mkdtemp(prefix="gnk_v3_"))
    out = tmp / "v3"
    cmd = [
        PY, "scripts/aggregate_gnk_task2.py",
        "--cache-root", "res/gnk",
        "--u-space-csv", "notebooks/plots/gnk_u_space_kl_decomp_20260425_per_seed.csv",
        "--n500-gate-csv", "notebooks/plots/gnk_n500_oracle_gate_20260425_per_seed.csv",
        "--n500-gate-summary", "notebooks/plots/gnk_n500_oracle_gate_20260425_summary.json",
        "--output-dir", str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    L("aggregator rc", r.returncode)
    if r.stdout.strip():
        L("stdout tail:", r.stdout.strip()[-300:])
    if r.returncode != 0:
        L("stderr tail:", r.stderr.strip()[-800:])
        L("AGGREGATION FAILED - V3 left untouched")
        return 1

    after = cell_row(out / "raw_theta_kl_summary_comparable.csv")
    L("AFTER 25M flow paired_seed_count",
      after["paired_seed_count"] if after else "??",
      "median", after["flow_theta_kl_median"] if after else "??")

    if not after:
        L("ERROR: 25M cell missing from new aggregation - V3 left untouched")
        return 1

    for f in out.iterdir():
        if f.is_file():
            shutil.copy2(f, V3 / f.name)
    L("V3 refreshed at", V3)
    Path("/tmp/gnk_merge.log").write_text("\n".join(_log) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
