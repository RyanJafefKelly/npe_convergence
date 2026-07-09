"""Build the coauthor report notebook (Gaussian NPE empirical results).

Regenerates ``coauthor_report.ipynb`` from canonical result files, executes it,
and exports a code-hidden PDF that reads as a standalone document.

    python build_notebook.py

It mirrors the ``notebooks/meeting_2026_05_18`` builder: a SOURCES dict points at
the canonical result CSVs, ``stage_inputs()`` copies them into a local ``data/``
so the report folder is self-contained (and Zenodo-bundleable), and the notebook
cells read only from ``data/``.

Scope (decided 2026-05-31, do not drift from this):

  Three sections, one overriding goal: get the Section 1 results into the paper
  and submit. Everything else is supporting.

  Section 1  Paper-ready results under Gaussian NPE. Mirrors the manuscript's
             result slots (g-and-k, MA(2) compatibility, stereological).
  Section 2  Posterior sanity checks. Plain Gaussian-NPE and flow-NPE overlays
             against the reference, plus the g-and-k BSL check Chris asked for.
  Section 3  Internal diagnostics. Not for the paper; lean; each item says why
             it is shown.

  Plus a TEMPORARY decision-support section (deleted before sending) that pulls
  the exact visuals for the two open calls (MA(2) keep/drop; g-and-k aside).

  Deliberately EXCLUDED everywhere (do not re-add):
    - any reference-audit or bug-fix narrative. The report reads as a clean
      standalone document, near submission tone.
    - infinite / non-finite KL cell bookkeeping.
    - MA(2) covariance-error / finite-n moment checks.
    - robust-scaling and restricted-prior as Section 1/2 results (Section 3 only).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from nbformat.v4 import (new_code_cell, new_markdown_cell, new_notebook,
                         new_raw_cell)

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
REPO = HERE.parents[1]

# Canonical sources. The g-and-k tables come from the v3 task-2 aggregation
# (post current-reference); the hexadecile / stereological / MA(2) compatibility
# come from the 2026-05-13 postprocessing closeout. Derived per-figure artifacts
# (overlays, BSL table, hexadecile-flow, dim-scaling, rejection-ABC, MA(2)
# compatibility per-method, delta1 refresh) currently live in the meeting
# notebook's data dir, which is their canonical home for now.
MEET = REPO / "notebooks/meeting_2026_05_18/data"
V3 = REPO / "notebooks/plots/gnk_task2_20260526_v3"
POST = REPO / "docs/paper_empirical_push_2026_05_13/postprocessing_20260513"
STEREO = POST / "stereological_current_cache_aquarius_20260513"

SOURCES = {
    # g-and-k paper-ready
    "gnk_kl_flow_vs_gaussian.csv": V3 / "raw_theta_kl_summary_comparable.csv",
    "gnk_theta_oracle_by_n.csv": V3 / "theta_oracle_by_n.csv",
    "gnk_coverage_paper_grid.csv": V3 / "coverage_paper_grid_all_params.csv",
    "gnk_bias_g.csv": V3 / "bias_g_paper_figure_values.csv",
    "gnk_hexadecile_gaussian.csv":
        POST / "gnk_hexadecile_gaussian/gnk_hexadecile_group_summary.csv",
    "gnk_hexadecile_flow.csv": MEET / "gnk_hexadecile_flow.csv",
    # g-and-k overlays + BSL sanity check
    "gnk_posterior_overlay.csv": MEET / "gnk_posterior_overlay.csv",
    "gnk_bsl_diagnostic.csv": MEET / "gnk_bsl_diagnostic.csv",
    "gnk_robust_scaling_overlay.csv": MEET / "gnk_robust_scaling_overlay.csv",
    # MA(2) compatibility (three delta0 cases)
    "ma2_compatibility_gaussian.csv": MEET / "ma2_compatibility_gaussian.csv",
    "ma2_compatibility_flow.csv": MEET / "ma2_compatibility_flow.csv",
    "ma2_delta1_refresh.csv": MEET / "ma2_delta1_refresh.csv",
    "ma2_posterior_overlay_seed_22.csv": MEET / "ma2_posterior_overlay_seed_22.csv",
    # stereological
    "stereological_coverage.csv": STEREO / "coverage_all_params.csv",
    "stereological_bias_by_seed.csv": STEREO / "bias_boxplot_by_seed.csv",
    "stereological_posterior_overlay.csv":
        STEREO / "posterior_overlay_density_n1000_seed1_with_blackjax_abc.csv",
    # Section 3 diagnostics
    "dim_scaling_pilot_kl_by_d.csv": MEET / "dim_scaling_pilot_kl_by_d.csv",
    "gnk_rejection_abc_summary.json": MEET / "gnk_rejection_abc_summary.json",
}


# ---------------------------------------------------------------------------
# Staging + derived tables
# ---------------------------------------------------------------------------

def _copy_if_needed(src: Path, dst: Path) -> None:
    if not src.exists():
        if not dst.exists():
            print("  MISSING source and no staged copy:", src)
        else:
            print("  source missing, keeping staged copy:", dst.name)
        return
    if src.resolve() == dst.resolve():
        return
    shutil.copy(src, dst)


def stage_inputs() -> None:
    DATA.mkdir(exist_ok=True)
    for name, src in SOURCES.items():
        _copy_if_needed(Path(src), DATA / name)
    print(f"staged or checked {len(SOURCES)} input files")


def build_ma2_three_case() -> None:
    """Unify the three MA(2) compatibility cases into one long table.

    delta0 = 0.01 / 0.99 from the per-method compatibility sweeps, delta0 = 1.0
    (well-specified) from the delta1 refresh. Output columns:
    [method, delta0, n_obs, n_sims, budget_label, kl_median, n_seeds].
    """
    frames = []

    # Gaussian compatibility: per-seed raw KL -> median per cell.
    g = pd.read_csv(DATA / "ma2_compatibility_gaussian.csv")
    g = g[g.delta0.isin([0.01, 0.99])]
    gg = (g.groupby(["delta0", "n_obs", "n_sims"])["kl"]
            .agg(kl_median="median", n_seeds="count").reset_index())
    gg["method"] = "gaussian_npe"
    frames.append(gg)

    # Flow compatibility: already aggregated to finite_kl_median.
    f = pd.read_csv(DATA / "ma2_compatibility_flow.csv")
    f = f[f.delta0.isin([0.01, 0.99])].copy()
    f = f.rename(columns={"finite_kl_median": "kl_median",
                          "finite_kl_rows": "n_seeds"})
    f["method"] = "flow_npe"
    frames.append(f[["method", "delta0", "n_obs", "n_sims", "kl_median",
                     "n_seeds"]])

    # Well-specified (delta0 = 1.0): both methods, already aggregated.
    d1 = pd.read_csv(DATA / "ma2_delta1_refresh.csv").copy()
    d1 = d1.rename(columns={"finite_kl_median": "kl_median",
                            "finite_kl_rows": "n_seeds"})
    frames.append(d1[["method", "delta0", "n_obs", "n_sims", "kl_median",
                      "n_seeds"]])

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["method", "delta0", "n_obs", "n_sims"])
    out.to_csv(DATA / "ma2_three_case_kl.csv", index=False)
    print("wrote ma2_three_case_kl.csv", out.shape,
          "delta0:", sorted(out.delta0.unique()))


def build_gnk_improvement_overlay() -> None:
    """Combine the g-and-k posterior-improvement curves into one overlay table.

    For the D2 question (can we tighten the g-and-k NPE posterior toward the
    reference?). The robust-scaling overlay already holds KDEs of the NUTS
    reference, vanilla z-score Gaussian-NPE, robust-asinh Gaussian-NPE, and BSL
    at the n=1000, seed 0 cell. We add the rejection-ABC (preconditioning-style
    trimming) posterior, KDE'd on the same per-parameter grid, so all five
    curves are directly comparable. Output: gnk_improvement_overlay.csv.
    """
    import pickle

    from scipy.stats import gaussian_kde

    rob = pd.read_csv(DATA / "gnk_robust_scaling_overlay.csv")
    frames = [rob]
    rej_path = (REPO / "res/gnk_rejection_abc/"
                "gaussian_npe_n_obs_1000_n_pool_10000000_seed_0_acc_0.01/"
                "posterior_samples.pkl")
    if rej_path.exists():
        with open(rej_path, "rb") as fh:
            rej = np.asarray(pickle.load(fh), dtype=float)
        cell = rob["cell"].iloc[0] if "cell" in rob.columns else ""
        new = []
        for j, param in enumerate(["A", "B", "g", "k"]):
            ref_x = rob[(rob.method == "NUTS reference")
                        & (rob.param == param)].x.to_numpy()
            if ref_x.size == 0:
                continue
            grid = np.linspace(ref_x.min(), ref_x.max(), max(ref_x.size, 200))
            dens = gaussian_kde(rej[:, j])(grid)
            for x, d in zip(grid, dens):
                new.append({"cell": cell,
                            "method": "Gaussian-NPE (rejection-ABC)",
                            "param": param, "x": float(x), "density": float(d)})
        frames.append(pd.DataFrame(new))
    else:
        print("  rejection-ABC samples not found, overlay omits that curve")
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(DATA / "gnk_improvement_overlay.csv", index=False)
    print("wrote gnk_improvement_overlay.csv", out.shape,
          "methods:", sorted(out.method.unique()))


# ---------------------------------------------------------------------------
# Cell helpers
# ---------------------------------------------------------------------------

def md(text: str) -> nbformat.NotebookNode:
    return new_markdown_cell(text)


def todo(text: str) -> nbformat.NotebookNode:
    """Visible internal TODO note. Strip before sending."""
    return new_markdown_cell(f"> **TODO:** {text}")


def needs(text: str) -> nbformat.NotebookNode:
    """Visible internal data-need / HPC-pull flag. Strip before sending."""
    return new_markdown_cell(f"> **NEEDS DATA:** {text}")


def code(source: str) -> nbformat.NotebookNode:
    """Code cell, hidden in the PDF so the report reads as prose + figures."""
    cell = new_code_cell(source.strip("\n"))
    cell.metadata["jupyter"] = {"source_hidden": True}
    return cell


def raw_latex(text: str) -> nbformat.NotebookNode:
    """Raw LaTeX passthrough, used to tweak the PDF preamble.

    The default nbconvert LaTeX template numbers every heading, which clashes
    with our manual 1A / 1B / 2 / 3 scheme. Disabling LaTeX section numbering
    leaves only the manual numbers.
    """
    cell = new_raw_cell(text)
    cell.metadata["raw_mimetype"] = "text/latex"
    return cell


def stub(label: str, source_file: str | None = None) -> nbformat.NotebookNode:
    src_line = f"  source: data/{source_file}" if source_file else "  source: TBD"
    return code(
        "# PLACEHOLDER. Replace with the figure/table code.\n"
        f"print('TODO: {label}')\n"
        f"print('{src_line}')\n"
    )


SETUP = r"""
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
C_BSL = "#2ca02c"
BUDGETS = ["N=n", "N=n log(n)", "N=n^(3/2)", "N=n^2"]
BUDGET_MATH = ["n", r"n\log n", r"n^{3/2}", r"n^2"]
N_OBS = [100, 500, 1000, 5000]
METHOD_NAME = {"flow_npe": "flow-NPE", "gaussian_npe": "Gaussian-NPE"}
PARAM_MATH = {"A": "A", "B": "B", "g": "g", "k": "k",
              "lambda": r"\lambda", "sigma": r"\sigma", "xi": r"\xi"}
GNK_TRUE = {"A": 3.0, "B": 1.0, "g": 2.0, "k": 0.5}
"""


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

def header_cells() -> list[nbformat.NotebookNode]:
    return [
        raw_latex(
            r"""\makeatletter
\setcounter{secnumdepth}{-1}
\renewcommand{\@seccntformat}[1]{}
\makeatother"""
        ),
        md(
            "This report has three parts:\n"
            "\n"
            "1. **Paper-ready results** under Gaussian NPE, intended for the "
            "manuscript.\n"
            "2. **Posterior sanity checks** against the reference posterior.\n"
            "3. **Internal diagnostics** supporting the results above, not "
            "intended for the paper."
        ),
        md(
            "This report collects the paper's empirical results, re-run under "
            "Gaussian NPE and shown next to flow NPE. Section 1 is the "
            "submission-ready material that maps onto the manuscript's result "
            "slots; Sections 2 and 3 are supporting posterior checks and "
            "internal diagnostics, not intended for the paper. Read Section 1 as "
            "the main content."
        ),
        md(
            "## Internal note (remove before sending)\n"
            "\n"
            "Status as of 2026-06-01. The sections are settled and the data is "
            "current. Before sending: delete this note and the Open-questions "
            "section at the end, and strip the inline TODO notes (the closing "
            "note lists the steps)."
        ),
        code(SETUP),
    ]


# ---------------------------------------------------------------------------
# Section 1: paper-ready results
# ---------------------------------------------------------------------------

def section1_cells() -> list[nbformat.NotebookNode]:
    cells: list[nbformat.NotebookNode] = []
    cells.append(md("# 1. Paper-ready results (Gaussian NPE)"))
    cells.append(
        md(
            "The result slots already in the manuscript, presented under "
            "Gaussian NPE alongside flow NPE. This section is split to match the "
            "manuscript: **1A** collects the results that appear in the main "
            "body, and **1B** collects the results that appear in the "
            "appendices (the per-parameter coverage and bias breakdowns, the full "
            "KL grid, and the additional stereological parameters)."
        )
    )

    # =====================================================================
    # 1A. Main-text results
    # =====================================================================
    cells.append(md(
        "# 1A. Main-text results\n"
        "\n"
        "These mirror the result slots in the manuscript's main body: g-and-k "
        "($g$), MA(2) compatibility, and the stereological rate $\\lambda$."
    ))

    # --- 1A.1 g-and-k (main text) ---
    cells.append(md(
        "## 1A.1 g-and-k\n"
        "\n"
        "Four parameters $(A, B, g, k)$ with octile summaries, so $d=11$. The "
        "reference is NUTS on the asymptotic summary-likelihood. The main text "
        "reports the bias and coverage for $g$ and the octile-vs-hexadecile KL "
        "comparison."
    ))

    cells.append(md(
        "### Posterior-mean bias for $g$\n"
        "\n"
        "Per-seed posterior-mean bias for $g$, flow and Gaussian side by side at "
        "each budget. The dashed line is zero bias."
    ))
    cells.append(code(r"""
bias = pd.read_csv(DATA / "gnk_bias_g.csv")
bias = bias[bias.N_label.isin(BUDGETS)]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, n in zip(axes, [1000, 5000]):
    for b, bud in enumerate(BUDGETS):
        for k, (raw, color) in enumerate([("flow_npe", C_FLOW),
                                          ("gaussian_npe", C_GAUSS)]):
            vals = bias[(bias.n == n) & (bias.N_label == bud)
                        & (bias.method == raw)]["seed_mean_bias"].dropna().values
            if len(vals) == 0:
                continue
            bp = ax.boxplot([vals], positions=[b * 2.6 + k], widths=0.9,
                            patch_artist=True,
                            flierprops=dict(marker=".", markersize=2,
                                            markerfacecolor=color, markeredgecolor=color))
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.55)
            for med in bp["medians"]:
                med.set_color("black")
    ax.axhline(0, color="black", ls="--", lw=0.8, alpha=0.5)
    ax.set_xticks([b * 2.6 + 0.5 for b in range(4)])
    ax.set_xticklabels([f"${b}$" for b in BUDGET_MATH], fontsize=8)
    ax.set_title(f"$n = {n}$")
axes[0].set_ylabel("posterior-mean bias for $g$")
axes[0].plot([], [], "s", color=C_FLOW, label="flow-NPE")
axes[0].plot([], [], "s", color=C_GAUSS, label="Gaussian-NPE")
axes[0].legend(frameon=False, fontsize=8, loc="upper right")
fig.tight_layout()
plt.show()
"""))

    cells.append(md(
        "### Coverage of credible intervals for $g$\n"
        "\n"
        "Monte Carlo coverage of the 95% credible intervals for $g$ at $n=1000$, "
        "per budget. The intervals run at or above nominal here."
    ))
    cells.append(code(r"""
covg = pd.read_csv(DATA / "gnk_coverage_paper_grid.csv")
covg = covg[(covg.n == 1000) & (covg.param == "g")
            & covg.N_label.isin(BUDGETS)].copy()
covg["method"] = covg["method"].map(METHOD_NAME)
tab = covg.pivot_table(index=["method", "param"], columns="N_label",
                       values="coverage_95_mean")
tab = tab.reindex(columns=[b for b in BUDGETS if b in tab.columns]).round(3)
tab.columns.name = "budget (95% coverage, g)"
tab
"""))

    cells.append(md(
        "### KL: octiles vs hexadeciles\n"
        "\n"
        "The same g-and-k inference with octile summaries ($d_s=7$, so $d=11$) "
        "and hexadecile summaries ($d_s=15$, so $d=19$). The higher-dimensional "
        "summary needs a larger budget for the same accuracy, the $d^2 n$ "
        "scaling showing up directly. The $n=100$ panel is the least trustworthy "
        "for Gaussian-NPE, since the reference posterior is far from Gaussian at "
        "$n=100$, so the $n=1000$ panel is the cleaner comparison."
    ))
    cells.append(code(r"""
gnk = pd.read_csv(DATA / "gnk_kl_flow_vs_gaussian.csv")
oct_df = gnk[gnk.standard_paper_grid == 1].copy()
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
    hf = hex_f[hex_f.n_obs == n]
    if "n_seeds" in hf.columns:
        hf = hf[hf.n_seeds >= 10]   # drop stray 1-2 seed budget cells
    hf = hf.sort_values("n_sims")
    if not hf.empty and "kl_median" in hf.columns:
        ax.plot(hf.n_sims, hf.kl_median, "o--", color=C_FLOW,
                label=r"flow hexadeciles ($d_s=15$)")
    ax.set_xscale("log")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"$n = {n}$")
    ax.set_xlabel("simulation budget $N$")
axes[0].set_ylabel("KL from reference posterior")
fig.tight_layout()
plt.show()
"""))

    # --- 1A.2 MA(2) ---
    cells.append(md(
        "## 1A.2 MA(2)\n"
        "\n"
        "Two parameters $(\\theta_1, \\theta_2)$ with the variance and first two "
        "autocovariances as summaries. The compatibility figure fixes "
        "$\\delta_0(y)$ and varies the budget $N$."
    ))
    cells.append(md(
        "### Compatibility: KL vs N\n"
        "\n"
        "KL from the reference posterior to the NPE posterior at $n=1000$, for "
        "severe incompatibility ($\\delta_0=0.01$) and mild incompatibility "
        "($\\delta_0=0.99$). Under severe incompatibility a larger budget does "
        "not help; under mild incompatibility the KL falls with the budget for "
        "both methods. This is the original two-panel result for the manuscript."
    ))
    cells.append(code(r"""
m = pd.read_csv(DATA / "ma2_three_case_kl.csv")
m = m[m.n_obs == 1000]
cases = [(0.01, "severe ($\\delta_0=0.01$)"),
         (0.99, "mild ($\\delta_0=0.99$)")]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, (d0, title) in zip(axes, cases):
    for raw, color in [("flow_npe", C_FLOW), ("gaussian_npe", C_GAUSS)]:
        s = m[(m.method == raw) & (m.delta0 == d0)].sort_values("n_sims")
        s = s[s.kl_median.notna() & (s.n_seeds > 0)]
        if not s.empty:
            ax.plot(s.n_sims, s.kl_median, "o-", color=color,
                    label=METHOD_NAME[raw])
    ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel("simulation budget $N$")
    ax.legend(frameon=False, fontsize=8)
axes[0].set_ylabel("KL from reference posterior")
fig.suptitle("MA(2) compatibility, $n=1000$ (note the differing y-scales)", y=1.02)
fig.tight_layout()
plt.show()
"""))

    # --- 1A.3 stereological (lambda, main text) ---
    cells.append(md(
        "## 1A.3 Stereological\n"
        "\n"
        "Three parameters $(\\lambda, \\sigma, \\xi)$ and four summaries, so "
        "$d=7$. The reference is SMC-ABC. This is the paper's motivating example. "
        "The main text reports the rate $\\lambda$."
    ))
    cells.append(md(
        "### Coverage for $\\lambda$\n"
        "\n"
        "Monte Carlo coverage of the 95% credible intervals for $\\lambda$, per "
        "budget and $n$. The $n=5000,\\ N=n^2$ cell is not run (simulation cost)."
    ))
    cells.append(code(r"""
cov = pd.read_csv(DATA / "stereological_coverage.csv")
cov = cov[(cov.N_label.isin(BUDGETS)) & (cov["param"] == "lambda")].copy()
cov["method"] = cov["method"].map(METHOD_NAME)
cov["N_label"] = pd.Categorical(cov["N_label"], categories=BUDGETS, ordered=True)
cov_table = cov.pivot_table(index=["method", "N_label"], columns="n",
                            values="coverage_95_mean", observed=True)
cov_table = cov_table.sort_index()[N_OBS].round(3)
cov_table = cov_table.where(cov_table.notna(), "")
cov_table.index.names = ["method", "budget"]
cov_table.columns.name = "n (95% coverage, lambda)"
cov_table
"""))

    cells.append(md(
        "### Posterior-mean bias for $\\lambda$\n"
        "\n"
        "Per-seed posterior-mean bias for $\\lambda$ by observation count, flow "
        "and Gaussian side by side at each budget."
    ))
    cells.append(code(r"""
bias = pd.read_csv(DATA / "stereological_bias_by_seed.csv")
bias = bias[bias.N_label.isin(BUDGETS)]
params = ["lambda"]
fig, axes = plt.subplots(1, 4, figsize=(13, 3.0), squeeze=False)
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
                                                markerfacecolor=color, markeredgecolor=color))
                for patch in bp["boxes"]:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.55)
                for med in bp["medians"]:
                    med.set_color("black")
        ax.axhline(0, color="black", ls="--", lw=0.8, alpha=0.5)
        ax.set_xticks([b * 2.6 + 0.5 for b in range(4)])
        ax.set_xticklabels([f"${b}$" for b in BUDGET_MATH], fontsize=7)
        ax.set_title(f"$n = {n}$")
        if c == 0:
            ax.set_ylabel(f"${PARAM_MATH[param]}$\nbias")
axes[0, 0].plot([], [], "s", color=C_FLOW, label="flow-NPE")
axes[0, 0].plot([], [], "s", color=C_GAUSS, label="Gaussian-NPE")
axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
plt.show()
"""))

    cells.append(md(
        "### Posterior for $\\lambda$ vs ABC-SMC\n"
        "\n"
        "Marginal posterior for $\\lambda$ on a single simulated dataset "
        "($n=1000$, seed 1), NPE at each budget against the SMC-ABC benchmark. "
        "The dashed line is the true $\\lambda$."
    ))
    cells.append(code(r"""
ster_ov = pd.read_csv(DATA / "stereological_posterior_overlay.csv")
lam = ster_ov[ster_ov.param == "lambda"]
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharex=True)
specs = [("flow_npe", "flow-NPE", plt.cm.Blues),
         ("gaussian_npe", "Gaussian-NPE", plt.cm.Reds)]
for ax, (raw, mname, cmap) in zip(axes, specs):
    shades = cmap(np.linspace(0.4, 0.95, len(BUDGETS)))
    for bud, shade, lab in zip(BUDGETS, shades, BUDGET_MATH):
        s = lam[(lam.method == raw) & (lam.N_label == bud)].sort_values("x")
        if not s.empty:
            ax.plot(s.x, s.density, color=shade, label=f"$N={lab}$")
    a = lam[lam.method == "abc_smc"].sort_values("x")
    if not a.empty:
        ax.plot(a.x, a.density, color="black", lw=1.4, label="SMC-ABC")
    ax.axvline(100.0, color="black", ls="--", lw=0.8, alpha=0.6)
    ax.set_title(mname)
    ax.set_xlabel(r"$\lambda$")
    ax.legend(frameon=False, fontsize=7)
fig.tight_layout()
plt.show()
"""))

    # =====================================================================
    # 1B. Appendix results
    # =====================================================================
    cells.append(md(
        "# 1B. Appendix results\n"
        "\n"
        "These mirror the manuscript's appendices: the full g-and-k KL grid and "
        "its remaining parameters ($A, B, k$), and the additional stereological "
        "parameters ($\\sigma, \\xi$). The MA(2) appendix in the paper is "
        "methodology only (KL estimation and exact partial-posterior sampling), "
        "so it carries no extra empirical panels here."
    ))

    # --- 1B.1 g-and-k (appendix) ---
    cells.append(md(
        "## 1B.1 g-and-k (appendix)\n"
        "\n"
        "The full KL grid across all $n$ and budgets $N$, and coverage for the "
        "remaining parameters $A$, $B$, $k$. This is the report's version of the "
        "paper's appendix KL table."
    ))
    cells.append(md(
        "### KL divergence, full $n$ by $N$ grid\n"
        "\n"
        "Median over seeds of the KL from the reference posterior to the NPE "
        "posterior (flow / Gaussian), at each $n$ and budget $N$. The last row, "
        "the moment-matched Gaussian, is a reference floor: it is the KL you get "
        "if you replace the NPE with the single Gaussian whose mean and "
        "covariance match the reference posterior exactly. No Gaussian-family "
        "approximation can beat it, so it marks how much of the remaining KL is "
        "the Gaussian-family limit rather than anything NPE can fix with a larger "
        "budget."
    ))
    cells.append(code(r"""
gnk = pd.read_csv(DATA / "gnk_kl_flow_vs_gaussian.csv")
grid = gnk[gnk.standard_paper_grid == 1].copy()

def kl_entry(r):
    return f"{r.flow_theta_kl_median:.2f} / {r.gaussian_theta_kl_median:.2f}"

grid["entry"] = grid.apply(kl_entry, axis=1)
kl_table = grid.pivot(index="N_label", columns="n", values="entry")
kl_table = kl_table.reindex(BUDGETS)[N_OBS]
oracle = pd.read_csv(DATA / "gnk_theta_oracle_by_n.csv").set_index("n")
floor_col = "K_theta_star_median" if "K_theta_star_median" in oracle.columns else oracle.columns[-1]
kl_table.loc["moment-matched Gaussian"] = [
    f"{oracle.loc[n, floor_col]:.4f}" for n in N_OBS]
kl_table.index.name = "budget / floor (flow / Gaussian)"
kl_table.columns.name = "n"
kl_table
"""))
    cells.append(md(
        "At each fixed $n$ the KL falls as the budget $N$ grows, and flow and "
        "Gaussian stay close to each other (Gaussian is modestly worse at the "
        "larger budgets, for example $2.67$ vs $2.22$ at $n=5000,\\ N=n^2$, but "
        "not by a wide margin). One cell runs against the expected pattern: at "
        "the largest budget $N=n^2$ the median KL is lowest at $n=1000$ and then "
        "rises at $n=5000$ (flow $1.82 \\to 2.22$, Gaussian $2.45 \\to 2.67$), "
        "rather than continuing to fall with $n$. The paper's appendix text "
        "states the KL decreases in both $n$ and $N$, so this $n=5000,\\ N=n^2$ "
        "cell is the one place that does not yet support that reading. It needs "
        "follow-up before the table goes in: see the Open-questions section.\n"
        "\n"
        "The KL also does not fall to zero. The moment-matched-Gaussian floor is "
        "small (well under 1 at the larger $n$), so most of the remaining KL sits "
        "above that floor. Calling that gap finite-$N$ amortisation error is "
        "plausible but not yet established here; quantifying it is on the "
        "Open-questions list."
    ))
    cells.append(md(
        "### Coverage of credible intervals for $A$, $B$, $k$\n"
        "\n"
        "Monte Carlo coverage of the 95% credible intervals at $n=1000$, per "
        "parameter and budget. As with $g$ in the main text, these run at or "
        "above nominal (for example $k$ near 1.0)."
    ))
    cells.append(code(r"""
covg = pd.read_csv(DATA / "gnk_coverage_paper_grid.csv")
covg = covg[(covg.n == 1000) & covg.param.isin(["A", "B", "k"])
            & covg.N_label.isin(BUDGETS)].copy()
covg["method"] = covg["method"].map(METHOD_NAME)
tab = covg.pivot_table(index=["method", "param"], columns="N_label",
                       values="coverage_95_mean")
tab = tab.reindex(columns=[b for b in BUDGETS if b in tab.columns]).round(3)
tab.columns.name = "budget (95% coverage)"
tab
"""))

    # --- 1B.2 stereological (appendix): sigma and xi ---
    cells.append(md(
        "## 1B.2 Stereological (appendix): $\\sigma$ and $\\xi$\n"
        "\n"
        "Coverage, posterior-mean bias, and posterior overlays for the shape and "
        "scale parameters $\\sigma$ and $\\xi$."
    ))
    cells.append(md(
        "### Coverage for $\\sigma$ and $\\xi$\n"
        "\n"
        "Monte Carlo coverage of the 95% credible intervals, per parameter, "
        "budget and $n$."
    ))
    cells.append(code(r"""
cov = pd.read_csv(DATA / "stereological_coverage.csv")
cov = cov[(cov.N_label.isin(BUDGETS)) & cov["param"].isin(["sigma", "xi"])].copy()
cov["method"] = cov["method"].map(METHOD_NAME)
cov["N_label"] = pd.Categorical(cov["N_label"], categories=BUDGETS, ordered=True)
cov_table = cov.pivot_table(index=["param", "method", "N_label"], columns="n",
                            values="coverage_95_mean", observed=True)
cov_table = cov_table.sort_index()[N_OBS].round(3)
cov_table = cov_table.where(cov_table.notna(), "")
cov_table.index.names = ["param", "method", "budget"]
cov_table.columns.name = "n (95% coverage)"
cov_table
"""))
    cells.append(md(
        "### Posterior-mean bias for $\\sigma$ and $\\xi$\n"
        "\n"
        "Per-seed posterior-mean bias by parameter and observation count, flow "
        "and Gaussian side by side at each budget."
    ))
    cells.append(code(r"""
bias = pd.read_csv(DATA / "stereological_bias_by_seed.csv")
bias = bias[bias.N_label.isin(BUDGETS)]
params = ["sigma", "xi"]
# sharey='row' puts each parameter on one common scale across n, so the
# effect of increasing n is read directly.
fig, axes = plt.subplots(2, 4, figsize=(13, 5.6), squeeze=False, sharey="row")
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
                                                markerfacecolor=color, markeredgecolor=color))
                for patch in bp["boxes"]:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.55)
                for med in bp["medians"]:
                    med.set_color("black")
        ax.axhline(0, color="black", ls="--", lw=0.8, alpha=0.5)
        ax.set_xticks([b * 2.6 + 0.5 for b in range(4)])
        ax.set_xticklabels([f"${b}$" for b in BUDGET_MATH], fontsize=7)
        if r == 0:
            ax.set_title(f"$n = {n}$")
        if c == 0:
            ax.set_ylabel(f"${PARAM_MATH[param]}$\nbias")
axes[0, 0].plot([], [], "s", color=C_FLOW, label="flow-NPE")
axes[0, 0].plot([], [], "s", color=C_GAUSS, label="Gaussian-NPE")
axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
plt.show()
"""))
    cells.append(md(
        "### Posteriors for $\\sigma$ and $\\xi$ vs ABC-SMC\n"
        "\n"
        "Marginal posteriors on a single simulated dataset ($n=1000$, seed 1), "
        "NPE at each budget against the SMC-ABC benchmark. Dashed lines are the "
        "true values ($\\sigma=2.0$, $\\xi=-0.1$)."
    ))
    cells.append(code(r"""
ster_ov = pd.read_csv(DATA / "stereological_posterior_overlay.csv")
truth = {"sigma": 2.0, "xi": -0.1}
specs = [("flow_npe", "flow-NPE", plt.cm.Blues),
         ("gaussian_npe", "Gaussian-NPE", plt.cm.Reds)]
params = ["sigma", "xi"]
fig, axes = plt.subplots(len(params), 2, figsize=(12, 6.6), squeeze=False)
for r, param in enumerate(params):
    d = ster_ov[ster_ov.param == param]
    for c, (raw, mname, cmap) in enumerate(specs):
        ax = axes[r, c]
        shades = cmap(np.linspace(0.4, 0.95, len(BUDGETS)))
        for bud, shade, lab in zip(BUDGETS, shades, BUDGET_MATH):
            s = d[(d.method == raw) & (d.N_label == bud)].sort_values("x")
            if not s.empty:
                ax.plot(s.x, s.density, color=shade, label=f"$N={lab}$")
        a = d[d.method == "abc_smc"].sort_values("x")
        if not a.empty:
            ax.plot(a.x, a.density, color="black", lw=1.4, label="SMC-ABC")
        ax.axvline(truth[param], color="black", ls="--", lw=0.8, alpha=0.6)
        if r == 0:
            ax.set_title(mname)
        ax.set_xlabel(f"${PARAM_MATH[param]}$")
        if r == 0 and c == 0:
            ax.legend(frameon=False, fontsize=7)
fig.tight_layout()
plt.show()
"""))

    return cells


# ---------------------------------------------------------------------------
# Section 2: posterior sanity checks
# ---------------------------------------------------------------------------

def section2_cells() -> list[nbformat.NotebookNode]:
    cells: list[nbformat.NotebookNode] = []
    cells.append(md("# 2. Posterior sanity checks"))
    cells.append(md(
        "Plain posterior overlays: Gaussian NPE and flow NPE against the "
        "reference posterior, to confirm by eye that the approximations are "
        "reasonable. Not intended for the paper."
    ))

    cells.append(md("## 2.1 Posterior overlays vs reference"))
    cells.append(md(
        "### g-and-k\n"
        "\n"
        "Marginal posteriors for the four parameters at two headline cells, "
        "reference against the two NPE variants."
    ))
    cells.append(code(r"""
ov = pd.read_csv(DATA / "gnk_posterior_overlay.csv")
cells_to_show = ["n=1000, N=n^2", "n=5000, N=n^2"]
fig, axes = plt.subplots(len(cells_to_show), 4, figsize=(13, 6))
for r, cell in enumerate(cells_to_show):
    sub = ov[ov.cell == cell]
    for c, param in enumerate(["A", "B", "g", "k"]):
        ax = axes[r, c]
        d = sub[sub.param == param]
        for method, color in [("flow-NPE", C_FLOW),
                              ("Gaussian-NPE", C_GAUSS),
                              ("Reference (flow convention)", C_REF)]:
            dd = d[d.method == method].sort_values("x")
            if not dd.empty:
                ax.plot(dd.x, dd.density, color=color, lw=1.3,
                        label=method.replace(" (flow convention)", " (reference)"))
        if r == 0 and c == 0:
            ax.legend(frameon=False, fontsize=7)
        if r == 0:
            ax.set_title(f"${param}$")
        ax.set_yticks([])
    axes[r, 0].set_ylabel(cell)
fig.tight_layout()
plt.show()
"""))

    cells.append(md(
        "### MA(2)\n"
        "\n"
        "Joint posterior for $(\\theta_1, \\theta_2)$ at a representative seed "
        "(seed 22, $n=1000$, $N=n^2$), flow and Gaussian against the true "
        "posterior."
    ))
    cells.append(code(r"""
from scipy.stats import gaussian_kde
ma2 = pd.read_csv(DATA / "ma2_posterior_overlay_seed_22.csv")
fig, ax = plt.subplots(figsize=(5.4, 5.0))
xg = np.linspace(ma2.t1.min(), ma2.t1.max(), 140)
yg = np.linspace(ma2.t2.min(), ma2.t2.max(), 140)
XX, YY = np.meshgrid(xg, yg)
grid = np.vstack([XX.ravel(), YY.ravel()])
for method, color in [("true posterior", C_REF),
                      ("flow-NPE", C_FLOW),
                      ("Gaussian-NPE", C_GAUSS)]:
    s = ma2[ma2.method == method]
    if len(s) < 20:
        continue
    ZZ = gaussian_kde(np.vstack([s.t1, s.t2]))(grid).reshape(XX.shape)
    ax.contour(XX, YY, ZZ, levels=4, colors=[color], linewidths=1.2)
    ax.plot([], [], color=color, label=method)
ax.set_xlabel(r"$\theta_1$")
ax.set_ylabel(r"$\theta_2$")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
plt.show()
"""))

    cells.append(md(
        "### Stereological\n"
        "\n"
        "Marginal posteriors for $(\\lambda, \\sigma, \\xi)$ at the largest "
        "available budget, NPE variants against SMC-ABC ($n=1000$, seed 1)."
    ))
    cells.append(code(r"""
ster_ov = pd.read_csv(DATA / "stereological_posterior_overlay.csv")
budget_order = [b for b in BUDGETS if b in set(ster_ov.N_label.dropna())]
top_budget = budget_order[-1] if budget_order else None
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
for ax, param in zip(axes, ["lambda", "sigma", "xi"]):
    d = ster_ov[ster_ov.param == param]
    for raw, color, lab in [("flow_npe", C_FLOW, "flow-NPE"),
                            ("gaussian_npe", C_GAUSS, "Gaussian-NPE")]:
        s = d[(d.method == raw) & (d.N_label == top_budget)].sort_values("x")
        if not s.empty:
            ax.plot(s.x, s.density, color=color, lw=1.3, label=lab)
    a = d[d.method == "abc_smc"].sort_values("x")
    if not a.empty:
        ax.plot(a.x, a.density, color="black", lw=1.4, label="SMC-ABC")
    ax.set_title(f"${PARAM_MATH[param]}$")
    ax.set_yticks([])
axes[0].legend(frameon=False, fontsize=8)
fig.suptitle(f"Stereological posteriors at {top_budget}", y=1.02)
fig.tight_layout()
plt.show()
"""))

    cells.append(md(
        "## 2.2 g-and-k BSL sanity check\n"
        "\n"
        "Bayesian synthetic likelihood (BSL) as an independent check on the "
        "g-and-k reference, requested by Chris. This is a single-cell check at "
        "$n=1000$, seed 0, not a full-grid result. At that cell BSL recovers the "
        "NUTS reference closely, which supports reading the NPE gap as an "
        "NPE-side effect rather than a problem with the reference."
    ))
    cells.append(md(
        "Direct posterior overlay: BSL against the NUTS reference, per parameter. "
        "Close agreement supports the reference."
    ))
    cells.append(code(r"""
ov = pd.read_csv(DATA / "gnk_robust_scaling_overlay.csv")
fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
for c, param in enumerate(["A", "B", "g", "k"]):
    ax = axes[c]
    d = ov[ov.param == param]
    for method, color, lab in [("NUTS reference", C_REF, "NUTS reference"),
                               ("BSL", C_BSL, "BSL")]:
        s = d[d.method == method].sort_values("x")
        if not s.empty:
            ax.plot(s.x, s.density, color=color, lw=1.4, label=lab)
    ax.axvline(GNK_TRUE[param], color="black", ls=":", lw=0.8, alpha=0.5)
    ax.set_title(f"${param}$")
    ax.set_yticks([])
    if c == 0:
        ax.legend(frameon=False, fontsize=8)
fig.suptitle("BSL vs NUTS reference (g-and-k, n=1000, seed 0)", y=1.05)
fig.tight_layout()
plt.show()
"""))

    return cells


# ---------------------------------------------------------------------------
# Section 3: internal diagnostics
# ---------------------------------------------------------------------------

def section3_cells() -> list[nbformat.NotebookNode]:
    cells: list[nbformat.NotebookNode] = []
    cells.append(md("# 3. Internal diagnostics"))
    cells.append(md(
        "Supporting checks, not for the paper. Each answers a specific question "
        "about whether the Section 1 results are sound. Kept short, with the "
        "reason stated up front."
    ))

    cells.append(md(
        "## 3.1 Posterior Gaussianity and the small-n regime\n"
        "\n"
        "Why this matters: the theory assumes a near-Gaussian (Bernstein-von "
        "Mises) posterior. To check this we take the g-and-k reference posterior "
        "and compute the KL to its own best-fitting Gaussian (matched mean and "
        "covariance). Zero means the reference is exactly Gaussian; larger means "
        "further from Gaussian. The figure shows this by $n$: small at the "
        "larger $n$ used in the headline results, much larger at $n=100$, which "
        "is the regime where the near-Gaussian assumption breaks down (and where "
        "Gaussian-NPE should not be expected to do well)."
    ))
    cells.append(code(r"""
oracle = pd.read_csv(DATA / "gnk_theta_oracle_by_n.csv").sort_values("n")
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.plot(oracle.n, oracle.K_theta_star_median, "o-", color=C_GAUSS)
if {"K_theta_star_q25", "K_theta_star_q75"} <= set(oracle.columns):
    ax.fill_between(oracle.n, oracle.K_theta_star_q25, oracle.K_theta_star_q75,
                    color=C_GAUSS, alpha=0.15, label="q25-q75 over seeds")
    ax.legend(frameon=False, fontsize=8)
ax.set_xscale("log")
ax.axhline(0, color="black", ls="--", lw=0.8, alpha=0.5)
ax.set_xlabel("number of observations $n$")
ax.set_ylabel("KL(reference || its best Gaussian fit)")
ax.set_title("How non-Gaussian is the g-and-k reference posterior?\n"
             "(0 = exactly Gaussian; lower is more Gaussian)")
fig.tight_layout()
plt.show()
"""))

    cells.append(md(
        "## 3.2 NPE does not fully recover the g-and-k reference\n"
        "\n"
        "Why this matters: if NPE cannot recover the g-and-k posterior even at "
        "the largest budget, that would weaken the g-and-k support for the "
        "theory. The point of this section is to see how large the gap is and "
        "where it comes from.\n"
        "\n"
        "At $n=1000$, seed 0 the NPE posterior does not sit exactly on the "
        "reference; the gap is mostly on $g$ and $k$. This is not specific to "
        "the Gaussian family. The flow variant in Section 2.1 misses in much the "
        "same way, so it is a property of NPE here, not of the Gaussian "
        "approximation. BSL recovers the reference closely (Section 2.2), so the "
        "reference itself is not in doubt.\n"
        "\n"
        "Two things change the match, both recorded as context only and neither "
        "a paper result:\n"
        "\n"
        "- Standardisation. The prior-predictive octile summaries span many "
        "orders of magnitude, so plain per-coordinate z-scoring compresses the "
        "informative region. Robust (asinh / median-IQR) standardisation "
        "improves the match and does not change the method or the theory's "
        "assumptions, so it is an admissible default. It would mean re-running "
        "the whole grid, so it is not a change to make lightly.\n"
        "- Rejection-ABC trimming. This also improves the match, but only by "
        "discarding simulations, which is preconditioning and does not match the "
        "theory's assumptions. Shown for contrast, not as a fix.\n"
        "\n"
        "The overlay below makes this easier to read than the KL numbers: it "
        "shows the reference, the two standardisations, rejection-ABC and BSL on "
        "the same axes, per parameter."
    ))
    cells.append(code(r"""
ov = pd.read_csv(DATA / "gnk_improvement_overlay.csv")
methods = [
    ("NUTS reference", C_REF, "-", "reference"),
    ("Gaussian-NPE (z-score)", C_GAUSS, "--", "Gaussian-NPE (vanilla z-score)"),
    ("Gaussian-NPE (robust asinh)", "#9467bd", "-", "Gaussian-NPE (robust asinh)"),
    ("Gaussian-NPE (rejection-ABC)", "#ff7f0e", "-", "Gaussian-NPE (rejection-ABC)"),
    ("BSL", C_BSL, ":", "BSL"),
]
fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
for c, param in enumerate(["A", "B", "g", "k"]):
    ax = axes[c]
    d = ov[ov.param == param]
    for method, color, ls, lab in methods:
        s = d[d.method == method].sort_values("x")
        if not s.empty:
            ax.plot(s.x, s.density, color=color, ls=ls, lw=1.3, label=lab)
    ax.axvline(GNK_TRUE[param], color="black", ls=":", lw=0.8, alpha=0.5)
    ax.set_title(f"${param}$")
    ax.set_yticks([])
    if c == 0:
        ax.legend(frameon=False, fontsize=6)
fig.suptitle("g-and-k posterior: vanilla vs robust vs rejection-ABC vs "
             "reference (n=1000, seed 0)", y=1.05)
fig.tight_layout()
plt.show()
"""))
    cells.append(md("The same comparison as KL from the reference:"))
    cells.append(code(r"""
rej = json.loads((DATA / "gnk_rejection_abc_summary.json").read_text())
rows = [
    ("vanilla z-score Gaussian-NPE", rej.get("vanilla_canonical_kl")),
    ("robust asinh Gaussian-NPE", rej.get("robust_canonical_kl")),
    ("rejection-ABC Gaussian-NPE", rej.get("kl_reference_to_npe_value")),
    ("BSL (reference cross-check)", rej.get("bsl_unique_kl")),
]
summary = pd.DataFrame(
    [(name, round(v, 3)) for name, v in rows if v is not None],
    columns=["method (g-and-k, n=1000, seed 0)", "KL(reference || .)"])
display(summary)
"""))

    cells.append(md(
        "## 3.3 Dimension scaling (pilot, the most direct test of the theory)\n"
        "\n"
        "This is the experiment that bears most directly on the central result. "
        "The theory says the budget must grow like $N \\gtrsim d^2 n$ to control "
        "the KL from the reference. So if we hold $N/(d^2 n)$ fixed and vary the "
        "summary dimension $d_s$, the KL should stay roughly controlled rather "
        "than blow up. This pilot fixes $n=500$, holds $N/(d^2 n) = 5$, and "
        "ranges $d_s$ over $\\{5,7,11,15,19\\}$ (so $d = d_s + d_\\theta$).\n"
        "\n"
        "Read honestly, the pilot does not yet show what we want: the median KL "
        "rises with $d$ rather than staying flat. This is the opposite of the "
        "behaviour the theory predicts, so it is the result we most need to "
        "understand before leaning on it. Three things could drive the rise "
        "without contradicting the theory, and none is resolved here: the "
        "constant $5$ may be too small (the condition is asymptotic in $N$); "
        "$n=500$ may be too small, so the g-and-k posterior is still somewhat "
        "non-Gaussian at this $n$ (Section 3.1); and it is worth pinning down "
        "against the exact theorem statement how much of the rise in raw KL is "
        "expected from the theory's own $d$-dependence versus a genuine shortfall "
        "in the budget. The plan (see Open questions) is to rerun at $n=1000$, "
        "sweep the constant rather than fix it at $5$, and add the MA(2) "
        "compatible case as a cleaner near-Gaussian second example. Treat the "
        "figure below as a provisional pilot."
    ))
    cells.append(code(r"""
dim = pd.read_csv(DATA / "dim_scaling_pilot_kl_by_d.csv")
finite = dim[dim["theta_kl"].notna()] if "theta_kl" in dim.columns else dim
if "d_s" in finite.columns and "theta_kl" in finite.columns:
    g = (finite.groupby(["method", "d_s"])["theta_kl"]
               .median().reset_index())
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for raw, color in [("flow_npe", C_FLOW), ("gaussian_npe", C_GAUSS)]:
        s = g[g.method == raw].sort_values("d_s")
        if not s.empty:
            ax.plot(s.d_s, s.theta_kl, "o-", color=color,
                    label=METHOD_NAME.get(raw, raw))
    ax.set_xlabel(r"summary dimension $d_s$  (budget held at $N=5\,d^2 n$, $n=500$)")
    ax.set_ylabel("median KL from reference")
    ax.set_title("Dimension-scaling pilot (n=500, N=5 d^2 n):\n"
                 "KL rises with d (provisional, see text)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    plt.show()
else:
    print("dim-scaling columns:", list(dim.columns))
    display(dim.head())
"""))

    return cells


# ---------------------------------------------------------------------------
# Open questions (internal, removed before sending)
# ---------------------------------------------------------------------------

def open_questions_cells() -> list[nbformat.NotebookNode]:
    """Internal: honest open items and notes to take into the group meeting.

    Removed before sending. Replaces the old decision-support scaffold.
    """
    cells: list[nbformat.NotebookNode] = []
    cells.append(md("---"))
    cells.append(md(
        "# Open questions and notes for discussion (internal, remove before sending)\n"
        "\n"
        "Honest open items to take into the group meeting. Several need a rerun "
        "or extended reasoning, not a report edit, and are flagged as such."
    ))
    cells.append(md(
        "**MA(2), well-specified case ($\\delta_0=1$).** This was examined and the "
        "earlier discrepancy traced to a bug, since addressed. It connects to the "
        "point David Warne raised in the meeting and does not reflect any problem "
        "with the results in the published arXiv version. The well-specified panel "
        "is currently out of the report; whether it returns as an appendix "
        "baseline is a minor call."
    ))
    cells.append(md(
        "**g-and-k KL at $n=5000,\\ N=n^2$ (1B.1).** At the largest budget the "
        "median KL rises from $n=1000$ to $n=5000$ rather than falling, which "
        "runs against the theory and against the paper's appendix wording. Needs "
        "follow-up before the appendix table is final: this cell has 91 of 101 "
        "seeds and one unstable flow seed (KL ~24), so check whether the rise "
        "survives the full seed set, or whether it is a genuine effect to "
        "explain. (Rerun / data.)"
    ))
    cells.append(md(
        "**Oracle floor vs the residual KL (1B.1).** The moment-matched-Gaussian "
        "floor is low, so most of the remaining KL sits above it. Calling that "
        "gap finite-$N$ amortisation error is asserted, not shown. Worth "
        "quantifying, since a non-trivial unexplained gap is a sore point for the "
        "theory. (Analysis.)"
    ))
    cells.append(md(
        "**Stereological $\\sigma$ and $\\xi$ (1B.2).** NPE recovers the rate "
        "$\\lambda$ well but does noticeably worse on $\\sigma$ and $\\xi$. The "
        "report should not look like it foregrounds $\\lambda$ and buries the "
        "rest. Decide how to present $\\sigma$ and $\\xi$ honestly. (Framing.)"
    ))
    cells.append(md(
        "**Why NPE underperforms BSL on g-and-k (3.2).** First, confirm the gap "
        "is not large enough to weaken the g-and-k support for the theory. "
        "Second, a useful follow-up is to show that sequential NPE or "
        "preconditioned NPE (with SMC-ABC) recovers the reference here. That "
        "would locate the gap in the one-shot amortised NPE setup rather than in "
        "neural methods as a class, and move the discussion toward why NPE "
        "misses. A GPT-5.5 Pro literature brief on improving standardisation "
        "within the theory's assumptions also fits here. (Rerun + Pro.)"
    ))
    cells.append(md(
        "**Dimension scaling (3.3).** The most direct test, and it does not yet "
        "show the predicted flatness. Rerun at $n=1000$, sweep the constant in "
        "$N = c\\,d^2 n$ rather than fixing $c=5$, and add the MA(2) compatible "
        "case as a near-Gaussian second example. Separately, pin down against the "
        "theorem whether the comparison should be on raw KL (my reading) or some "
        "normalised quantity. (Rerun + Pro / coauthors.)"
    ))
    cells.append(md(
        "**Posterior Gaussianity for the other examples (3.1).** The "
        "best-Gaussian-fit-by-$n$ check is only done for g-and-k. The same plot "
        "for the stereological and MA(2) references would show whether the "
        "near-Gaussian assumption holds there too. (Rerun.)"
    ))
    cells.append(md(
        "**Coverage audit.** Confirm every main-text and appendix result in the "
        "paper is represented here and matches, including the choice of median "
        "(this report) vs mean with standard deviation (paper appendix KL "
        "table). (Audit.)"
    ))
    return cells


def closing_cells() -> list[nbformat.NotebookNode]:
    return [
        md("---"),
        todo(
            "Before sending: delete the internal note at the top and the "
            "Open-questions section above, strip all TODO notes, and confirm code "
            "cells are hidden."
        ),
    ]


# ---------------------------------------------------------------------------

def build() -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.cells = (
        header_cells()
        + section1_cells()
        + section2_cells()
        + section3_cells()
        + open_questions_cells()
        + closing_cells()
    )
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    # Title/date so the exported PDF is not the nbconvert default "Notebook".
    nb.metadata["title"] = "NPE convergence: Gaussian NPE empirical results"
    nb.metadata["authors"] = [{"name": "Ryan Kelly"}]
    nb.metadata["date"] = "2026-05-31"
    return nb


def main() -> None:
    stage_inputs()
    build_ma2_three_case()
    build_gnk_improvement_overlay()

    nb = build()
    nb_path = HERE / "coauthor_report.ipynb"
    nbformat.write(nb, nb_path)
    print(f"wrote {nb_path.name} with {len(nb.cells)} cells")

    try:
        from nbconvert.preprocessors import ExecutePreprocessor

        ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
        ep.preprocess(nb, {"metadata": {"path": str(nb_path.parent)}})
        nbformat.write(nb, nb_path)
        print("executed notebook")
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"execution skipped: {exc}")

    out_pdf = nb_path.with_suffix(".pdf")
    try:
        from nbconvert import PDFExporter

        pdf_exporter = PDFExporter()
        pdf_exporter.exclude_input = True
        pdf_exporter.exclude_output_prompt = True
        # PDFExporter's default raw_mimetypes drops text/latex raw cells, which
        # would strip our preamble tweak (manual section numbering). Keep it.
        pdf_exporter.raw_mimetypes = ["text/latex", "application/pdf", ""]
        body, _ = pdf_exporter.from_notebook_node(nb)
        out_pdf.write_bytes(body)
        print(f"wrote {out_pdf.name}")
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"PDF export skipped: {exc}")


if __name__ == "__main__":
    main()
