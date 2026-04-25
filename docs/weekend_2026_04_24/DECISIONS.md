# Decisions

Record only actual decisions. Keep each entry short enough that future readers can recover why the empirical path was chosen.

## Template

```markdown
## YYYY-MM-DD: decision title

Decision: ...
Reason: ...
Implication: ...
Evidence:
- ...
```

## Pending decisions

- Which scaled-budget grid to use for high-budget GNK Gaussian-NPE.
- Whether flow-NPE endpoints are worth running in the high-budget array.
- Whether gnk_model simulator-control pilot results require a paper-framing change.

## 2026-04-25: GNK n=500 oracle gate

Decision: Treat n=500 as acceptable for the main high-budget BvM-rate curve under the median-based gate.
Reason: The cached theta-space Gaussian oracle KL has median 0.099533 nats over 101 seeds, just below the <= 0.1 nats acceptability threshold.
Implication: HPC calibration may use n=500, while noting that the acceptance is borderline because the IQR is [0.077337, 0.135198]. Full-array submission remains blocked pending one calibration job and explicit dry-run review.
Evidence:
- `notebooks/plots/gnk_n500_oracle_gate_20260425_summary.json`
- `notebooks/plots/gnk_n500_oracle_gate_20260425_per_seed.csv`

## 2026-04-25: GNK coordinate-aware Gaussian-NPE decomposition

Decision: Use the u-space decomposition output from reviewed implementation commit `1da4752` as the native-coordinate Gaussian-NPE diagnostic, while retaining K_theta^* as the theta-space BvM target-Gaussianity diagnostic.
Reason: Scientific-code review found the KL directions and analytic Gaussian-Gaussian decomposition correct after qualifying finite-sample kNN affine-invariance wording and excluding the one-seed `N=31623` diagnostic row from the main Delta component plot.
Implication: Downstream log-corrected scaled-budget plots may consume `notebooks/plots/gnk_u_space_kl_decomp_20260425_per_seed.csv`, excluding `N=n` and incomplete seed-count groups from main theorem-facing panels unless explicitly labelled diagnostic.
Evidence:
- `notebooks/plots/gnk_u_space_kl_decomp_20260425_per_seed.csv`
- `notebooks/plots/gnk_u_space_kl_decomp_20260425_summary.json`
- `notebooks/plots/gnk_u_space_kl_decomp_20260425_N_gt_n_plot_metadata.json`
- `notebooks/plots/gnk_eta_vs_u_oracle_check_20260425.json`

## 2026-04-25: GNK log-corrected scaled-budget diagnostic

Decision: Include both the original `x = N/(d_total^2 n)` panel and the log-corrected `x_log = N/(d_total^2 n log(N)^2)` panel in the accepted GNK Gaussian-NPE native u-space diagnostic figure set.
Reason: The two-panel figure preserves the existing scaled-budget view while retaining the BvM corollary's logarithmic factor in the finite-N visualisation. Natural logs are used, and the main theorem-facing filters remain `N>n` with complete 101-seed `(n,N)` groups only.
Implication: Priority 4 is complete. The plot is a diagnostic re-expression of the existing decomposition CSV, not new compute or a new rate; high-budget HPC remains blocked pending one calibration job and dry-run review.
Evidence:
- `notebooks/plots/gnk_u_space_kl_decomp_20260425_N_gt_n_delta_u_total_scaled_budget_log_corrected.pdf`
- `notebooks/plots/gnk_u_space_kl_decomp_20260425_N_gt_n_plot_metadata.json`
