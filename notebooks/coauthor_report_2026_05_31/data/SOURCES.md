# Data sources for the coauthor report

`build_notebook.py` reads from this `data/` directory. The skeleton ships with
placeholder cells that do not load anything, so it builds with an empty `data/`.
When you wire up a real figure, vendor its source file into here so the report
folder stays self-contained (useful for the Zenodo bundle).

Each file below maps to where it currently lives. Paths are relative to the repo
root.

## Section 1 (paper-ready)

| file | source |
| --- | --- |
| gnk_kl_flow_vs_gaussian.csv | notebooks/meeting_2026_05_18/data/ |
| gnk_kl_paired_per_seed.csv | notebooks/meeting_2026_05_18/data/ |
| gnk_paper_grid_bootstrap_ci.csv | notebooks/meeting_2026_05_18/data/ |
| coverage_paper_grid_all_params.csv | notebooks/plots/gnk_task2_20260502/ |
| bias_g_paper_figure_values.csv | notebooks/plots/gnk_task2_20260502/ |
| gnk_hexadecile_gaussian.csv | notebooks/meeting_2026_05_18/data/ |
| gnk_hexadecile_flow.csv | notebooks/meeting_2026_05_18/data/ |
| ma2_compatibility_gaussian.csv | notebooks/meeting_2026_05_18/data/ |
| ma2_compatibility_flow.csv | notebooks/meeting_2026_05_18/data/ |
| ma2_b0_kl.csv | notebooks/meeting_2026_05_18/data/ |
| ma2_b0_per_seed.csv | notebooks/meeting_2026_05_18/data/ |
| ma2_delta1_refresh.csv | notebooks/meeting_2026_05_18/data/ |
| stereological_coverage.csv | notebooks/meeting_2026_05_18/data/ |
| stereological_bias_by_seed.csv | notebooks/meeting_2026_05_18/data/ |
| stereological_posterior_overlay.csv | notebooks/meeting_2026_05_18/data/ |

## Section 2 (sanity checks)

| file | source |
| --- | --- |
| gnk_posterior_overlay.csv | notebooks/meeting_2026_05_18/data/ |
| ma2_posterior_overlay_seed_22.csv | notebooks/meeting_2026_05_18/data/ |
| gnk_bsl_diagnostic.csv | notebooks/meeting_2026_05_18/data/ |

## Section 3 (internal diagnostics)

| file | source |
| --- | --- |
| gnk_theta_oracle_by_n.csv | notebooks/meeting_2026_05_18/data/ |
| gnk_rejection_abc_summary.json | notebooks/meeting_2026_05_18/data/ |
| gnk_robust_scaling_overlay.csv | notebooks/meeting_2026_05_18/data/ |
| dim_scaling_pilot_kl_by_d.csv | notebooks/meeting_2026_05_18/data/ |

## Vendor everything at once

Run from the repo root to copy the current files into this folder:

```bash
D=notebooks/coauthor_report_2026_05_31/data
M=notebooks/meeting_2026_05_18/data
T=notebooks/plots/gnk_task2_20260502
cp "$M"/gnk_kl_flow_vs_gaussian.csv "$M"/gnk_kl_paired_per_seed.csv \
   "$M"/gnk_paper_grid_bootstrap_ci.csv "$M"/gnk_hexadecile_gaussian.csv \
   "$M"/gnk_hexadecile_flow.csv "$M"/ma2_compatibility_gaussian.csv \
   "$M"/ma2_compatibility_flow.csv "$M"/ma2_b0_kl.csv "$M"/ma2_b0_per_seed.csv \
   "$M"/ma2_delta1_refresh.csv "$M"/stereological_coverage.csv \
   "$M"/stereological_bias_by_seed.csv "$M"/stereological_posterior_overlay.csv \
   "$M"/gnk_posterior_overlay.csv "$M"/ma2_posterior_overlay_seed_22.csv \
   "$M"/gnk_bsl_diagnostic.csv "$M"/gnk_theta_oracle_by_n.csv \
   "$M"/gnk_rejection_abc_summary.json "$M"/gnk_robust_scaling_overlay.csv \
   "$M"/dim_scaling_pilot_kl_by_d.csv "$D"/
cp "$T"/coverage_paper_grid_all_params.csv "$T"/bias_g_paper_figure_values.csv "$D"/
```
