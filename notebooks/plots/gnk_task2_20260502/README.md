# GNK Task 2 Outputs

All artifacts in this directory were generated from existing GNK octile caches and reviewed diagnostic CSV/JSON inputs. No cache files were modified.

Closeout and interpretation:

- `docs/weekend_2026_05_02/GNK_TASK2_CLOSEOUT.md`
- `gnk_task2_octile_hpc_rerun_note.md`

Key separation:

- `theta_oracle_by_n.csv` contains theta-space `K_theta^*` only.
- `gaussian_u_space_delta_N_u_by_n_N.csv` contains native Gaussian-NPE `Delta_N,u` diagnostics only.
- The one-row `gaussian_npe_n_obs_1000_n_sims_31623` cache group is excluded from complete-group summaries.

Main caveats:

- Flow-NPE `n=5000, N=n^2` has 65 complete artifact seeds, not 101; missing artifact seeds are `0-35`.
- The `n=100` raw theta-KL summaries use 72 finite paired seeds because some `kl.txt` values are non-finite.
- Flow seed `41` has non-finite replicate-level bias diagnostics in two cells; see the closeout before using bias summaries as paper-facing values.

Generated files:
- `bias_boxplot_values_paper_grid.csv`
- `bias_g_paper_figure_values.csv`
- `bias_summary_paper_grid.csv`
- `complete_group_seed_count_inventory.csv`
- `coverage_paper_grid_all_params.csv`
- `coverage_paper_grid_table_wide.csv`
- `gaussian_u_space_delta_N_u_by_n_N.csv`
- `gaussian_u_space_delta_N_u_excluded_groups.csv`
- `gaussian_u_space_delta_N_u_per_seed.csv`
- `gnk_task2_octile_hpc_rerun_note.md`
- `paper_grid_seed_count_inventory.csv`
- `raw_theta_kl_paired_per_seed.csv`
- `raw_theta_kl_summary_by_method.csv`
- `raw_theta_kl_summary_comparable.csv`
- `raw_theta_kl_summary_complete_groups.csv`
- `run_summary.json`
- `seed_count_inventory_all.csv`
- `theta_oracle_by_n.csv`
- `theta_oracle_n500_gate_crosscheck.csv`
