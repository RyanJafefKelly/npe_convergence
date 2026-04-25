# Results

This file records empirical outputs, neutral interpretations, and links back to commands, scripts, cache paths, commit hashes, and timestamps.

## Conventions

- Keep theta-space BvM diagnostics separate from u-space Gaussian-NPE diagnostics.
- State cache paths and whether caches were read-only.
- State the exact script or command used.
- State the commit hash used for every generated figure or table.
- For figures, record output path, timestamp, source data path, and plotting script.
- Keep MA2-b0 interpretations separate from BvM-rate evidence.

## n=500 BvM Oracle Gate

Status: completed.

Computed the theta-space moment-matched Gaussian oracle KL using the existing
`compute_kl_oracle` convention: fit `G_theta^*` on the first 5,000 NUTS samples
and evaluate Perez-Cruz KL on 2,000 held-out NUTS samples versus 2,000 samples
from the fitted Gaussian.

- Command: `python scripts/compute_gnk_n500_oracle_gate.py --n-obs 500 --seeds 0,100 --cache-dir res/gnk --output-dir notebooks/plots --output-prefix gnk_n500_oracle_gate_20260425`
- Created at: `2026-04-24T23:56:37.145233+00:00`.
- Run-time commit hash: `5d0c02f`; this records the code state when the oracle gate was computed. The task script, outputs, and documentation are archived in the enclosing task commit on branch `exp-oracle-gate-n500`.
- Cache path used: `res/gnk/nuts_cache_v2_n_obs_500_seed_{0..100}.pkl`; individual cache paths are listed in the per-seed CSV.
- Cache access: read-only; the script opens NUTS pickle caches in `rb` mode and writes only the CSV/JSON outputs listed below.
- Cache provenance/checksums: not computed for this gate; exact per-seed cache filenames are recorded in the CSV.
- KL estimator details: Perez-Cruz estimator via `npe_convergence.metrics.kullback_leibler`; fit `G_theta^*` on NUTS samples `0:5000`; evaluate on held-out NUTS samples `5000:7000`; draw 2,000 Gaussian oracle samples using `np.random.default_rng(seed)`, where `seed` is the posterior seed.
- Number of seeds: 101.
- Posterior samples per seed: 10,000.
- K_theta^*(n=500) median: 0.099533 nats.
- K_theta^*(n=500) IQR: [0.077337, 0.135198] nats.
- Recommendation: use n=500 for the high-budget curve by the median-based `<= 0.1` nats decision rule, but note that the median is close to the threshold and the upper quartile exceeds 0.1.
- Output path:
  - `notebooks/plots/gnk_n500_oracle_gate_20260425_per_seed.csv`
  - `notebooks/plots/gnk_n500_oracle_gate_20260425_summary.json`

## Coordinate Reconciliation and u-space KL Decomposition

Status: completed after scientific-code review.

Implemented a coordinate-aware Gaussian-NPE decomposition for cached GNK runs.
The script reads existing NUTS and Gaussian-NPE sample caches only, reconstructs
`Qhat_N,u` from saved Gaussian-NPE posterior sample moments in the logit
coordinate. Exact KLs and analytic Gaussian-Gaussian decomposition terms are
invariant to the common affine standardisation by `mu_eta/sigma_eta`; finite-
sample kNN oracle estimates are eta-coordinate estimates of the same exact
u-space target and may differ slightly after diagonal standardisation.

- Subset check command: `python scripts/compute_gnk_u_space_kl_decomp.py --n-values 500 --N-values 500 --seeds 0,1 --output-prefix gnk_u_space_kl_decomp_20260425_subset`
- Subset plot command: `python scripts/plot_gnk_u_space_kl_decomp.py --input-csv notebooks/plots/gnk_u_space_kl_decomp_20260425_subset_per_seed.csv --output-prefix gnk_u_space_kl_decomp_20260425_subset`
- Full table command: `python scripts/compute_gnk_u_space_kl_decomp.py --output-prefix gnk_u_space_kl_decomp_20260425`
- Main plot command: `python scripts/plot_gnk_u_space_kl_decomp.py --input-csv notebooks/plots/gnk_u_space_kl_decomp_20260425_per_seed.csv --output-prefix gnk_u_space_kl_decomp_20260425_N_gt_n --exclude-N-equals-n --min-seeds 101`
- Eta-vs-standardised-u robustness command: `python scripts/check_gnk_eta_vs_u_oracle.py`
- Run-time git hash recorded in outputs: `4df05eb`; reviewed implementation/output commit `1da4752` archives the scripts, documentation, and generated outputs.
- CSV/summary timestamp after the clipping-count/doc-caveat rerun: `2026-04-25T01:43:25.673014+00:00`.
- Figure metadata timestamp after excluding the one-seed diagnostic row: `2026-04-25T01:50:51.693698+00:00`.
- Cache paths used: `res/gnk/nuts_cache_v2_n_obs_{n}_seed_{seed}.pkl`, `res/gnk/nuts_cache_v2_flow_n_obs_{n}_seed_{seed}.pkl`, and `res/gnk/gaussian_npe_n_obs_{n}_n_sims_{N}_seed_{seed}/posterior_samples.pkl`.
- Cache access: read-only; no cache files were overwritten.
- Output table path: `notebooks/plots/gnk_u_space_kl_decomp_20260425_per_seed.csv`.
- Summary metadata path: `notebooks/plots/gnk_u_space_kl_decomp_20260425_summary.json`.
- Figure paths:
  - `notebooks/plots/gnk_u_space_kl_decomp_20260425_N_gt_n_delta_u_mean_cov.pdf`
  - `notebooks/plots/gnk_u_space_kl_decomp_20260425_N_gt_n_coord_offset_vs_n.pdf`
  - `notebooks/plots/gnk_u_space_kl_decomp_20260425_N_gt_n_plot_metadata.json`
- Eta-vs-standardised-u robustness output:
  - `notebooks/plots/gnk_eta_vs_u_oracle_check_20260425.csv`
  - `notebooks/plots/gnk_eta_vs_u_oracle_check_20260425.json`
- Rows: 1,617 per-seed/per-config rows over `n in {100, 500, 1000, 5000}`. The raw table includes `N=n` rows as low-budget diagnostics and retains the one-seed legacy `n=1000,N=31623` row; the main Delta component figure excludes `N=n` and requires at least 101 seeds per `(n,N)` group.
- Grid completeness:
  - n=100: N=100, 460, 1000, 10000 each have 101 seeds.
  - n=500: N=500, 3107, 11180, 250000 each have 101 seeds.
  - n=1000: N=1000, 6907, 31622, 1000000 each have 101 seeds; N=31623 has 1 seed and is retained as a legacy single-run diagnostic.
  - n=5000: N=5000, 42585, 353553, 25000000 each have 101 seeds.
- K_theta^* median by n: n=100: 0.669815; n=500: 0.099535; n=1000: 0.048707; n=5000: 0.005491 nats.
- K_u^* median by n: n=100: 0.556715; n=500: 0.081014; n=1000: 0.043254; n=5000: -0.007986 nats.
- Coordinate offset median by n: n=100: -0.088838; n=500: -0.013023; n=1000: -0.009555; n=5000: -0.007648 nats.
- Delta_N,u over all raw rows: total median 4.871100 nats, IQR [3.337669, 7.171507]; mean component median 1.008930; covariance component median 3.776737.
- Delta_N,u over `N>n` rows: total median 4.055106 nats, IQR [2.923939, 5.440763]; mean component median 0.861482; covariance component median 3.261894.
- Median Delta_N,u split by `(n,N)` for `N>n`:
  - n=100: N=460 total 6.083045, mean 1.752549, cov 4.161703; N=1000 total 5.314056, mean 1.659089, cov 3.665982; N=10000 total 3.285552, mean 1.082266, cov 2.107228.
  - n=500: N=3107 total 6.860109, mean 1.569612, cov 5.344438; N=11180 total 4.441182, mean 0.668807, cov 3.750724; N=250000 total 2.634332, mean 0.437295, cov 2.103115.
  - n=1000: N=6907 total 5.722742, mean 0.908342, cov 4.832505; N=31622 total 3.777168, mean 0.429033, cov 3.327859; N=31623 total 5.095534, mean 0.980671, cov 4.114863; N=1000000 total 2.386606, mean 0.417457, cov 1.819056.
  - n=5000: N=42585 total 4.418036, mean 0.682200, cov 3.704735; N=353553 total 2.933239, mean 0.541313, cov 2.353513; N=25000000 total 2.602777, mean 0.514448, cov 2.069701.
- Self-consistency check: reconstructed `Qhat_N,u` versus fresh samples from the reconstructed Gaussian has median Perez-Cruz KL 0.001694 nats over all rows and 0.001446 nats over `N>n` rows, consistent with saved samples matching their reconstructed u-space Gaussian. Earlier free-text audit notes reported fresh-resample self-consistency near zero; no machine-readable cache for those diagnostics was found, so this script recomputes the check.
- Eta-vs-standardised-u robustness check: exact KL and analytic Gaussian-Gaussian terms are affine invariant, but finite-sample kNN estimates need not be invariant to diagonal standardisation. For representative checks `(n,N,seed)=(100,1000,0),(500,3107,0),(5000,42585,0)`, the standardised-u minus eta K_u^* differences were -0.016438, +0.007653, and -0.001168 nats, respectively. These are small relative to the main medians/IQRs, but K_u^* should be described as a finite-sample eta-coordinate kNN estimate of the same exact u-space target.
- Clipping check: the logit transform clips theta to `[1e-6, 10-1e-6]`. Gaussian-NPE sample clipping count is zero across the full table. NUTS clipping count is one sample total across unique `(n,seed)` NUTS caches, so clipping is negligible.
- Numerical note: sample-based Perez-Cruz oracle KL estimates were retried with deterministic Gaussian jitter at scale `eps * max(pooled_sd, 1.0)` only when duplicate/zero nearest-neighbour distances produced non-finite values. This occurred for 26 unique n=100 seeds for both theta-space and u-space oracle KL, appearing as 104 repeated rows in the per-config table for each KL type because oracle values are repeated across N for a fixed `(n, seed)`. No n=500, n=1000, or n=5000 oracle rows needed jitter. Analytic Gaussian-Gaussian `Delta_N,u` terms were finite without this estimator.
- Interpretation caveat: `K_theta^*` is the theta-space BvM-premise diagnostic. `K_u^* - K_theta^*` is the coordinate-projection offset. `Delta_N,u` is native u-space Gaussian-NPE approximation error, not a pure BvM residual.

## Log-Corrected Scaled-Budget Plots

Status: completed.

Added a companion scaled-budget figure derived from the existing reviewed GNK
u-space Gaussian-NPE decomposition CSV, not from a fresh compute pass. The
figure preserves the original scaled budget axis `x = N/(d_total^2 n)` in the
left panel and adds the log-corrected finite-N axis
`x_log = N/(d_total^2 n log(N)^2)` in the right panel, with a shared y-axis for
median `Delta_N,u`. The dotted reference overlay is proportional to
`log(N)/sqrt(N/(d_total^2 n))`, equivalently `1/sqrt(x_log)` in the
log-corrected panel.

- Command: `python scripts/plot_gnk_u_space_kl_decomp.py --input-csv notebooks/plots/gnk_u_space_kl_decomp_20260425_per_seed.csv --output-prefix gnk_u_space_kl_decomp_20260425_N_gt_n --exclude-N-equals-n --min-seeds 101`
- Commit hash recorded in output metadata: `f2b995e`.
- Created at: `2026-04-25T06:24:44.778617+00:00`.
- Plotting script: `scripts/plot_gnk_u_space_kl_decomp.py`.
- Input table path: `notebooks/plots/gnk_u_space_kl_decomp_20260425_per_seed.csv`.
- Cache access: no cache files were read or overwritten by this plotting command.
- Main plotting filter: excludes `N=n` and requires at least 101 seeds per `(n,N)` group, leaving 12 plotted groups and 1,212 per-seed rows.
- Figure paths:
  - `notebooks/plots/gnk_u_space_kl_decomp_20260425_N_gt_n_delta_u_mean_cov.pdf`
  - `notebooks/plots/gnk_u_space_kl_decomp_20260425_N_gt_n_coord_offset_vs_n.pdf`
  - `notebooks/plots/gnk_u_space_kl_decomp_20260425_N_gt_n_delta_u_total_scaled_budget_log_corrected.pdf`
  - `notebooks/plots/gnk_u_space_kl_decomp_20260425_N_gt_n_plot_metadata.json`
- Whether natural logs were used: yes. The current scaled-budget plotting script did not previously specify a log-correction convention, so the new `x_log` panel uses natural logs.
- Neutral interpretation: the figure plots median `Delta_N,u`, the native u-space Gaussian-NPE error, after excluding `N=n` and incomplete seed-count groups. The `d_total^2 n` scaling organises the curves across `n` better than raw budget alone. The log-corrected scaled-budget panel shows somewhat improved cross-`n` alignment relative to the original scaled budget, especially in the low-to-mid cached budget range, but residual separation remains. The log-corrected axis retains the BvM corollary's existing logarithmic factor in the finite-`N` visualisation; it is not a new rate. The log factor helps explain why apparently large values of `N/(d_total^2 n)` can remain pre-asymptotic at the cached budgets.

## HPC Calibration

Status: pending after n=500 oracle gate. Full high-budget array submission remains blocked pending one calibration job and explicit dry-run review.

Required record:

- Selected n: 500.
- Calibration x:
- Resolved N:
- Seed:
- Dry-run command:
- Submit command:
- Output directory:
- Timing metadata path:
- Validation curve path:
- Predicted Gaussian-NPE u-space mu/Sigma path:
- Sample path:
- Recommendation for full array:

## gnk_model Simulator-Control Pilot

Status: blocked pending calibration infrastructure.

Required record:

- Simulator flag/name:
- Command:
- Commit hash:
- Output directory:
- x grid:
- Seeds:
- Interpretation:

## Hexadecile Aggregation

Status: pending.

Required record:

- Command:
- Commit hash:
- Cache path:
- d_s:
- d_theta:
- d_total:
- Figure paths:
- Interpretation caveat: failures to collapse may reflect changed summary map or conditioning, not theory failure.

## MA2-b0 Compatibility Figure

Status: pending.

Required record:

- Command:
- Commit hash:
- Reference cache status:
- Output paths:
- Interpretation: compatibility failure only; exclude from pooled BvM-rate fits.
