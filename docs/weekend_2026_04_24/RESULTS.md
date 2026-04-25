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

Status: pending.

Required record:

- Command:
- Commit hash:
- Cache paths used:
- Output table path:
- Figure paths:
- K_theta^* summary:
- K_u^* summary:
- Coordinate offset summary:
- Delta_N,u total summary:
- Delta_N,u mean/cov split:
- Self-consistency check:
- Interpretation caveat: Delta_N,u is native-coordinate Gaussian-NPE error, not pure BvM residual.

## Log-Corrected Scaled-Budget Plots

Status: pending.

Required record:

- Command:
- Commit hash:
- Input table path:
- Figure paths:
- Whether natural logs were used:
- Neutral interpretation:

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
