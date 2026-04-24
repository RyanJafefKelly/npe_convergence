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

Status: pending.

Required record:

- Command:
- Commit hash:
- Cache path used:
- Number of seeds:
- Posterior samples per seed:
- K_theta^*(n=500) median:
- K_theta^*(n=500) IQR:
- Recommendation: use n=500 or n=1000 for high-budget curve.
- Output path:

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

Status: blocked pending n=500 oracle gate.

Required record:

- Selected n:
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

