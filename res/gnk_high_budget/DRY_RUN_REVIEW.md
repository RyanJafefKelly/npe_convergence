# GNK High-Budget Gaussian-NPE Dry-Run Review

Prepared at: 2026-04-26T01:01:15Z

No PBS submission was run.

## Scope

| Item | Value |
|---|---:|
| Model | empirical GNK |
| Method | Gaussian-NPE |
| Excluded method | flow-NPE |
| n | 500 |
| d | 11 |
| x values | 25, 50 |
| Seeds | 0 through 100 inclusive |
| Output namespace | res/gnk_high_budget/ |

## Budget Numbers

Formula: `N = x * d^2 * n = x * 11^2 * 500`.

| x | N |
|---:|---:|
| 25 | 1,512,500 |
| 50 | 3,025,000 |

No `x > 50` rows are present.

## Row Counts

| Row type | Count |
|---|---:|
| Total grid rows | 202 |
| Runnable PBS array rows | 201 |
| Reused rows | 1 |

The single reused row is `x=50`, `seed=88`, `N=3,025,000`.

## Reused Calibration Row

Source output:

`res/gnk_hpc_calibration/gnk_gaussian_npe_n500_x50_seed88_20260425T065035Z`

Compatibility checks passed for:

| Field | Expected |
|---|---:|
| n | 500 |
| d | 11 |
| x | 50 |
| N | 3,025,000 |
| seed | 88 |
| method | Gaussian-NPE |

Hyperparameter compatibility was checked against:

`res/gnk_high_budget/configs/gnk_gaussian_npe_n500_x50_seed89_20260426T010115Z.yaml`

The following matched: method, simulator, n, d_s, d_theta, d, x, N, GPU request, learning rate, batch size, max epochs, patience, validation split, and hidden dimensions.

Observed completed calibration metadata:

| Metric | Value |
|---|---:|
| Exit status | 0 |
| Scheduler job id | 20344975.aqua |
| Wall time seconds | 23,405.01 |
| Approx wall time | 6h 30m |
| Simulation seconds | 207.65 |
| Training seconds | 23,180.03 |
| Peak RSS KB | 1,775,196 |
| KL theta kNN 2000 | 2.3282495190889403 |
| MMD theta 2000 | 0.21408498287200928 |

## Task 1 U-Space Evaluation

Task 1 post-calibration u-space decomposition exists at:

`res/gnk_hpc_calibration/gnk_gaussian_npe_n500_x50_seed88_20260425T065035Z/evaluation/gnk_hpc_calibration_seed88_u_space_eval_20260426T005408Z.json`

Gate flags:

| Gate | Value |
|---|---:|
| schema_compatible | true |
| finite_sane_decomposition | true |
| passes_operational_evaluation_gate | true |

Reviewed sample-moment decomposition:

| Quantity | Value |
|---|---:|
| K_theta_star | 0.09305122348924681 |
| K_u_star | 0.250866626549176 |
| coord_offset | 0.1578154030599292 |
| Delta_N_u | 2.074183110746632 |
| Delta_N_u_mean_component | 0.46664382486125144 |
| Delta_N_u_covariance_component | 1.6075392858853803 |
| self_consistency_kl_reconstructed_Qhat_N_u | 0.004333260890615903 |

All listed values are finite.

This row is excluded from the PBS array and represented by:

`res/gnk_high_budget/reuse_markers/gnk_high_budget_n500_x50_seed88_reuse.json`

## PBS Plan

PBS script:

`npe_convergence/scripts/pbs_jobs/gnk_high_budget_gaussian_npe_array.sh`

Prepared submit command, not run:

`qsub npe_convergence/scripts/pbs_jobs/gnk_high_budget_gaussian_npe_array.sh`

Resource request:

| PBS request | Value |
|---|---:|
| Array indices | 0-200 |
| Concurrency cap | 20 |
| Walltime | 47:00:00 |
| Memory | 64GB |
| CPUs | 4 |
| GPUs | 0 |

The `47h/64GB/4CPU` request is intentionally conservative relative to the completed seed-88 calibration, which used about 6.5h wall time and about 1.7GB peak RSS.

## Array Index Mapping Checkpoints

| Manifest index | PBS index | Action | x | Seed | N |
|---:|---:|---|---:|---:|---:|
| 0 | 0 | run | 25 | 0 | 1,512,500 |
| 100 | 100 | run | 25 | 100 | 1,512,500 |
| 101 | 101 | run | 50 | 0 | 3,025,000 |
| 188 | 188 | run | 50 | 87 | 3,025,000 |
| 189 | none | reuse | 50 | 88 | 3,025,000 |
| 190 | 189 | run | 50 | 89 | 3,025,000 |
| 201 | 200 | run | 50 | 100 | 3,025,000 |

The wrapper resolves `PBS_ARRAY_INDEX` against rows with `action == "run"`, not raw manifest rows. Its dry-run mode was checked for indices 0, 188, 189, and 200.

## Runtime Guards

`npe_convergence/scripts/run_gnk_high_budget_array_job.py` supports an index-check mode:

`python npe_convergence/scripts/run_gnk_high_budget_array_job.py --manifest res/gnk_high_budget/dry_run_manifest_20260426T010115Z.json --array-index 189 --dry-run`

Runtime overwrite protection:

- refuses to run if the target output directory already exists
- refuses to run if any expected output file already exists, including per-run stdout/stderr logs
- writes Python stdout/stderr to per-run log paths under each output directory

## Human-Readable Artifact Pointers

Full machine-readable artifacts:

- `res/gnk_high_budget/dry_run_manifest_20260426T010115Z.csv`
- `res/gnk_high_budget/dry_run_manifest_20260426T010115Z.json`

Useful spot-check configs:

- `res/gnk_high_budget/configs/gnk_gaussian_npe_n500_x25_seed0_20260426T010115Z.yaml`
- `res/gnk_high_budget/configs/gnk_gaussian_npe_n500_x50_seed89_20260426T010115Z.yaml`

Output path pattern for new runnable rows:

`res/gnk_high_budget/runs/gnk_gaussian_npe_n500_x{x}_seed{seed}_20260426T010115Z/`

The reused `x=50, seed=88` row points to the existing calibration namespace instead of this new output pattern.
