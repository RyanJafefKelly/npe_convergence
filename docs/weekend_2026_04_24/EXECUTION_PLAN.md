# Weekend empirical push: 2026-04-24

## Scientific objective

Support the theory-first JMLR paper by clarifying finite-N NPE approximation under BvM, while keeping MA2-b0 separate as a compatibility-failure example.

## Current priority order

1. n=500 BvM oracle gate. Completed: median K_theta^*(n=500) = 0.099533 nats over 101 seeds; acceptable by the median-based <= 0.1 rule, but borderline.
2. Coordinate reconciliation: theta oracle, u oracle, coordinate offset. Completed on `exp-u-space-kl-decomp`; scientific-code review issues addressed.
3. Gaussian-NPE u-space KL decomposition. Completed on `exp-u-space-kl-decomp`; scientific-code review issues addressed.
4. Log-corrected scaled-budget plots. Completed on `exp-log-corrected-scaling`; uses `notebooks/plots/gnk_u_space_kl_decomp_20260425_per_seed.csv`, excludes `N=n`, and requires complete 101-seed groups for the main theorem-facing scaled-budget panels.
5. HPC calibration for high-budget GNK curve. Completed on `exp-hpc-calibration`: one non-array n=500, x=50, N=3,025,000 Gaussian-NPE calibration job completed with exit status 0 in about 6.5h wall time and about 1.7GB RSS.
6a. Evaluate completed empirical-GNK calibration output. Next required task: consume the n=500, x=50, N=3,025,000, seed-88 output through the reviewed u-space decomposition and check schema/numerical sanity.
6b. If 6a passes, prepare a bounded staged empirical-GNK Gaussian-NPE array dry-run for n=500, x in {25,50}, seeds 0:100, reusing the completed x=50 seed-88 output if format-compatible.
7. gnk_model simulator-control pilot. Retry prepared on `exp-gnk-model-control`; if the currently running PBS job is this retry, keep its worktree isolated and evaluate it separately after completion.
8. Hexadecile cache inventory and aggregation. Do not launch new high-budget hexadecile jobs before inventorying existing NUTS/Gaussian-NPE caches.
9. MA2-b0 cache inventory and compatibility figure sanity check. Keep separate from pooled BvM-rate fits.
10. Stereological only if nearly automatic.

## Hard constraints

- Do not modify `paper.tex` unless explicitly assigned.
- Do not overwrite caches.
- Do not launch any high-budget array until the completed empirical-GNK seed-88 calibration output has passed reviewed u-space decomposition format/sanity checks and Ryan has reviewed the dry-run table.
- Do not launch x > 50 for GNK high-budget empirical Gaussian-NPE in the next staged array.
- Do not launch flow-NPE endpoints or the old broad x in {25,50,100,200,500} grid.
- Do not mutate the worktree used by the running `gnk_model` PBS job; prepare empirical-GNK array work in a separate worktree or clone if needed.
- Do not let agents make paper-framing changes; agents may record neutral evidence in docs.
- Use u-space for exact Gaussian-NPE analytic decomposition.
- Use theta-space oracle KL for the BvM premise.
- Keep MA2-b0 out of the BvM-rate narrative.
- Every generated figure must record script, commit hash, cache paths, and timestamp.

## Coordinate notation

u = (logit(theta / 10) - mu_eta) / sigma_eta.

K_theta^* = KL(P_theta || G_theta^*)
K_u^*     = KL(P_u || G_u^*)

Delta_N,u = KL(G_u^* || Qhat_N,u)
Delta_N,theta = KL(P_theta || Qhat_N,theta) - K_theta^*
Delta_N,theta = Delta_N,u + (K_u^* - K_theta^*)

Interpret Delta_N,u as native-coordinate Gaussian-NPE error, not pure BvM target geometry.

## Decision rules

### n=500 oracle gate

- If K_theta^*(n=500) <= 0.1 nats: n=500 is acceptable for the high-budget curve.
- If 0.1 < K_theta^*(n=500) < 0.3 nats: n=500 is diagnostic only; be cautious.
- If K_theta^*(n=500) >= 0.3 nats: use n=1000 for main high-budget BvM-rate evidence.

### High-budget HPC

- n=500 passed the median-based oracle gate and is selected for calibration/main high-budget planning, with the caveat that acceptance is borderline because the IQR is [0.077337, 0.135198].
- One calibration job has completed: n=500, x=50, N=3,025,000, seed 88, PBS job `20344975.aqua`, exit status 0, about 6.5h wall time, about 1.7GB RSS.
- Treat individual x=50 jobs as operationally feasible under the tested 47h/64GB PBS request.
- Before any staged array, evaluate the calibration output with the reviewed u-space decomposition and review a dry-run table.
- If the calibration output is schema-compatible and all reviewed decomposition quantities are finite/sane, the next approved planning target is bounded empirical-GNK Gaussian-NPE only: n=500, x in {25,50}, seeds 0:100, with x=50 seed=88 reused if compatible.
- Stop short of x > 50, flow-NPE endpoints, and broad-grid submission until the bounded array has been reviewed.

### Completed empirical-GNK calibration evaluation gate

Evaluate:

- n = 500
- x = 50
- N = 3,025,000
- seed = 88

Inputs:

- `res/gnk_hpc_calibration/gnk_gaussian_npe_n500_x50_seed88_20260425T065035Z/gaussian_npe_u_posterior.npz`
- `res/gnk_hpc_calibration/gnk_gaussian_npe_n500_x50_seed88_20260425T065035Z/posterior_samples_10k.npz`
- `res/gnk/nuts_cache_v2_n_obs_500_seed_88.pkl`
- reviewed convention from `scripts/compute_gnk_u_space_kl_decomp.py`

Compute and record:

- `K_theta^*`
- `K_u^*`
- `coord_offset = K_u^* - K_theta^*`
- `Delta_N,u`
- `Delta_N,u_mean`
- `Delta_N,u_cov`
- self-consistency/reconstruction diagnostic where available

Proceed to the bounded dry-run only if:

- required files exist;
- `gaussian_npe_u_posterior.npz` has `mu_u.shape == (4,)` and `cov_u.shape == (4,4)`;
- posterior samples have shape `(10000, 4)` in theta/u/eta where expected;
- covariance is SPD or passes the same numerical checks used in the reviewed decomposition;
- `Delta_N,u`, mean component, and covariance component are finite;
- the self-consistency diagnostic is in the same broad regime as the existing reconstruction check;
- no cache is overwritten.

Stop and inspect if the evaluation path needs ad hoc schema changes, any component is NaN/Inf, the covariance component is negative beyond numerical tolerance, the schema is aggregation-ambiguous, or the result accidentally uses theta-space samples to reconstruct a u-space Gaussian.

Do not require the raw `metrics.json` KL/MMD to be small. Those calibration diagnostics are not the scientific comparison; use the reviewed u-space decomposition.

### Bounded high-budget empirical-GNK dry-run

If the calibration evaluation gate passes, prepare but do not submit:

- d = d_s + d_theta = 7 + 4 = 11
- N = x d^2 n = 60,500 x
- x = 25 gives N = 1,512,500
- x = 50 gives N = 3,025,000
- seeds 0:100
- total new jobs: 201 if x=50 seed=88 is reused

Use a fresh namespace such as `res/gnk_high_budget/`. Do not write into old cached `res/gnk/gaussian_npe_n_obs_*` directories unless a reviewed runner explicitly guarantees no overwrites.

Dry-run rows must include:

```text
n, x, N, seed, output_dir, config_path,
validation_curve_path,
gaussian_npe_u_posterior_path,
posterior_samples_10k_path,
timing_metadata_path,
stdout_log_path,
stderr_log_path,
collision_status,
reuse_existing
```

The dry-run must either show `x=50, seed=88, reuse_existing=true` or exclude that job and record reuse separately. No `collision_status=true` rows should be submitted.

Default first-pass PBS request for the bounded array is `ncpus=1`, `mem=8GB`, `walltime=24:00:00`, unless existing scripts rely on four workers. If so, use `ncpus=4`, `mem=16GB`, `walltime=24:00:00` for the staged array and optimise later. Prefer a concurrency cap around 20-30 jobs if Aquarius supports it.

Hold or delete remaining queued jobs if repeated early failures, non-empty repeated stderr, missing outputs, malformed validation curves, memory above roughly 6GB under an 8GB request, x=50 walltime pressure, unexpected output namespaces, or overwrite attempts appear.

## Branch and worktree discipline

Before starting any task, read `docs/weekend_2026_04_24/AGENT_BRIEF.md`. It links the compact project context in `_brief_for_chatgpt_round4.md` and gives the operational rules for agents.

Use one branch or worktree per task. Do not let two agents edit the same plotting script at the same time.

Use flat task branch names in this repo. Avoid nested names like `exp/<task-name>` because Git refs can conflict when a flat `exp` ref exists or has existed.

Suggested branches:

- `exp-oracle-gate-n500`
- `exp-u-space-kl-decomp`
- `exp-log-corrected-scaling`
- `exp-hpc-calibration`
- `exp-gnk-model-control`
- `exp-hexadecile-aggregation`
- `exp-ma2-compatibility`

Suggested start commands:

```bash
git status
git checkout main
git pull
git checkout -b exp-<task-name>
```

Or use worktrees:

```bash
git worktree add ../npe-exp-<task-name> -b exp-<task-name>
```

Suggested close-out commands:

```bash
git status
git diff --stat
git diff
pytest ...
python scripts/... --dry-run
```

Every task should end with:

- Files changed.
- Commands run.
- Outputs generated.
- Tests passed.
- Known limitations.
- Recommended next step.

## Agent roles

Repo docs in `docs/weekend_2026_04_24/` are the source of truth. Agents should read these files before starting and update them before handing work back.

The compact high-level project context is `_brief_for_chatgpt_round4.md`. Agents should use that as the common background reference rather than older exploratory briefs.

Coding agents execute narrow tasks. They may update `RESULTS.md`, but they should not silently decide paper framing.

Use GPT-5.5 Pro for novel, mathematically delicate, or heavy-reasoning questions. Routine coding and implementation from an agreed plan can be handled by GPT-5.5 high unless something novel or complicated arises.

## Review strategy

Use cross-model review for the dangerous tasks:

| Task | Implementer | Reviewer |
| --- | --- | --- |
| u-space KL decomposition | Codex | Claude Code |
| HPC calibration scripts | Claude Code or Codex | the other |
| gnk_model control | Claude Code or Codex | the other |

For simple plot changes, self-review is acceptable unless the diff is larger than expected.

Generic review prompt:

```text
Review this branch as a scientific-code reviewer for a JMLR-targeted theory paper.

Do not focus on cosmetic style. Focus on:
1. mathematical correctness;
2. coordinate-system consistency;
3. cache safety;
4. reproducibility;
5. whether figure labels support the intended theorem-facing claim;
6. whether the code accidentally changes existing behaviour.

Return:
- blocking issues;
- nonblocking issues;
- tests/commands you recommend before merge;
- whether the branch is safe to merge.
```
