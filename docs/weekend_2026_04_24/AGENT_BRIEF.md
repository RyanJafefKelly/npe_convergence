# Agent Brief

This is the entry point for coding agents working on the weekend empirical push.

## Required reading before any task

1. `docs/weekend_2026_04_24/EXECUTION_PLAN.md`
2. `docs/weekend_2026_04_24/RUNS.yaml`
3. `docs/weekend_2026_04_24/RESULTS.md`
4. `docs/weekend_2026_04_24/DECISIONS.md`
5. `docs/weekend_2026_04_24/WORKTREE_BASELINE.md`
6. [`_brief_for_chatgpt_round4.md`](../../_brief_for_chatgpt_round4.md)

Use `_brief_for_chatgpt_round4.md` as the compact high-level project context. Do not require agents to read older brief files unless the user explicitly asks.

## Operating rules

- Complete only the assigned task.
- Start by checking `git status`.
- Treat paths listed in `WORKTREE_BASELINE.md` as pre-existing local state unless your task explicitly touches them.
- Use a task branch or worktree unless the user says otherwise.
- Use flat branch names such as `exp-oracle-gate-n500`. Avoid nested `exp/...` names in this repo because Git refs may conflict with existing flat refs.
- Do not modify `paper.tex` unless explicitly assigned.
- Do not overwrite caches.
- Do not launch high-budget arrays before the completed seed-88 empirical-GNK calibration passes reviewed u-space decomposition format/sanity checks and Ryan reviews the dry-run table.
- Do not launch x > 50, flow-NPE endpoints, or the old broad x in {25,50,100,200,500} grid.
- Do not mutate the worktree used by a running `gnk_model` PBS job; use a separate worktree or clone for empirical-GNK high-budget preparation.
- Do not change paper framing; agents may update docs with neutral evidence and operational status.
- Keep generated outputs in clearly named paths and record them in `RUNS.yaml` and `RESULTS.md`.
- Use exact cache paths in reports whenever possible.

## Scientific guardrails

- Theta-space oracle KL supports the BvM target Gaussianity premise.
- Gaussian-NPE is exactly Gaussian in u-space, where `u = (logit(theta / 10) - mu_eta) / sigma_eta`.
- u-space decomposition is native-coordinate Gaussian-NPE error, not pure BvM residual.
- MA2-b0 is compatibility failure only, not part of BvM-rate evidence.
- Flow-NPE vs Gaussian-NPE is not a horse race here. Under BvM, tracking is evidence for family sufficiency.
- The completed GNK HPC calibration is operational feasibility evidence only until evaluated through the reviewed u-space decomposition.
- The next empirical-GNK gate is operational correctness: schema compatibility, finite/sane u-space decomposition outputs, and no cache overwrite. The high-budget Delta value does not need to be small before preparing the bounded array.
- The approved next planning target, after that gate passes, is bounded empirical-GNK Gaussian-NPE only: n=500, x in {25,50}, seeds 0:100, reusing x=50 seed=88 if compatible.
- The empirical point is to support and sharpen the theory-first JMLR paper, not to introduce unrelated methodology.

## Current task queue

1. Evaluate the completed empirical-GNK n=500, x=50, N=3,025,000, seed-88 calibration output through the reviewed u-space decomposition.
2. If that passes, prepare a high-budget array dry-run only: n=500, x in {25,50}, seeds 0:100, fresh `res/gnk_high_budget/` namespace, collision checks, and explicit reuse/exclusion of x=50 seed=88.
3. Submit and monitor the array only after Ryan explicitly approves the dry-run and gives the final `qsub` instruction.
4. Aggregate completed bounded-array outputs through the same reviewed decomposition; require complete 101-seed groups for theorem-facing medians and label incomplete groups diagnostic.
5. Evaluate the running `gnk_model` control separately after it finishes, using its `u_space_decomposition.json`; do not merge interpretation into empirical-GNK until both paths use the same decomposition convention.
6. Defer hexadecile and MA2-b0 work to cache-inventory tasks.

## Standard task close-out

End every task with:

- Files changed.
- Commands run.
- Outputs generated.
- Tests/checks passed.
- Known limitations.
- Recommended next step.

## Standard prompt prefix

```text
You are working on npe_convergence. First read docs/weekend_2026_04_24/AGENT_BRIEF.md, then EXECUTION_PLAN.md, RUNS.yaml, RESULTS.md, DECISIONS.md, and _brief_for_chatgpt_round4.md. Complete only the assigned task. Do not modify paper.tex. Do not overwrite caches. End with files changed, commands run, outputs generated, tests/checks passed, limitations, and recommended next step.
```
