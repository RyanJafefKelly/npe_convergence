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
- Do not launch large HPC arrays before the n=500 oracle gate and one calibration job.
- Keep generated outputs in clearly named paths and record them in `RUNS.yaml` and `RESULTS.md`.
- Use exact cache paths in reports whenever possible.

## Scientific guardrails

- Theta-space oracle KL supports the BvM target Gaussianity premise.
- Gaussian-NPE is exactly Gaussian in u-space, where `u = (logit(theta / 10) - mu_eta) / sigma_eta`.
- u-space decomposition is native-coordinate Gaussian-NPE error, not pure BvM residual.
- MA2-b0 is compatibility failure only, not part of BvM-rate evidence.
- Flow-NPE vs Gaussian-NPE is not a horse race here. Under BvM, tracking is evidence for family sufficiency.
- The empirical point is to support and sharpen the theory-first JMLR paper, not to introduce unrelated methodology.

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
