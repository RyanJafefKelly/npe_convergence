# Weekend empirical push: 2026-04-24

## Scientific objective

Support the theory-first JMLR paper by clarifying finite-N NPE approximation under BvM, while keeping MA2-b0 separate as a compatibility-failure example.

## Current priority order

1. n=500 BvM oracle gate.
2. Coordinate reconciliation: theta oracle, u oracle, coordinate offset.
3. Gaussian-NPE u-space KL decomposition.
4. Log-corrected scaled-budget plots.
5. HPC calibration for high-budget GNK curve.
6. gnk_model simulator-control pilot.
7. Hexadecile aggregation.
8. MA2-b0 compatibility figure sanity check.
9. Stereological only if nearly automatic.

## Hard constraints

- Do not modify `paper.tex` unless explicitly assigned.
- Do not overwrite caches.
- Do not launch large HPC arrays before the n=500 oracle gate and one calibration job.
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

- Select n only after the n=500 oracle gate is recorded in `DECISIONS.md`.
- Run one calibration job before submitting any full array.
- Prepare dry-run tables before submission.
- Do not submit the full array until the dry-run table has been reviewed.

## Branch and worktree discipline

Before starting any task, read `docs/weekend_2026_04_24/AGENT_BRIEF.md`. It links the compact project context in `_brief_for_chatgpt_round4.md` and gives the operational rules for agents.

Use one branch or worktree per task. Do not let two agents edit the same plotting script at the same time.

Suggested branches:

- `exp/oracle-gate-n500`
- `exp/u-space-kl-decomp`
- `exp/log-corrected-scaling`
- `exp/hpc-calibration`
- `exp/gnk-model-control`
- `exp/hexadecile-aggregation`
- `exp/ma2-compatibility`

Suggested start commands:

```bash
git status
git checkout main
git pull
git checkout -b exp/<task-name>
```

Or use worktrees:

```bash
git worktree add ../npe-exp-<task-name> -b exp/<task-name>
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
