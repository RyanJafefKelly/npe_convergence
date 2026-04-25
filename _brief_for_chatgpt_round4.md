# Project brief round4: NPE convergence empirical push

This is the current compact project context for agents and external reviewers. It supersedes earlier exploratory briefs for day-to-day task execution. Earlier notes and email drafts remain useful history, but the current source of truth is:

- `docs/weekend_2026_04_24/EXECUTION_PLAN.md`
- `docs/weekend_2026_04_24/RUNS.yaml`
- `docs/weekend_2026_04_24/RESULTS.md`
- `docs/weekend_2026_04_24/DECISIONS.md`

## People and paper

Ryan Kelly is handling the experimental and computational side of a JMLR-targeted, theory-first paper on the statistical accuracy of simulation-based posterior estimation. The co-authors are David T. Frazier, Chris Drovandi, and David J. Warne. David T. Frazier is the theory lead and corresponding author.

The paper is not meant to become a new-methods or software paper. The empirical work should showcase and stress-test the theory, make the theorem-facing claims credible, and be useful to the co-authors when they decide how to frame the final empirical section.

The working title/context is the statistical accuracy of neural posterior and likelihood estimation methods. The empirical focus is neural posterior estimation (NPE) for partial posteriors conditional on summary statistics.

## Theory compass

The experiments should support the paper's theoretical contributions, not wander into unrelated benchmarking.

1. General concentration: NPE targets the partial posterior `Pi(theta | S_n)`. Its statistical accuracy is governed by the posterior contraction rate plus an approximation/amortisation term.
2. Dimension burden for general approximators: for a generic conditional density class, approximation rates can imply infeasible simulation budgets as dimension `d = d_s + d_theta` grows.
3. BvM/Gaussian-family corollary: when the partial posterior satisfies a Bernstein-von Mises result, a Gaussian-family NPE should be sufficient asymptotically. The relevant budget axis is roughly `N / (d^2 n)`, with logarithmic factors still present.
4. Compatibility failure: if the observed summary is incompatible with the assumed model, increasing `N` cannot repair the target. MA2-b0 belongs here, not in the BvM-rate story.

The empirical section should be JMLR-appropriate: clear, theorem-facing, reproducible, sober about limitations, and careful not to overclaim from finite compute.

## Current empirical narrative

Earlier working hypotheses have been overturned. The current GNK story is:

- The GNK partial posterior is essentially Gaussian at the relevant `n`. Moment-matched Gaussian oracle KL is about 0.05 nats at `n=1000`, about 0.005 at `n=5000`, and near estimator noise at larger `n`. This supports the BvM premise.
- The old idea that Gaussian-NPE failed because the GNK posterior was banana-shaped or strongly non-Gaussian is wrong.
- Both flow-NPE and Gaussian-NPE retain a substantial finite-`N` residual above the Gaussian oracle at cached budgets.
- Flow-NPE and Gaussian-NPE track each other closely under BvM. This is not a horse race; tracking is evidence for family sufficiency.
- The residual appears systematic, not a seed-1 artifact, not a known transform-chain bug, not stale NUTS-cache provenance, and not fully closed by a small Gaussian-NPE hyperparameter sweep.
- The most useful framing is: BvM target Gaussianity holds; family sufficiency is visible; `N/(d^2 n)` partially organises the decay; finite-`N` constants/amortisation residuals are practically important.

This is a better paper story than "Gaussian-NPE beats flow". The theory does not require Gaussian-NPE to win by a large margin at every feasible `N`; under BvM, the point is that a restricted Gaussian family can track a flexible family because the target is already Gaussian.

## GNK details to keep straight

GNK is the main BvM-rate example.

- Parameters: `theta = (A, B, g, k)`.
- Prior: uniform on `(0, 10)` for each parameter.
- True values: approximately `(3, 1, 2, 0.5)`.
- Default summaries: octiles, so `d_s = 7`, `d_theta = 4`, `d = 11`.
- Hexadeciles are a summary-resolution stress test, expected `d_s = 15`, `d = 19`.
- Reference posterior: NUTS using the asymptotic multivariate-normal summary likelihood for quantiles.
- The `gnk_density` bug was fixed at commit `af438f5` on March 11, 2026. Current NUTS caches should be checked for post-fix provenance when relevant.

For BvM diagnostics, the source of truth is theta-space:

- `K_theta^* = KL(P_theta || G_theta^*)`
- `P_theta` is the NUTS posterior in theta-space.
- `G_theta^*` is the moment-matched Gaussian in theta-space.

For Gaussian-NPE analytic decomposition, the native training/inference coordinate is u-space:

- `u = (logit(theta / 10) - mu_eta) / sigma_eta`
- `K_u^* = KL(P_u || G_u^*)`
- `Delta_N,u = KL(G_u^* || Qhat_N,u)`
- `coord_offset = K_u^* - K_theta^*`
- `Delta_N,theta = Delta_N,u + coord_offset`

Interpret `Delta_N,u` as native-coordinate Gaussian-NPE error. Do not call it pure BvM residual.

## What has already been ruled out or downgraded

- Non-Gaussian GNK target geometry as the main cause of Gaussian-NPE underperformance.
- Seed-1 as the explanation for the residual.
- A known NUTS-reference bug after the `af438f5` fix.
- A simple transform-chain or Gaussian-NPE sampling bug.
- A simple "Gaussian-NPE cannot represent the target" family-misspecification story.
- A small Gaussian-NPE hyperparameter tweak as a complete fix.

These points do not mean future agents should never verify assumptions. They mean agents should not spend their task budget rediscovering stale hypotheses unless a new diagnostic directly requires it.

## Current weekend objective

The immediate push is to clarify finite-`N` NPE approximation under BvM, while keeping MA2-b0 separate as compatibility failure.

Priority order:

1. n=500 theta-space BvM oracle gate.
2. Coordinate reconciliation: theta oracle, u oracle, coordinate offset.
3. Gaussian-NPE u-space KL decomposition.
4. Log-corrected scaled-budget plots.
5. HPC calibration for high-budget GNK curve. Completed: one non-array n=500, x=50, N=3,025,000 Gaussian-NPE calibration job completed in about 6.5h wall time with about 1.7GB RSS.
6. Post-calibration high-budget diagnostic: evaluate the calibration output with the reviewed u-space decomposition before treating it as a theorem-facing high-budget point or approving any broad array.
7. gnk_model simulator-control pilot.
8. Hexadecile aggregation.
9. MA2-b0 compatibility sanity check.
10. Stereological only if nearly automatic.

The broad high-budget HPC curve remains blocked until the completed calibration output is evaluated with the reviewed u-space decomposition and a staged x/seed grid dry-run is reviewed.

## How to interpret outcomes

For GNK:

- Low `K_theta^*` means the BvM target Gaussianity premise holds.
- Decay of residuals with `N/(d^2 n)` supports the corollary's budget organisation.
- Flow and Gaussian-NPE tracking each other supports family sufficiency under BvM.
- A nonzero residual at feasible budgets should be framed as finite-`N`/amortisation cost, not as failure of BvM target Gaussianity.
- Large constants are a result worth reporting honestly, not something to hide.

For hexadeciles:

- Collapse after using each summary choice's own `d` supports `d^2 n` organisation.
- Failure to collapse may reflect changed summary map, conditioning, or identification, not necessarily a theory failure.

For MA2-b0:

- The role is compatibility failure only.
- It should show that increasing `N` cannot fix an incompatible observed summary.
- Do not include MA2-b0 in pooled BvM-rate fits.

For gnk_model simulator-control:

- The pilot asks whether the finite-`N` residual persists when NPE training pairs are generated from the same asymptotic MVN summary likelihood used by NUTS.
- If the residual shrinks substantially, empirical-GNK simulator/control mismatch may matter more than currently believed.
- If the residual persists, the finite-`N` amortisation explanation strengthens.

## Practical stance for agents

Agents are implementers and reviewers, not owners of the scientific narrative. They should update `RESULTS.md` with evidence and neutral interpretations, but should not silently change the paper framing.

Use GPT-5.5 Pro or a similarly strong reasoning pass for novel mathematical interpretation, delicate KL identities, or theorem-facing claims. Routine implementation from a clear plan can be done by coding agents.

Never modify `paper.tex` unless explicitly assigned. Do not overwrite caches. Every generated figure or table should record command/script, commit hash, cache/input paths, output paths, and timestamp.
