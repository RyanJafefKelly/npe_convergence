# Discussion notes (meeting 2026-05-18)

Scratch notes and asides, separate from the clean results in
`empirical_results_summary.pdf`. Not for the coauthor handout unless you decide
to raise them.

## Asides to consider raising

- **Sequential vs amortised.** Results look better with sequential NPE than
  with the amortised NPE used throughout the paper. Worth deciding whether this
  belongs in this paper or stays separate.

- **The amortisation gap.** The residual gap between the NPE posterior and the
  reference, at feasible budgets, looks more like finite-simulation /
  amortisation error than a Gaussian-family limitation. This connects to the
  preconditioned-NPE work. Open question: bring that angle into this paper, or
  keep it for the preconditioned-NPE paper.

## Things to investigate later (not meeting-blocking)

- **ABC-SMC for stereological.** The single local ABC-SMC run converged poorly
  on `sigma` and `xi` (sigma posterior near 0.4 vs true 2.0). The notebook
  overlays ABC-SMC for `lambda` only, matching the paper. A cleaner or larger
  ABC-SMC run would let the overlay cover all three parameters. Ryan mentioned
  being open to a bigger one-shot SMC-ABC run as a reference.

- **g-and-k failed fits.** Seed 41 makes flow-NPE fail outright (near-uniform
  posterior) at n=1000 N=n^2 and both n=5000 budgets. A width check also
  flagged a fraction of the n=1000, N=n^(3/2) fits as wide for both methods.
  Worth a closer look at fit reliability at the marginal budgets.

## Open framing questions

- Is the GNK Gaussian-oracle / scaled-budget diagnostic in or out of the paper?
- Does the n=5000, N=n^2 cell need strict completion, or report-with-caveat?
