# Decisions

Record only actual decisions. Keep each entry short enough that future readers can recover why the empirical path was chosen.

## Template

```markdown
## YYYY-MM-DD: decision title

Decision: ...
Reason: ...
Implication: ...
Evidence:
- ...
```

## Pending decisions

- Which scaled-budget grid to use for high-budget GNK Gaussian-NPE.
- Whether flow-NPE endpoints are worth running in the high-budget array.
- Whether gnk_model simulator-control pilot results require a paper-framing change.

## 2026-04-25: GNK n=500 oracle gate

Decision: Treat n=500 as acceptable for the main high-budget BvM-rate curve under the median-based gate.
Reason: The cached theta-space Gaussian oracle KL has median 0.099533 nats over 101 seeds, just below the <= 0.1 nats acceptability threshold.
Implication: HPC calibration may use n=500, while noting that the acceptance is borderline because the IQR is [0.077337, 0.135198]. Full-array submission remains blocked pending one calibration job and explicit dry-run review.
Evidence:
- `notebooks/plots/gnk_n500_oracle_gate_20260425_summary.json`
- `notebooks/plots/gnk_n500_oracle_gate_20260425_per_seed.csv`
