# GNK Task 2 Octile HPC Rerun Note

- Standard paper-grid status: flow_npe 15/16 groups and gaussian_npe 16/16 groups have 101 seeds with `kl.txt`, `estimated_coverage.npy`, and `biases.npy`.
- Conclusion: no broad GNK octile HPC rerun is needed for the Task 2 co-author/debug aggregation. A targeted flow-NPE recovery/rerun would be needed only if the paper-style flow cell with a shortfall must be refreshed at 101 complete seeds.
- Deferred likely next HPC-prep item: include a targeted manifest/dry-run candidate for only `flow_npe n=5000 N=25000000 seeds=0-35` if Ryan proceeds tomorrow. Keep this separate from any broad GNK octile launch.
- Do not run `qsub` from this task output.
- `K_theta^*` is reported separately as the theta-space BvM target-Gaussianity diagnostic. `Delta_N,u` is reported separately as native Gaussian-NPE u-space approximation error.
- The Gaussian-NPE rows are framed as BvM/Gaussian-family diagnostics, not as a flow-vs-Gaussian horse race.

Standard paper-grid shortfalls:
- flow_npe n=5000 N=25000000: seed_count=101, kl=65, coverage=65, bias=65, missing artifact seeds=0-35

Raw theta-space KL caveat:
- The n=100 groups have 101 `kl.txt` files per method but include non-finite KL values. The comparable raw theta-KL tables use finite paired values only: 72 paired seeds for each n=100 budget.

Bias caveat:
- Flow seed `41` has non-finite replicate-level bias diagnostics in `n=1000, N=1000000` and `n=5000, N=353553`. If these bias summaries become paper-facing, prefer finite-value filtering or targeted evaluation repair for seed `41`.

Legacy diagnostic excluded from complete summaries:
- gaussian_npe n=1000 N=31623: seed_count=1, complete_kl_coverage_bias_seed_count=1

Nonstandard/high-n groups with cache gaps, outside the Task 2 standard paper-grid refresh:
- flow_npe n=10000 N=100000000: seed_count=50, kl=0, coverage=0, bias=0
- flow_npe n=20000 N=2828427: seed_count=101, kl=47, coverage=47, bias=47
- gaussian_npe n=10000 N=100000000: seed_count=101, kl=0, coverage=0, bias=0
