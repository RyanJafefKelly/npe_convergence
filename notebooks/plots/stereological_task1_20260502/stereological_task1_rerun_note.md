# Stereological Task 1 Rerun Note

- Standard coverage/bias cache status: flow_npe 11/16 groups have at least 100 complete seeds; gaussian_npe 11/16 groups have at least 100 complete seeds.
- Standard flow posterior-sample status: 11/16 groups have at least 100 saved posterior sample files.
- Standard Gaussian posterior-sample status: 0/16 groups have any saved posterior sample file.
- ABC-SMC n=1000 benchmark sample file present: yes.

Recommendation: a targeted stereological rerun or recovery is needed if the paper-style standard grid must be complete at 100 seeds for every `(n, N)` cell. Do not launch broad stereological scripts from this task output alone; prepare a collision-checked manifest containing only the shortfall groups listed below.

Caveat: Gaussian-NPE cache directories in this snapshot contain coverage and bias arrays but no `posterior_samples.pkl` files, so a Gaussian posterior overlay cannot be regenerated from saved posterior draws. A fresh Gaussian run or a cache-safe posterior export would only be needed if a Gaussian posterior overlay is required for the paper-facing figure.

Gaussian overlay status: the n=1000 Gaussian coverage/bias caches exist for `N in {n, n log(n), n^(3/2)}`, but there are no saved Gaussian posterior sample files for the overlay inputs.

Coverage/bias standard-grid shortfalls:
- flow_npe n=500 N=n^2 N=250000: 94
- flow_npe n=1000 N=n^2 N=1000000: 63
- flow_npe n=5000 N=n log(n) N=42585: 98
- flow_npe n=5000 N=n^(3/2) N=353553: 84
- flow_npe n=5000 N=n^2 N=25000000: 3
- gaussian_npe n=500 N=n^2 N=250000: 95
- gaussian_npe n=1000 N=n^2 N=1000000: 70
- gaussian_npe n=5000 N=n log(n) N=42585: 99
- gaussian_npe n=5000 N=n^(3/2) N=353553: 85
- gaussian_npe n=5000 N=n^2 N=25000000: 0

Flow posterior standard-grid shortfalls:
- flow_npe n=500 N=n^2 N=250000: 94
- flow_npe n=1000 N=n^2 N=1000000: 63
- flow_npe n=5000 N=n log(n) N=42585: 98
- flow_npe n=5000 N=n^(3/2) N=353553: 84
- flow_npe n=5000 N=n^2 N=25000000: 3

Gaussian posterior standard-grid shortfalls:
- gaussian_npe n=100 N=n N=100: 0
- gaussian_npe n=100 N=n log(n) N=460: 0
- gaussian_npe n=100 N=n^(3/2) N=1000: 0
- gaussian_npe n=100 N=n^2 N=10000: 0
- gaussian_npe n=500 N=n N=500: 0
- gaussian_npe n=500 N=n log(n) N=3107: 0
- gaussian_npe n=500 N=n^(3/2) N=11180: 0
- gaussian_npe n=500 N=n^2 N=250000: 0
- gaussian_npe n=1000 N=n N=1000: 0
- gaussian_npe n=1000 N=n log(n) N=6907: 0
- gaussian_npe n=1000 N=n^(3/2) N=31622: 0
- gaussian_npe n=1000 N=n^2 N=1000000: 0
- gaussian_npe n=5000 N=n N=5000: 0
- gaussian_npe n=5000 N=n log(n) N=42585: 0
- gaussian_npe n=5000 N=n^(3/2) N=353553: 0
- gaussian_npe n=5000 N=n^2 N=25000000: 0

Nonstandard/incomplete diagnostic groups:
- flow_npe n=5000 N=2500000: complete coverage/bias seeds=26, posterior seeds=26
