import numpy as np
import math


def run_experiment(experiment_fn, seed: int = 0, **kwargs):
    n_obs = [100, 500, 1000, 5000, 10_000]

    n_sims = [lambda n: n,
              lambda n: int(n * math.log(n)),
              lambda n: int(n ** (3/2)),
              lambda n: n ** 2]

    kl_mat = np.zeros((len(n_obs), len(n_sims)))
    mmd_mat = np.zeros((len(n_obs), len(n_sims)))

    for jj, f in enumerate(n_sims):
        for ii, n in enumerate(n_obs):
            try:
                kl, mmd = experiment_fn(seed, n, f(n), **kwargs)
            except ValueError as e:
                kl = None
                mmd = None
                print(f"Error: {e}")
            print(f"n_obs: {n}, n_sims: {f(n)}, kl: {kl}, mmd: {mmd}")
            kl_mat[ii, jj] = kl
            mmd_mat[ii, jj] = mmd

    return None


if __name__ == "__main__":
    run_experiment(None)
