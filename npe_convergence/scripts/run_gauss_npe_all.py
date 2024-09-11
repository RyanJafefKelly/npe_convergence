import math

import numpy as np
from run_gauss_mean_npe import run_gauss_npe  # type: ignore


def run_gauss_npe_all():
    n_obs = [100, 500, 1000, 5000]

    n_sims = [lambda n: n,
              lambda n: int(n * math.log(n)),
              lambda n: int(n ** (3/2)),
              lambda n: n ** 2]

    kl_mat = np.zeros((len(n_obs), len(n_sims)))

    for ii, n in enumerate(n_obs):
        for jj, f in enumerate(n_sims):
            try:
                kl = run_gauss_npe(n, f(n))
            except ValueError as e:
                kl = None
                print(f"Error: {e}")
            print(f"n_obs: {n}, n_sims: {f(n)}, kl: {kl}")
            kl_mat[ii, jj] = kl

    return None


if __name__ == "__main__":
    run_gauss_npe_all()
