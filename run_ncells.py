"""
run_ncells.py
-------------
Sensitivity to the snapshot size n_cells. Sweeps the number of cells per snapshot
and reports, per training simulator, the k_on 90% coverage against exact CME data,
plus the correlation between the empirical and analytic Fano factor.

Expected reading:
  * CME (control) stays calibrated at every snapshot size.
  * the misspecified CLE gets WORSE as n_cells grows: more data sharpens an already
    wrong posterior, the signature of genuine misspecification rather than noise.
  * corr(empirical, analytic Fano) rises with n_cells: the observable diagnostic
    sharpens as the snapshot grows.

Usage:
    python run_ncells.py --n_sims 8000 --n_cells 50 200 500 2000 --sims cme cle
Cost note: each (simulator, n_cells) regenerates its training set; the CLE at large
n_cells is the bottleneck. Drop --n_sims for a quick look.
"""

import argparse
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

import run_experiment as R
from telegraph import exact_fano


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_sims", type=int, default=8000)
    p.add_argument("--n_cells", type=int, nargs="+", default=[50, 200, 500, 2000])
    p.add_argument("--n_test", type=int, default=400)
    p.add_argument("--n_post", type=int, default=1000)
    p.add_argument("--sims", type=str, nargs="+", default=["cme", "cle"])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cov = {s: [] for s in args.sims}
    corr = []
    for nc in args.n_cells:
        for sim in args.sims:
            torch.manual_seed(args.seed)
            post = R.train(sim, args.n_sims, nc, args.seed)
            tt, bias, ranks, n_post, emp = R.evaluate(
                post, args.n_test, nc, seed=999, n_post=args.n_post)
            cov[sim].append(R.coverage_90(ranks, n_post)[0])  # k_on
            if sim == args.sims[0]:
                ana = np.array([exact_fano(*r) for r in R.theta_to_rates(tt)])
                corr.append(np.corrcoef(np.log(ana),
                                        np.log(np.clip(emp, 1e-3, None)))[0, 1])
        print(f"[n_cells={nc:5d}] " +
              "  ".join(f"{s}={cov[s][-1]:.2f}" for s in args.sims) +
              f"  | Fano corr={corr[-1]:.3f}")

    print("\n=== k_on 90% coverage vs n_cells ===")
    print("  n_cells: " + "  ".join(f"{nc:6d}" for nc in args.n_cells))
    for s in args.sims:
        print(f"  {s.upper():4s}:  " + "  ".join(f"{v:6.2f}" for v in cov[s]))
    print("  Fanocorr " + "  ".join(f"{v:6.3f}" for v in corr))


if __name__ == "__main__":
    main()
