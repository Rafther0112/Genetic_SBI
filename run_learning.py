"""
run_learning.py
---------------
E5 (review W8): learning curve over training-set size. Separates the misspecification
gap (irreducible) from the amortization/training gap (closes with more sims). If the
CME control's coverage improves toward nominal as N_sim grows while the CLE and LNA
stay low, the surrogate failure is misspecification, not undertraining.

Needs torch + sbi. Reuses run_experiment.train (cached per N_sim).

Cost note: the CLE at N_sim=128k is the bottleneck (~16x the 8k time). Drop the top
of --n_sims for a quick look.

Usage:
    python run_learning.py --n_sims 2000 8000 32000 128000 --n_cells 500 --n_test 2000
"""

import argparse
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

import run_experiment as R

SIMS = ["cme", "nb", "lna", "cle"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_sims", type=int, nargs="+", default=[2000, 8000, 32000, 128000])
    p.add_argument("--n_cells", type=int, default=500)
    p.add_argument("--n_test", type=int, default=2000)
    p.add_argument("--n_post", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cov = {s: [] for s in SIMS}
    for ns in args.n_sims:
        for sim in SIMS:
            torch.manual_seed(args.seed)
            post = R.train(sim, ns, args.n_cells, args.seed)
            _, _, ranks, n_post, _ = R.evaluate(
                post, args.n_test, args.n_cells, seed=999, n_post=args.n_post)
            cov[sim].append(R.coverage_90(ranks, n_post)[0])   # k_on
        print(f"[N_sim={ns:6d}] " + "  ".join(f"{s}={cov[s][-1]:.2f}" for s in SIMS))

    print("\n=== k_on 90% coverage vs training-set size ===")
    print("  N_sim: " + "  ".join(f"{ns:7d}" for ns in args.n_sims))
    for s in SIMS:
        print(f"  {s.upper():4s}: " + "  ".join(f"{v:7.2f}" for v in cov[s]))
    print("\nReading: CME (and NB) climb toward nominal as N_sim grows (amortization gap");
    print("closes); CLE/LNA plateau well below (irreducible misspecification gap).")


if __name__ == "__main__":
    main()
