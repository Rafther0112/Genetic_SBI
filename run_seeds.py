"""
run_seeds.py
------------
Robustness check: repeat the three-way experiment across random seeds and report
mean +/- std of the per-parameter bias and 90% CI coverage. Each seed resamples the
training data, re-initializes the network, and draws a fresh test set, so the spread
reflects genuine run-to-run variability.

Usage:
    python run_seeds.py --n_sims 8000 --n_cells 500 --n_test 400 --seeds 0 1 2 3 4

Cost note: each seed regenerates the CLE training set (the bottleneck, ~6-8 min at
8000 sims) unless it is already cached. For a quick check, drop --n_sims to ~3000.
"""

import argparse
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

import run_experiment as R

PARAMS = ["k_on", "k_off", "k_syn"]
SIMS = ["cme", "cle", "lna", "nb"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_sims", type=int, default=8000)
    p.add_argument("--n_cells", type=int, default=500)
    p.add_argument("--n_test", type=int, default=400)
    p.add_argument("--n_post", type=int, default=1000)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = p.parse_args()

    bias = {s: [] for s in SIMS}     # each entry: (3,) per seed
    cov = {s: [] for s in SIMS}

    for seed in args.seeds:
        for sim in SIMS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            post = R.train(sim, args.n_sims, args.n_cells, seed)
            _, b, ranks, n_post, _ = R.evaluate(
                post, args.n_test, args.n_cells, seed=1000 + seed, n_post=args.n_post)
            bias[sim].append(np.abs(b).mean(0))
            cov[sim].append(R.coverage_90(ranks, n_post))
        c = cov  # shorthand
        print(f"[seed {seed}] k_on coverage  " +
              "  ".join(f"{s}={c[s][-1][0]:.2f}" for s in SIMS))

    print(f"\n=== robustness over {len(args.seeds)} seeds (mean +/- std) ===")
    for sim in SIMS:
        B = np.stack(bias[sim]); C = np.stack(cov[sim])
        print(f"\n{sim.upper()}")
        print("  bias    " + "  ".join(
            f"{p}={B[:,i].mean():.2f}+/-{B[:,i].std():.2f}" for i, p in enumerate(PARAMS)))
        print("  cov90   " + "  ".join(
            f"{p}={C[:,i].mean():.2f}+/-{C[:,i].std():.2f}" for i, p in enumerate(PARAMS)))


if __name__ == "__main__":
    main()