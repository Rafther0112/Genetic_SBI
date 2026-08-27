"""
run_reference.py
----------------
P0-5 driver: generate gold-standard reference posteriors for the shared exact-CME
test set, to be scored against NPE posteriors with C2ST / MMD (see metrics.py, E4).

For each test observation i we run emcee on the exact Poisson-Beta likelihood and
save n_samples posterior draws (log10 space). Observations are independent, so the
loop is embarrassingly parallel; pass --workers to spread it over cores.

Saves cache/references_{seed}_{n_ref}.npz with:
    ref_samples : (n_ref, n_samples, 3)   reference posterior draws (log10)
    theta_log   : (n_ref, 3)              true params
    emp_fano    : (n_ref,)                empirical Fano (for binning)

Needs numpy + scipy + emcee (no torch / sbi). CPU only.

Usage:
    python run_reference.py --seed 0 --n_test 400 --n_ref 400 --n_samples 2000 --workers 8
"""

import argparse
import os
from functools import partial
from multiprocessing import Pool

import numpy as np

import testset
import reference_posterior as RP


def _one(i, counts, n_samples):
    return RP.sample_reference(counts[i], n_samples=n_samples, seed=i)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_test", type=int, default=400)
    p.add_argument("--n_cells", type=int, default=500)
    p.add_argument("--n_ref", type=int, default=400)
    p.add_argument("--n_samples", type=int, default=2000)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--outdir", type=str, default="cache")
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    ts = testset.build_or_load(args.seed, args.n_test, args.n_cells)
    counts = ts["counts"]
    n_ref = min(args.n_ref, len(counts))

    fn = partial(_one, counts=counts, n_samples=args.n_samples)
    if args.workers > 1:
        with Pool(args.workers) as pool:
            refs = pool.map(fn, range(n_ref))
    else:
        refs = [fn(i) for i in range(n_ref)]
    ref_samples = np.stack(refs)

    path = os.path.join(args.outdir, f"references_{args.seed}_{n_ref}.npz")
    np.savez(path, ref_samples=ref_samples,
             theta_log=ts["theta_log"][:n_ref], emp_fano=ts["emp_fano"][:n_ref])
    print(f"[written] {path}   shape {ref_samples.shape}")


if __name__ == "__main__":
    main()
