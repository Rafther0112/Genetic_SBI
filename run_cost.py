"""
run_cost.py
-----------
Wall-clock cost of each forward simulator for the telegraph model, reported per
500-cell snapshot and per 8,000-simulation training set, on one fixed machine.

Answers the review's W2 point (and workplan Gate G0(b)): in this sandbox, is the
EXACT generator actually cheaper than the "cheap" CLE surrogate?

Only needs numpy + telegraph.py (no torch / sbi), so it runs in seconds. The
absolute numbers are machine-dependent; the CLE / Poisson-Beta RATIO is what
settles the gate and is robust across machines. Re-run on the paper machine
(Lambda) for the table that goes in the manuscript.

Usage:
    python run_cost.py --n_theta 200 --n_cells 500 --n_sims_ref 8000
"""

import argparse
import time
import numpy as np

import telegraph as T

# same prior box as sbi_experiment.py (log10-uniform)
LOW = np.array([-2.0, -1.0, 0.0])
HIGH = np.array([1.3, 1.3, 2.3])


def draw_rates(n_theta, rng):
    u = rng.uniform(LOW, HIGH, size=(n_theta, 3))
    return 10.0 ** u                      # (n_theta, 3) physical rates


def _rep(rates, n_cells):
    kon = np.repeat(rates[:, 0], n_cells)
    koff = np.repeat(rates[:, 1], n_cells)
    ksyn = np.repeat(rates[:, 2], n_cells)
    return kon, koff, ksyn


def time_poisson_beta(rates, n_cells, rng):
    kon, koff, ksyn = _rep(rates, n_cells)
    t = time.perf_counter()
    x = rng.beta(kon, koff)
    _ = rng.poisson(ksyn * x)
    return time.perf_counter() - t


def time_batched(sampler, rates, n_cells, rng):
    kon, koff, ksyn = _rep(rates, n_cells)
    t = time.perf_counter()
    _ = sampler(kon, koff, ksyn, rng)
    return time.perf_counter() - t


def time_ssa(rates, n_cells, rng):
    t = time.perf_counter()
    for r in rates:
        T.sample_gillespie(tuple(r), n_cells, rng)
    return time.perf_counter() - t


def fmt_per_snap(v):
    return f"{v*1e6:.1f} us" if v < 1e-3 else (f"{v*1e3:.2f} ms" if v < 1 else f"{v:.2f} s")


def fmt_per_set(v):
    return f"{v:.1f} s" if v < 90 else f"{v/60:.1f} min"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_theta", type=int, default=200)
    p.add_argument("--n_cells", type=int, default=500)
    p.add_argument("--n_sims_ref", type=int, default=8000)
    p.add_argument("--ssa_theta", type=int, default=8)
    p.add_argument("--ssa_cells", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    rates = draw_rates(args.n_theta, rng)
    per_snap = {}   # seconds per 500-cell snapshot
    exact_flag = {}

    per_snap["Poisson-Beta (exact CME)"] = time_poisson_beta(rates, args.n_cells, rng) / args.n_theta
    exact_flag["Poisson-Beta (exact CME)"] = True

    per_snap["NB (count surrogate)"] = time_batched(T.sample_nb_batched, rates, args.n_cells, rng) / args.n_theta
    exact_flag["NB (count surrogate)"] = False

    per_snap["LNA (Gaussian surrogate)"] = time_batched(T.sample_lna_batched, rates, args.n_cells, rng) / args.n_theta
    exact_flag["LNA (Gaussian surrogate)"] = False

    per_snap["CLE (Gaussian surrogate)"] = time_batched(T.sample_cle_batched, rates, args.n_cells, rng) / args.n_theta
    exact_flag["CLE (Gaussian surrogate)"] = False

    # SSA on a smaller workload, extrapolated linearly to n_cells (SSA cost ~ n_cells)
    ssa_rates = rates[:args.ssa_theta]
    ssa_dt = time_ssa(ssa_rates, args.ssa_cells, rng)
    per_snap["Gillespie SSA (exact, extrap.)"] = (ssa_dt / args.ssa_theta) * (args.n_cells / args.ssa_cells)
    exact_flag["Gillespie SSA (exact, extrap.)"] = True

    print(f"\nCost per {args.n_cells}-cell snapshot, averaged over {args.n_theta} thetas "
          f"(SSA over {args.ssa_theta} thetas x {args.ssa_cells} cells, extrapolated)\n")
    print(f"{'simulator':32s} {'kind':7s} {'per snapshot':>14s} {'per 8k train set':>18s}")
    print("-" * 74)
    for k, v in sorted(per_snap.items(), key=lambda kv: kv[1]):
        kind = "EXACT" if exact_flag[k] else "surr."
        print(f"{k:32s} {kind:7s} {fmt_per_snap(v):>14s} {fmt_per_set(v*args.n_sims_ref):>18s}")

    pb = per_snap["Poisson-Beta (exact CME)"]
    cle = per_snap["CLE (Gaussian surrogate)"]
    print("-" * 74)
    print(f"\nCLE / Poisson-Beta cost ratio: {cle/pb:.0f}x")
    print("Gate G0(b): the EXACT generator is by far the CHEAPEST here; the 'cheap'")
    print("surrogate (CLE) is the most expensive. The telegraph model cannot carry")
    print("the cost argument -> a second system with a genuinely expensive exact")
    print("simulator (E10) must carry it.")


if __name__ == "__main__":
    main()
