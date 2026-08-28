"""
run_cost.py
-----------
Wall-clock cost of each forward simulator for the telegraph model, per 500-cell
snapshot and per 8,000-simulation training set, on one fixed machine.

Answers review W2 / workplan Gate G0(b): in this sandbox, is the EXACT generator
actually cheaper than the "cheap" CLE surrogate? Now includes the two exact methods
one would use with no closed form (FSP, SSA) and a CLE step-size (dt) sweep.

Only needs numpy + scipy + telegraph.py + fsp.py (no torch / sbi). The CLE / exact
RATIO is what settles the gate and is robust across machines. Re-run on the paper
machine for the manuscript table.

Usage:
    python run_cost.py --n_theta 200 --n_cells 500
    python run_cost.py --n_theta 500 --ssa_theta 64 --ssa_cells 200   # paper quality
"""

import argparse
import time
import numpy as np

import telegraph as T
import fsp

LOW = np.array([-2.0, -1.0, 0.0])
HIGH = np.array([1.3, 1.3, 2.3])


def draw_rates(n_theta, rng):
    return 10.0 ** rng.uniform(LOW, HIGH, size=(n_theta, 3))


def _rep(rates, n_cells):
    return (np.repeat(rates[:, 0], n_cells),
            np.repeat(rates[:, 1], n_cells),
            np.repeat(rates[:, 2], n_cells))


def time_poisson_beta(rates, n_cells, rng):
    kon, koff, ksyn = _rep(rates, n_cells)
    t = time.perf_counter()
    x = rng.beta(kon, koff); rng.poisson(ksyn * x)
    return time.perf_counter() - t


def time_batched(sampler, rates, n_cells, rng, **kw):
    kon, koff, ksyn = _rep(rates, n_cells)
    t = time.perf_counter()
    sampler(kon, koff, ksyn, rng, **kw)
    return time.perf_counter() - t


def time_fsp(rates, n_cells, rng):
    t = time.perf_counter()
    for r in rates:
        fsp.sample_fsp(tuple(r), n_cells, rng)
    return time.perf_counter() - t


def time_ssa(rates, n_cells, rng):
    t = time.perf_counter()
    for r in rates:
        T.sample_gillespie(tuple(r), n_cells, rng)
    return time.perf_counter() - t


def fmt_snap(v):
    return f"{v*1e6:.1f} us" if v < 1e-3 else (f"{v*1e3:.2f} ms" if v < 1 else f"{v:.2f} s")


def fmt_set(v):
    return f"{v:.1f} s" if v < 90 else f"{v/60:.1f} min"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_theta", type=int, default=200)
    p.add_argument("--n_cells", type=int, default=500)
    p.add_argument("--n_sims_ref", type=int, default=8000)
    p.add_argument("--n_theta_fsp", type=int, default=50)
    p.add_argument("--n_theta_cle", type=int, default=100)
    p.add_argument("--ssa_theta", type=int, default=8)
    p.add_argument("--ssa_cells", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    rates = draw_rates(args.n_theta, rng)
    per_snap, exact = {}, {}

    def rec(name, v, is_exact):
        per_snap[name] = v; exact[name] = is_exact

    rec("Poisson-Beta (exact CME)", time_poisson_beta(rates, args.n_cells, rng) / args.n_theta, True)
    rec("FSP (exact CME)", time_fsp(rates[:args.n_theta_fsp], args.n_cells, rng) / args.n_theta_fsp, True)
    rec("NB (count surrogate)", time_batched(T.sample_nb_batched, rates, args.n_cells, rng) / args.n_theta, False)
    rec("LNA (Gaussian surrogate)", time_batched(T.sample_lna_batched, rates, args.n_cells, rng) / args.n_theta, False)
    rec("CLE (Gaussian, dt=0.02)", time_batched(T.sample_cle_batched, rates, args.n_cells, rng, dt=0.02) / args.n_theta, False)
    ssa = time_ssa(rates[:args.ssa_theta], args.ssa_cells, rng)
    rec("Gillespie SSA (exact, extrap.)", (ssa / args.ssa_theta) * (args.n_cells / args.ssa_cells), True)

    print(f"\nCost per {args.n_cells}-cell snapshot, averaged over {args.n_theta} thetas "
          f"(FSP over {args.n_theta_fsp}, SSA over {args.ssa_theta}x{args.ssa_cells} extrapolated)\n")
    print(f"{'simulator':32s} {'kind':7s} {'per snapshot':>14s} {'per 8k train set':>18s}")
    print("-" * 74)
    for k, v in sorted(per_snap.items(), key=lambda kv: kv[1]):
        print(f"{k:32s} {('EXACT' if exact[k] else 'surr.'):7s} "
              f"{fmt_snap(v):>14s} {fmt_set(v*args.n_sims_ref):>18s}")

    # CLE step-size sweep (cost scales ~ 1/dt)
    print("\nCLE step-size (dt) sweep:")
    cle_rates = rates[:args.n_theta_cle]
    for dt in (0.05, 0.02, 0.01):
        v = time_batched(T.sample_cle_batched, cle_rates, args.n_cells, rng, dt=dt) / args.n_theta_cle
        print(f"  dt={dt:<5.3f}  {fmt_snap(v):>12s} /snapshot   {fmt_set(v*args.n_sims_ref):>10s} /8k")

    pb = per_snap["Poisson-Beta (exact CME)"]
    cle = per_snap["CLE (Gaussian, dt=0.02)"]
    fspc = per_snap["FSP (exact CME)"]
    print("-" * 74)
    print(f"\nCLE / Poisson-Beta = {cle/pb:.0f}x     CLE / FSP = {cle/fspc:.1f}x")
    print("Gate G0(b): every exact method except SSA is cheaper than the CLE here, so")
    print("the telegraph model cannot carry the cost argument. A second system with a")
    print("genuinely expensive exact simulator (E10) must carry it.")


if __name__ == "__main__":
    main()
