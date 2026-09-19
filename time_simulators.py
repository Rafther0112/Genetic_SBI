"""
time_simulators.py
------------------
Fills the empty cells of the cost table (Table 5): wall-clock of every simulator used in
the paper, on this machine, for a subset of parameters, extrapolated to the training-set
size actually used. Also prints the fraction of the telegraph prior above Fano* and
the correlation between empirical and analytic Fano at the paper's snapshot size.

Telegraph   : cme (Poisson-Beta), lna, nb, cle  via sbi_experiment.simulate_batch
              hybrid                             via run_hybrid.simulate
Three-state : every name in run_threestate.SIMS via run_threestate.simulate
SIR         : every name in run_sir.SIMS         via run_sir.simulate
(FSP and SSA for the telegraph are already in the table from run_cost.py.)

Run on the SAME machine as run_cost.py, with nothing else heavy running.
Usage:
    python time_simulators.py --n_theta 200 --n_cells 500 --n_real 200
"""
import argparse, time
import numpy as np


def timed(fn, *args):
    t = time.perf_counter(); fn(*args); return time.perf_counter() - t


def report(system, name, sec, n_theta, n_train):
    per = sec / n_theta * n_train
    unit = f"{per:.1f} s" if per < 120 else f"{per/60:.1f} min"
    print(f"  {system:12s} {name:10s} {sec:8.2f} s for {n_theta:4d} theta  ->  {unit} per {n_train} sims")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_theta", type=int, default=200)
    ap.add_argument("--n_cells", type=int, default=500)
    ap.add_argument("--n_real", type=int, default=200)
    ap.add_argument("--n_fsp", type=int, default=100)
    ap.add_argument("--n_ssa", type=int, default=32,
                    help="SSA cost varies by orders of magnitude across the prior; "
                         "this many parameters are timed individually")
    ap.add_argument("--n_train_sir", type=int, default=3000)
    ap.add_argument("--fano_star", type=float, default=4.3)
    a = ap.parse_args()

    from sbi_experiment import make_prior, simulate_batch, theta_to_rates
    from telegraph import exact_fano
    print("\n=== wall-clock (extrapolated linearly) ===")
    th = make_prior().sample((a.n_theta,)).numpy()
    for sim in ["cme", "lna", "nb", "cle"]:
        report("telegraph", sim, timed(simulate_batch, th, sim, a.n_cells, np.random.default_rng(0)),
               a.n_theta, 8000)
    try:
        import run_hybrid as H
        report("telegraph", "hybrid", timed(H.simulate, th, "hybrid", a.n_cells, np.random.default_rng(0)),
               a.n_theta, 8000)
    except Exception as e:
        print(f"  telegraph hybrid: skipped ({e})")
    try:
        import run_threestate as T3
        th3 = T3.prior().sample((a.n_theta,)).numpy()
        for sim in T3.SIMS:
            report("three-state", sim, timed(T3.simulate, th3, sim, a.n_cells, np.random.default_rng(0)),
                   a.n_theta, 8000)
    except Exception as e:
        print(f"  three-state: skipped ({e})")
    try:
        import fsp
        fn = None
        for name in ("sample_fsp_batched", "sample_fsp", "fsp_sample", "sample"):
            fn = getattr(fsp, name, None)
            if fn is not None:
                break
        if fn is None:
            print(f"  telegraph FSP: skipped (no sampler found in fsp.py; it has {[x for x in dir(fsp) if not x.startswith('_')]})")
        else:
            nf = a.n_fsp
            report("telegraph", "FSP", timed(lambda: [fn(t, a.n_cells, np.random.default_rng(0)) for t in theta_to_rates(th[:nf])]),
                   nf, 8000)
    except Exception as e:
        print(f"  telegraph FSP: skipped ({e})")
    try:
        from telegraph import sample_gillespie
        # SSA cost is roughly proportional to the number of reaction events per cell,
        # about k_syn * p_on * t_max, which spans decades over a log-uniform prior. A mean
        # over a handful of parameters is dominated by its largest draw, so we time each
        # parameter separately and report the median and the range.
        nc = min(a.n_cells, 100)
        rates = theta_to_rates(th[:a.n_ssa])
        per = np.array([timed(sample_gillespie, t, nc, np.random.default_rng(0)) for t in rates])
        per8k = per * (a.n_cells / nc) * 8000          # seconds per 8,000-simulation set
        load = rates[:, 2] * rates[:, 0] / (rates[:, 0] + rates[:, 1])   # k_syn * p_on
        r = np.corrcoef(np.log(load), np.log(per))[0, 1]
        q = np.percentile(per8k, [50, 25, 75, 5, 95])
        print(f"  {'telegraph':12s} {'SSA':10s} per-theta over {a.n_ssa} draws, {nc} cells, "
              f"scaled to {a.n_cells} cells x 8000 sims:")
        print(f"               median {q[0]/60:.1f} min   IQR {q[1]/60:.1f}-{q[2]/60:.1f} min   "
              f"5-95% {q[3]/60:.2f}-{q[4]/60:.1f} min   mean {per8k.mean()/60:.1f} min")
        print(f"               corr(log k_syn*p_on, log time) = {r:.2f}  "
              f"(cost spans {per8k.max()/per8k.min():.0f}x across the prior)")
    except Exception as e:
        print(f"  telegraph SSA: skipped ({e})")
    try:
        import run_sir as S
        n_sir = max(10, a.n_theta // 4)
        ths = S.prior().sample((n_sir,)).numpy()
        for sim in S.SIMS:
            report("SIR", sim, timed(S.simulate, ths, sim, a.n_real, np.random.default_rng(0)),
                   n_sir, a.n_train_sir)
    except Exception as e:
        print(f"  SIR: skipped ({e})")

    print("\n=== telegraph prior checks ===")
    rng = np.random.default_rng(1)
    big = make_prior().sample((200000,)).numpy()
    F = np.array([exact_fano(*r) for r in theta_to_rates(big[:50000])])
    print(f"  fraction of prior with analytic Fano > 2      : {(F > 2).mean():.2f}")
    print(f"  fraction of prior with analytic Fano > {a.fano_star:<4}  : {(F > a.fano_star).mean():.2f}")
    t2 = make_prior().sample((2000,)).numpy()
    x = simulate_batch(t2, "cme", a.n_cells, rng)
    ana = np.array([exact_fano(*r) for r in theta_to_rates(t2)])
    corr = np.corrcoef(np.log(ana), np.log(np.clip(x[:, 2], 1e-3, None)))[0, 1]
    print(f"  corr(log analytic Fano, log empirical Fano) at {a.n_cells} cells: {corr:.3f}")


if __name__ == "__main__":
    main()