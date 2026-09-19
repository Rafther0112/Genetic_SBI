"""
run_zerofrac.py
---------------
Produces Figure 1 (premise_validation.png) and the numbers of Sections 4.1 and 4.3: mean,
Fano factor and zero fraction of the exact CME, the full CLE and the discrete-promoter
hybrid along the burst-frequency sweep, against the closed forms. No other script in the
repository generates that figure, so this one is its source of record.

Reuses run_hybrid.simulate, so it uses exactly the simulators of the paper.
Needs numpy + whatever run_hybrid imports.   Usage:
    python run_zerofrac.py --n_cells 20000 --koff 1 --ksyn 40
"""
import argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import run_hybrid as H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--koff", type=float, default=1.0)
    ap.add_argument("--ksyn", type=float, default=40.0)
    ap.add_argument("--n_cells", type=int, default=20000)
    ap.add_argument("--kon", type=float, nargs="+",
                    default=[0.025, 0.056, 0.1, 0.3, 1.0, 3.0, 20.0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", default="premise_validation",
                    help="basename for the figure; empty string to skip plotting")
    ap.add_argument("--n_plot", type=int, default=40,
                    help="k_on grid points for the figure (log-spaced over the sweep)")
    a = ap.parse_args()

    try:
        from scipy.special import hyp1f1
        exact_zero = lambda k: float(hyp1f1(k, k + a.koff, -a.ksyn))
    except Exception:
        exact_zero = None
    exact_mean = lambda k: a.ksyn * k / (k + a.koff)
    sims = ["cme", "cle", "hybrid"]

    print(f"k_off = {a.koff}, k_syn = {a.ksyn}, {a.n_cells} cells per point\n")
    print(f"{'k_on':>8s}  {'mean: cme':>10s} {'cle':>8s} {'hybrid':>8s}   "
          f"{'zero: cme':>10s} {'cle':>8s} {'hybrid':>8s}   {'analytic zero':>13s}")
    for k in a.kon:
        th = np.log10(np.array([[k, a.koff, a.ksyn]]))
        m, z = {}, {}
        for sim in sims:
            rng = np.random.default_rng(a.seed)
            # ask for the raw counts by simulating one theta with n_cells cells
            x = H.simulate(th, sim, a.n_cells, rng)          # (1, 6) summaries
            m[sim] = np.expm1(x[0, 0])                        # log1p(mean) -> mean
            z[sim] = x[0, 3]                                  # zero fraction
        az = f"{exact_zero(k):13.3f}" if exact_zero else " " * 13
        print(f"{k:8.3f}  {m['cme']:10.2f} {m['cle']:8.2f} {m['hybrid']:8.2f}   "
              f"{z['cme']:10.3f} {z['cle']:8.3f} {z['hybrid']:8.3f}   {az}")
        if abs(k - a.kon[-1]) < 1e-12:
            for sim in ["cle", "hybrid"]:
                d = 100 * (m[sim] - exact_mean(k)) / exact_mean(k)
                print(f"           mean deficit of {sim} at k_on={k}: {d:+.1f}% "
                      f"(analytic exact mean {exact_mean(k):.2f})")
    if a.plot:
        grid = np.logspace(np.log10(min(a.kon)), np.log10(max(a.kon)), a.n_plot)
        curves = {s: {k: [] for k in ("mean", "fano", "zero")} for s in sims}
        for k in grid:
            th = np.log10(np.array([[k, a.koff, a.ksyn]]))
            for sim in sims:
                x = H.simulate(th, sim, a.n_cells, np.random.default_rng(a.seed))
                curves[sim]["mean"].append(np.expm1(x[0, 0]))
                curves[sim]["fano"].append(x[0, 2])
                curves[sim]["zero"].append(x[0, 3])
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        col = {"cme": "#0f6e56", "cle": "#ba7517", "hybrid": "#444444"}
        lab = {"cme": "exact CME", "cle": "CLE", "hybrid": "hybrid (discrete promoter)"}
        fig, ax = plt.subplots(1, 3, figsize=(12.6, 3.6))
        for j, (key, ylab) in enumerate([("mean", "mean mRNA"), ("fano", "Fano factor"),
                                         ("zero", "fraction of zero-count cells")]):
            for sim in sims:
                ax[j].plot(grid, curves[sim][key], "-", color=col[sim], lw=1.8, label=lab[sim])
            if key == "fano":
                ax[j].plot(grid, [1 + a.ksyn * a.koff / ((k + a.koff) * (k + a.koff + 1))
                                  for k in grid], ":", color="k", lw=1, label="closed form")
                ax[j].set_yscale("log")
            if key == "mean":
                ax[j].plot(grid, [exact_mean(k) for k in grid], ":", color="k", lw=1,
                           label="closed form")
            if key == "zero" and exact_zero:
                ax[j].plot(grid, [exact_zero(k) for k in grid], ":", color="k", lw=1,
                           label="closed form")
            ax[j].axvspan(grid[0], 0.3, color="0.92", zorder=0)
            ax[j].set_xscale("log"); ax[j].set_xlabel(r"burst frequency $k_{on}$")
            ax[j].set_ylabel(ylab); ax[j].legend(fontsize=7)
        plt.tight_layout()
        for ext in ("png", "pdf"):
            plt.savefig(f"{a.plot}.{ext}", dpi=150, bbox_inches="tight")
        print(f"\n[plot] {a.plot}.png/.pdf  ({a.n_plot} k_on points, shaded = bursty regime)")
        print(f"[grid] k_on from {grid[0]:.4g} to {grid[-1]:.4g}, log-spaced, "
              f"k_off={a.koff}, k_syn={a.ksyn}")

    print("\nSections 4.1 and 4.3 quote the bursty-end values and the fast-switching mean")
    print("deficit from this table. Paste it to Claude.")


if __name__ == "__main__":
    main()