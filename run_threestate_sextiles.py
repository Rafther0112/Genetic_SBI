"""
run_threestate_sextiles.py
--------------------------
Regenerates Figure 3 and Table 4 (three-state promoter) with equal-count SEXTILES of
the empirical Fano factor, one shared exact test set per seed, and every estimator in
run_threestate.SIMS evaluated on that same set. Reuses prior, simulate, train,
flow_sample, SIMS and PARAM from run_threestate.py (run from the repo root).

Writes results/threestate_seed{S}.npz per seed, threestate_coverage.png/.pdf, and
prints the per-sextile table (n, Fano range, coverage per simulator) plus overall
coverage, ready for Table 4 and Section 4.6.

Needs torch + sbi.   Usage:
    python run_threestate_sextiles.py --n_sims 8000 --n_cells 500 --n_test 2000 --seeds 0
    python run_threestate_sextiles.py --plot_only --seeds 0      # re-plot from saved npz
"""
import argparse, os
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

import run_threestate as T3

N_BINS = 6
COL = {"exact": "#0f6e56", "hybrid": "#444444", "nb": "#2b6cb0", "cle": "#ba7517"}


def run_seed(seed, a):
    rng = np.random.default_rng(1000 + seed)
    torch.manual_seed(1000 + seed)
    tt = T3.prior().sample((a.n_test,)).numpy()
    x_test = torch.as_tensor(T3.simulate(tt, "exact", a.n_cells, rng))
    fano = x_test[:, 2].numpy()
    edges = np.quantile(fano, np.linspace(0, 1, N_BINS + 1)); edges[-1] += 1e-9
    b = np.clip(np.digitize(fano, edges) - 1, 0, N_BINS - 1)
    out = dict(edges=edges, bin_idx=b)
    for sim in T3.SIMS:
        torch.manual_seed(seed)
        post = T3.train(sim, a.n_sims, a.n_cells, seed)
        inside = np.empty(a.n_test, bool)
        for i in range(a.n_test):
            s = T3.flow_sample(post, x_test[i], a.n_post).numpy()
            u = (s[:, T3.PARAM] < tt[i, T3.PARAM]).mean()
            inside[i] = 0.05 <= u <= 0.95
        out[sim] = inside
        print(f"[seed {seed}] {sim}: overall {inside.mean():.2f}", flush=True)
    np.savez(f"results/threestate_seed{seed}.npz", **out)


def boot(v, rng, n=1000):
    return np.percentile([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)], [2.5, 97.5])


def report(a):
    D = [dict(np.load(f"results/threestate_seed{s}.npz")) for s in a.seeds]
    sims = [s for s in T3.SIMS if s in D[0]]
    rng = np.random.default_rng(0)
    E = np.mean([d["edges"] for d in D], axis=0)
    print("\n=== Table 4 (three-state), k_syn coverage per sextile; pooled over seeds", a.seeds, "===")
    print("bin  Fano range        n   " + "  ".join(f"{s:>8s}" for s in sims))
    M = {s: [] for s in sims}; L = {s: [] for s in sims}; H = {s: [] for s in sims}
    for b in range(N_BINS):
        n = np.mean([(d["bin_idx"] == b).sum() for d in D])
        vals = []
        for s in sims:
            v = np.concatenate([d[s][d["bin_idx"] == b] for d in D]).astype(float)
            M[s].append(v.mean()); lo, hi = boot(v, rng); L[s].append(lo); H[s].append(hi)
            vals.append(f"{v.mean():8.2f}")
        print(f"{b}    [{E[b]:6.2f},{E[b+1]:7.2f}]  {n:5.0f}  " + "  ".join(vals))
    print("all                          " + "  ".join(
        f"{np.mean(np.concatenate([d[s] for d in D])):8.2f}" for s in sims))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = np.sqrt(np.clip(E[:-1], 0.3, None) * E[1:])
    plt.figure(figsize=(5, 3.6))
    for s in sims:
        m = np.array(M[s])
        plt.errorbar(xs, m, yerr=[m - np.array(L[s]), np.array(H[s]) - m], fmt="o-",
                     color=COL.get(s, None), label=s, capsize=2, lw=1.5)
    plt.axhline(0.9, ls=":", color="gray"); plt.xscale("log"); plt.ylim(0, 1)
    plt.xlabel("empirical Fano factor (sextile)"); plt.ylabel(r"90% CI coverage of $k_{syn}$")
    plt.legend(fontsize=8); plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(f"threestate_coverage.{ext}", dpi=150, bbox_inches="tight")
    print("[plot] threestate_coverage.png/.pdf (bars: bootstrap 95% CI over pooled test points)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_sims", type=int, default=8000)
    ap.add_argument("--n_cells", type=int, default=500)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--n_post", type=int, default=1000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--plot_only", action="store_true")
    a = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    if not a.plot_only:
        for s in a.seeds:
            run_seed(s, a)
    report(a)


if __name__ == "__main__":
    main()
