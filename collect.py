"""
collect.py
----------
Aggregates results_lastmile/ into the numbers the paper needs, mean +/- std over seeds and
paired differences within seed (same test set, same simulations).

  python collect.py crossover
  python collect.py placement
  python collect.py diag  [--tv_thr 0.05 --unc_thr 0.10]   # set to the paper's thresholds
"""

import argparse, glob, json
from collections import defaultdict
import numpy as np

# Trained-estimator coverage behind Figure 5B / Table 6 (k_on telegraph, k_syn three-state)
TRAINED_COV = {("telegraph", "nb"): 0.91, ("telegraph", "lna"): 0.51,
               ("telegraph", "cle"): 0.18, ("threestate", "hybrid"): 0.93,
               ("threestate", "nb"): 0.79, ("threestate", "cle"): 0.23}


def ms(v):
    v = np.asarray(v, float)
    return f"{v.mean():+.3f} ± {v.std():.3f} (n={len(v)})"


def load(pattern):
    rows = [json.load(open(f)) for f in sorted(glob.glob(pattern))]
    mock = [r for r in rows if r.get("mock")]
    if mock:
        print(f"!! {len(mock)} files come from MOCK_NPE dry runs; do not report them")
    return rows


def crossover():
    rows = load("results_lastmile/crossover/*.json")
    g = defaultdict(list)
    for r in rows:
        g[(r["system"], r["surrogate"], r["f"], r["k_exact"])].append(r)
    for (sysm, sur, f, k), rs in sorted(g.items()):
        print(f"\n{sysm} / {sur}  f={f}  ({k} exact sims), {len(rs)} seeds")
        for arm in ["exact_only", "mixture"]:
            print(f"  {arm:10s} cov {ms([r[arm]['coverage'] for r in rs])}   "
                  f"bias {ms([r[arm]['signed_bias'] for r in rs])}   "
                  f"width(std) {ms([r[arm]['post_std'] for r in rs])}")
        d_cov = [r["mixture"]["dist_nominal"] - r["exact_only"]["dist_nominal"] for r in rs]
        d_bias = [r["mixture"]["abs_bias"] - r["exact_only"]["abs_bias"] for r in rs]
        wins = sum(x < 0 for x in d_cov)
        print(f"  mixture - exact_only:  dist-to-nominal {ms(d_cov)}  |bias| {ms(d_bias)}"
              f"   -> mixture closer to nominal in {wins}/{len(rs)} seeds")


def placement():
    rows = load("results_lastmile/placement/*.json")
    g = defaultdict(list)
    for r in rows:
        g[(r["system"], r["surrogate"], r["target"], r["f"])].append(r)
    for key, rs in sorted(g.items()):
        print(f"\n{key}: {len(rs)} seeds  (positive = better than uniform, paired by seed)")
        for sch in ["exact_only", "fano_hard", "fano_soft2", "fano_soft6", "inverse"]:
            if sch not in rs[0]:
                continue
            dd = [r["uniform"]["dist_nominal"] - r[sch]["dist_nominal"] for r in rs]
            db = [r["uniform"]["abs_bias"] - r[sch]["abs_bias"] for r in rs]
            print(f"  {sch:11s} dist {ms(dd)}   |bias| {ms(db)}   "
                  f"better in {sum(x > 0 for x in dd)}/{len(rs)} seeds")
        print(f"  uniform coverage {ms([r['uniform']['coverage'] for r in rs])}")


def diag(tv_thr, unc_thr):
    from scipy.stats import spearmanr, kendalltau
    rows = load("results_lastmile/diag_cost/*.json")
    tv = defaultdict(list); unc = defaultdict(list)          # (system, pair, N) -> seeds
    for r in rows:
        for pair, byN in r["pairs"].items():
            for N, d in byN.items():
                tv[(r["system"], pair, int(N))].append(d["tv_mean"])
                unc[(r["system"], pair, int(N))].append(d["uncovered_mean"])
    Ns = sorted({k[2] for k in tv})
    pairs = sorted({(k[0], k[1]) for k in tv})
    print("mean TV / uncovered over seeds, and verdict (support / shape / none)")
    for (s, pr) in pairs:
        line = []
        for N in Ns:
            t, u = np.mean(tv[(s, pr, N)]), np.mean(unc[(s, pr, N)])
            v = "support" if u > unc_thr else ("shape" if t > tv_thr else "none")
            line.append(f"N={N}: {t:.3f}/{u:.3f} {v}")
        print(f"  {s:10s} {pr:7s} | " + " | ".join(line))
    ranked = [p for p in pairs if p in TRAINED_COV]
    if len(ranked) >= 3:
        print("\nranking vs trained coverage (Spearman, expected negative) and vs N_max (Kendall)")
        ref = [np.mean(tv[(s, pr, Ns[-1])]) for s, pr in ranked]
        for N in Ns:
            t = [np.mean(tv[(s, pr, N)]) for s, pr in ranked]
            rho = spearmanr(t, [TRAINED_COV[p] for p in ranked])[0]
            tau = kendalltau(t, ref)[0]
            print(f"  N={N:6d}  Spearman(TV, coverage)={rho:+.2f}  Kendall(TV_N, TV_Nmax)={tau:+.2f}"
                  f"   over {len(ranked)} pairs")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["crossover", "placement", "diag"])
    p.add_argument("--tv_thr", type=float, default=0.05)
    p.add_argument("--unc_thr", type=float, default=0.10)
    a = p.parse_args()
    {"crossover": crossover, "placement": placement}.get(a.what, lambda: diag(a.tv_thr, a.unc_thr))()
