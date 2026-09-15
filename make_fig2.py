"""
make_fig2.py
------------
Rebuild Figure 2 (money_plot) and the numbers of Tables 1 and 4 from the output of
run_reconcile.py, so the figure, the tables and the text all come from ONE run in
which every estimator is evaluated on the SAME test set per seed.

Reads  results/reconcile.csv  (and results/leakage.csv if present).
Writes money_plot.png / money_plot.pdf and prints:
  * Table 4 rows (per-sextile k_on coverage, mean +/- std over seeds, n, Fano range)
  * Table 1 check (overall coverage mean +/- std, signed bias) and leaked fraction
  * App. A check: per-seed Spearman(bin, coverage) and whether ties occur

No torch/sbi needed.   Usage:
    python make_fig2.py --csv results/reconcile.csv --leak results/leakage.csv
"""
import argparse, csv, os, sys
from collections import defaultdict
import numpy as np

ORDER = ["cme", "nb", "lna", "cle", "hybrid"]
COL = {"cme": "#0f6e56", "nb": "#2b6cb0", "lna": "#7e4ea8", "cle": "#ba7517", "hybrid": "#444444"}
LAB = {"cme": "CME (control)", "nb": "NB (count control)", "lna": "LNA", "cle": "CLE", "hybrid": "hybrid"}
MK = {"cme": "o", "nb": "D", "lna": "^", "cle": "s", "hybrid": "v"}


def find(cols, *cands, required=True):
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c in low:
            return low[c]
    if required:
        sys.exit(f"[make_fig2] none of {cands} in CSV columns {cols}. Tell Claude the header.")
    return None


def spearman(x, y):
    from scipy.stats import spearmanr
    return spearmanr(x, y).correlation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/reconcile.csv")
    ap.add_argument("--leak", default="results/leakage.csv")
    ap.add_argument("--param", default="k_on")
    ap.add_argument("--out", default="money_plot")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.csv)))
    cols = list(rows[0].keys())
    print("[columns]", cols)
    c_seed = find(cols, "seed")
    c_sim = find(cols, "sim", "simulator", "estimator")
    c_par = find(cols, "param", "parameter")
    c_bin = find(cols, "bin")
    c_cov = find(cols, "coverage", "cov90", "cov")
    c_lo = find(cols, "cov_lo", "coverage_lo", "lo", required=False)
    c_hi = find(cols, "cov_hi", "coverage_hi", "hi", required=False)
    c_sb = find(cols, "signed_bias", "bias")
    c_n = find(cols, "n", "n_bin", "count", required=False)
    c_el = find(cols, "edge_lo", "fano_lo", "bin_lo", "lo_edge", required=False)
    c_eh = find(cols, "edge_hi", "fano_hi", "bin_hi", "hi_edge", required=False)

    R = [r for r in rows if r[c_par] == a.param]
    if not R:
        sys.exit(f"[make_fig2] no rows with {c_par} == {a.param}; values: {sorted({r[c_par] for r in rows})}")
    sims = [s for s in ORDER if any(r[c_sim].lower() == s for r in R)]
    seeds = sorted({r[c_seed] for r in R}, key=lambda s: int(float(s)))
    bins = sorted({int(float(r[c_bin])) for r in R if r[c_bin] != "all"})
    print(f"[info] sims={sims} seeds={seeds} bins={bins}")

    cov = defaultdict(dict); sb = defaultdict(dict); lo = defaultdict(dict); hi = defaultdict(dict)
    nb = defaultdict(dict); el = defaultdict(dict); eh = defaultdict(dict); overall = defaultdict(dict)
    for r in R:
        s, sd = r[c_sim].lower(), r[c_seed]
        if r[c_bin] == "all":
            overall[(s, sd)] = (float(r[c_cov]), float(r[c_sb])); continue
        b = int(float(r[c_bin]))
        cov[(s, b)][sd] = float(r[c_cov]); sb[(s, b)][sd] = float(r[c_sb])
        if c_lo: lo[(s, b)][sd] = float(r[c_lo]); hi[(s, b)][sd] = float(r[c_hi])
        if c_n: nb[b][sd] = int(float(r[c_n]))
        if c_el: el[b][sd] = float(r[c_el]); eh[b][sd] = float(r[c_eh])

    # x positions: geometric centre of seed-averaged sextile edges if available
    if c_el:
        E_lo = np.array([np.mean(list(el[b].values())) for b in bins])
        E_hi = np.array([np.mean(list(eh[b].values())) for b in bins])
        xs = np.sqrt(np.clip(E_lo, 0.3, None) * E_hi); xlog = True
    else:
        xs = np.array(bins, float); xlog = False
        print("[warn] no edge columns found: x-axis uses sextile index")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for s in sims:
        m_sb = np.array([np.mean(list(sb[(s, b)].values())) for b in bins])
        e_sb = np.array([np.std(list(sb[(s, b)].values())) for b in bins])
        ax[0].errorbar(xs, m_sb, yerr=e_sb, fmt=MK[s] + "-", color=COL[s], label=LAB[s], capsize=2, lw=1.5)
        m_c = np.array([np.mean(list(cov[(s, b)].values())) for b in bins])
        if c_lo:
            l = np.array([np.mean(list(lo[(s, b)].values())) for b in bins])
            h = np.array([np.mean(list(hi[(s, b)].values())) for b in bins])
            yerr = [m_c - l, h - m_c]
        else:
            yerr = np.array([np.std(list(cov[(s, b)].values())) for b in bins])
        ax[1].errorbar(xs, m_c, yerr=yerr, fmt=MK[s] + "-", color=COL[s], label=LAB[s], capsize=2, lw=1.5)
    ax[0].axhline(0, ls=":", color="gray")
    ax[1].axhline(0.9, ls=":", color="gray", label="nominal 90%")
    ax[0].set_ylabel(r"signed posterior-mean bias of $k_{on}$ ($\log_{10}$)")
    ax[1].set_ylabel(r"90% CI coverage of $k_{on}$"); ax[1].set_ylim(0, 1)
    ax[0].set_title("A. Signed bias vs burstiness"); ax[1].set_title("B. Coverage vs burstiness")
    for x in ax:
        if xlog: x.set_xscale("log")
        x.set_xlabel("empirical Fano factor of snapshot (sextile)"); x.legend(fontsize=8)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(f"{a.out}.{ext}", dpi=150, bbox_inches="tight")
    bars = "mean over seeds of per-seed bootstrap 95% CI" if c_lo else "std over seeds"
    print(f"[plot] {a.out}.png/.pdf  (coverage bars = {bars}; bias bars = std over seeds)")

    # ---- Table 4 rows
    print("\n=== Table 4 rows (paste) ===")
    for b in bins:
        n = int(round(np.mean(list(nb[b].values())))) if c_n else -1
        rng = f"$[{np.mean(list(el[b].values())):.1f},{np.mean(list(eh[b].values())):.1f}]$" if c_el else "?"
        cells = " & ".join(
            f"${np.mean(list(cov[(s, b)].values())):.2f}{{\\pm}}.{int(round(100*np.std(list(cov[(s, b)].values())))):02d}$"
            for s in sims)
        print(f"    {b} & {rng} & {n} & {cells} \\\\")
    if c_n:
        per_seed = {sd: sum(nb[b].get(sd, 0) for b in bins) for sd in seeds}
        print(f"[n check] per-seed totals: {per_seed}")

    # ---- Table 1 check
    print("\n=== Table 1 check (bin=all, mean +/- std over seeds) ===")
    for s in sims:
        v = [overall[(s, sd)] for sd in seeds if (s, sd) in overall]
        if v:
            c = np.array([x[0] for x in v]); g = np.array([x[1] for x in v])
            print(f"  {s.upper():6s} coverage {c.mean():.2f} +/- {c.std():.2f}   signed bias {g.mean():+.2f}")

    # ---- App. A check
    print("\n=== App. A: Spearman(sextile index, coverage) per seed ===")
    for s in sims:
        out = []
        for sd in seeds:
            y = [cov[(s, b)].get(sd, np.nan) for b in bins]
            ties = len(set(np.round(y, 4))) < len(y)
            out.append(f"{spearman(bins, y):+.2f}{'(tie)' if ties else ''}")
        print(f"  {s.upper():6s} " + "  ".join(out))

    # ---- leakage
    if os.path.exists(a.leak):
        L = list(csv.DictReader(open(a.leak))); lc = list(L[0].keys())
        ls = find(lc, "sim", "simulator"); lb = find(lc, "bin"); lf = find(lc, "leaked_frac", "leaked")
        print("\n=== leaked fraction (bin=all, mean over seeds) ===")
        for s in sims:
            v = [float(r[lf]) for r in L if r[ls].lower() == s and r[lb] == "all"]
            if v: print(f"  {s.upper():6s} {np.mean(v):.2f}")


if __name__ == "__main__":
    main()
