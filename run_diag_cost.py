"""
run_diag_cost.py
----------------
Item 3: how many simulations does the classifier diagnostic (Section 4.4) need before its
ranking and its support/shape verdicts stabilise?

Per seed we simulate the exact simulator twice (the second copy is the exact-vs-exact
control) and each surrogate once, all on the same N_max prior draws, then subsample the
first N draws for every N in --Ns. For each (pair, N):

  * a classifier separates exact (label 1) from surrogate (label 0) on (log theta, x),
    with cross-fitted (K-fold, out-of-sample) probabilities;
  * the draws are split into equal-count bins of the analytic Fano factor;
  * per bin, TV_b = max(0, 2 AUC_b - 1), and the uncovered mass is the fraction of exact
    draws whose estimated density ratio p_s/p_e = (1 - c)/c falls below --eps;
  * the summary is the bin average of TV_b and of the uncovered mass.

BEFORE trusting the curve, check the N = N_max row against Table 6 (e.g. telegraph NB TV
about 0.37, LNA 0.94, CLE 0.98). If your original implementation used a different
classifier, number of bins or eps, either set the flags to match or plug your function into
`diagnose()` below; the cost curve must come from the same estimator as Table 6.

Examples
  python run_diag_cost.py --system telegraph --surrogates nb lna cle --seed 0
  python run_diag_cost.py --system threestate --surrogates hybrid nb cle --seed 0
"""

import argparse
import numpy as np
import common as C


def classifier_probs(F, y, seed, folds=5):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold
    p = np.empty(len(y))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(F, y):
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                             early_stopping=True, random_state=seed)
        clf.fit(F[tr], y[tr])
        p[te] = clf.predict_proba(F[te])[:, 1]
    return p


def diagnose(theta, x_exact, x_sur, fano, n_bins, eps, seed):
    from sklearn.metrics import roc_auc_score
    n = len(theta)
    F = np.vstack([np.hstack([theta, x_exact]), np.hstack([theta, x_sur])])
    y = np.r_[np.ones(n), np.zeros(n)]
    c = np.clip(classifier_probs(F, y, seed), 1e-6, 1 - 1e-6)
    edges = np.quantile(fano, np.linspace(0, 1, n_bins + 1)); edges[-1] += 1e-9
    b = np.clip(np.digitize(fano, edges) - 1, 0, n_bins - 1)
    bb = np.r_[b, b]
    tv, unc = [], []
    for k in range(n_bins):
        m = bb == k
        auc = roc_auc_score(y[m], c[m]) if len(np.unique(y[m])) == 2 else 0.5
        tv.append(max(0.0, 2 * auc - 1))
        ce = c[:n][b == k]                               # exact draws in the bin
        unc.append(float(np.mean((1 - ce) / ce < eps)))
    return dict(tv_bins=tv, uncovered_bins=unc,
                tv_mean=float(np.mean(tv)), uncovered_mean=float(np.mean(unc)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--system", default="telegraph", choices=["telegraph", "threestate"])
    p.add_argument("--surrogates", nargs="+", default=["nb", "lna", "cle"])
    p.add_argument("--Ns", type=int, nargs="+", default=[250, 500, 1000, 2000, 4000, 12000])
    p.add_argument("--n_cells", type=int, default=500)
    p.add_argument("--n_bins", type=int, default=6)
    p.add_argument("--eps", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results_lastmile/diag_cost")
    a = p.parse_args()

    spec = C.system_spec(a.system)
    n_max = max(a.Ns)
    rng = np.random.default_rng(20_000 + a.seed)
    theta = C.sample_prior(n_max, rng, spec)
    tag = f"diag{a.seed}_{n_max}"
    x_ex = C.cached_sims(tag, theta, "cme", a.n_cells, a.seed, a.system)
    x_ex2 = C.cached_sims(tag + "_copy", theta, "cme", a.n_cells, a.seed, a.system)
    sur = {s: C.cached_sims(tag, theta, s, a.n_cells, a.seed, a.system)
           for s in a.surrogates}
    fano = C.analytic_fano(theta, a.system)
    lt = np.asarray(theta)

    res = dict(system=a.system, seed=a.seed, Ns=a.Ns, n_bins=a.n_bins, eps=a.eps, pairs={})
    for name, xs in [("exact", x_ex2)] + list(sur.items()):
        res["pairs"][name] = {}
        for N in sorted(a.Ns):
            d = diagnose(lt[:N], x_ex[:N], xs[:N], fano[:N], a.n_bins, a.eps, a.seed)
            res["pairs"][name][str(N)] = d
            print(f"[{a.system}/{name}] N={N:6d}  TV={d['tv_mean']:.3f}  "
                  f"uncovered={d['uncovered_mean']:.3f}", flush=True)
    C.save_json(f"{a.out}/{a.system}_seed{a.seed}.json", res)


if __name__ == "__main__":
    main()
