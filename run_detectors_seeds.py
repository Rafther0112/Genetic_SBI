"""
run_detectors_seeds.py
----------------------
Section 4.4 over five seeds. Same detectors and failure label as run_detectors.py
(empirical Fano, zero fraction, leaked posterior mass, OOD density on the LNA training
summaries; label = the LNA-trained 90% interval misses the true k_on), now with
  * torch seeded per seed, so every number is reproducible
  * AUC mean +/- std over seeds, and a paired Fano-vs-OOD difference
  * the LNA failure rate per seed (to reconcile with Table 1's coverage 0.51)
  * the flag at Fano* : fraction of test observations flagged and fraction of failures caught
  * the empirical-vs-analytic Fano correlation at this snapshot size
The posterior-predictive check is added only if run_detectors.py exposes a function
named ppc_score(post, x_test, n_cells, rng) returning one score per observation.

Needs torch + sbi + scikit-learn.   Usage:
    python run_detectors_seeds.py --n_sims 8000 --n_cells 500 --n_test 2000 --seeds 0 1 2 3 4 --fano_star 4.3
"""
import argparse, csv, os
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import multivariate_normal
from sklearn.metrics import roc_auc_score

import run_experiment as R
from sbi_experiment import make_prior, simulate_batch, theta_to_rates
from telegraph import exact_fano
from boxflow import LOW, HIGH

# Posterior-predictive check, copied verbatim from run_detectors.py so the two scripts
# score the same statistic: draw n_pp parameters from the surrogate posterior, simulate
# from the SURROGATE, and measure how far the observed summaries fall from that cloud.
N_PP = 40


def ppc_scores(post, x_test, tt, n_cells, n_post):
    lo, hi = np.asarray(LOW), np.asarray(HIGH)
    rng_ppc = np.random.default_rng(7)            # same seed as run_detectors.py
    out = np.empty(len(tt))
    for i in range(len(tt)):
        s = R._flow_sample(post, x_test[i], n_post).numpy()
        idx = rng_ppc.choice(len(s), N_PP, replace=False)
        xpp = simulate_batch(np.clip(s[idx], lo, hi), "lna", n_cells, rng_ppc)
        mu_pp = xpp.mean(0); sd_pp = xpp.std(0) + 1e-6
        out[i] = np.mean(((x_test[i].numpy() - mu_pp) / sd_pp) ** 2)
    return out


def one_seed(seed, a):
    torch.manual_seed(seed)
    theta_tr, x_tr = R.cached_training_set("lna", a.n_sims, a.n_cells, seed)
    post = R.train("lna", a.n_sims, a.n_cells, seed)
    Xtr = x_tr.numpy()
    mvn = multivariate_normal(Xtr.mean(0), np.cov(Xtr, rowvar=False) + 1e-4 * np.eye(Xtr.shape[1]),
                              allow_singular=True)

    torch.manual_seed(1000 + seed)
    tt = make_prior().sample((a.n_test,)).numpy()
    x_test = torch.as_tensor(simulate_batch(tt, "cme", a.n_cells, np.random.default_rng(1000 + seed)),
                             dtype=torch.float32)
    X = x_test.numpy()
    scores = {"empirical Fano": X[:, 2], "zero fraction": X[:, 3],
              "OOD density": -mvn.logpdf(X)}
    leaked = np.empty(a.n_test); label = np.empty(a.n_test)
    lo, hi = np.asarray(LOW), np.asarray(HIGH)
    for i in range(a.n_test):
        s = R._flow_sample(post, x_test[i], a.n_post).numpy()
        leaked[i] = np.mean(np.any((s < lo) | (s > hi), axis=1))
        r = (s[:, 0] < tt[i, 0]).mean()
        label[i] = 1.0 if (r < 0.05 or r > 0.95) else 0.0
    scores["leaked mass"] = leaked
    if not a.no_ppc:
        scores["posterior-predictive"] = ppc_scores(post, x_test, tt, a.n_cells, a.n_post)

    auc = {k: roc_auc_score(label, v) for k, v in scores.items()}
    flag = X[:, 2] > a.fano_star
    ana = np.array([exact_fano(*r) for r in theta_to_rates(tt)])
    corr = np.corrcoef(np.log(ana), np.log(np.clip(X[:, 2], 1e-3, None)))[0, 1]
    return dict(seed=seed, failure_rate=label.mean(), flagged=flag.mean(),
                caught=(flag & (label == 1)).sum() / max(label.sum(), 1),
                corr_logfano=corr, **{f"auc_{k}": v for k, v in auc.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_sims", type=int, default=8000)
    ap.add_argument("--n_cells", type=int, default=500)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--n_post", type=int, default=1000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--fano_star", type=float, default=4.3)
    ap.add_argument("--no_ppc", action="store_true",
                    help="skip the posterior-predictive check (it adds ~1 min per seed)")
    ap.add_argument("--out", default="results/detectors_seeds.csv")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    rows = []
    for seed in a.seeds:
        rows.append(one_seed(seed, a))
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    print(f"\n=== over seeds {a.seeds} (mean +/- std) ===")
    for k in rows[0]:
        if k == "seed": continue
        v = np.array([r[k] for r in rows])
        print(f"  {k:28s} {v.mean():.3f} +/- {v.std():.3f}")
    print("\n  paired differences (same test points per seed):")
    for base in ["OOD density", "posterior-predictive"]:
        k = f"auc_{base}"
        if k not in rows[0]:
            continue
        for probe in ["empirical Fano", "zero fraction", "leaked mass"]:
            d = np.array([r[f"auc_{probe}"] - r[k] for r in rows])
            print(f"    AUC({probe}) - AUC({base}): {d.mean():+.3f} +/- {d.std():.3f}"
                  f"  (positive in {int((d > 0).sum())}/{len(d)} seeds)")
    dz = np.array([r["auc_zero fraction"] - r["auc_empirical Fano"] for r in rows])
    print(f"    AUC(zero fraction) - AUC(empirical Fano): {dz.mean():+.3f} +/- {dz.std():.3f}"
          f"  (positive in {int((dz > 0).sum())}/{len(dz)} seeds)")
    print("  LNA coverage implied by failure rate: "
          f"{1 - np.mean([r['failure_rate'] for r in rows]):.2f} (Table 1 reports 0.51)")
    print(f"[written] {a.out}")


if __name__ == "__main__":
    main()