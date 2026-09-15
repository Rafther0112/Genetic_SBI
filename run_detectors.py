"""
run_detectors.py
----------------
E8 (originality, W2): compare failure detectors by ROC/AUC. Per test observation we
score the empirical Fano factor, the zero fraction, the leaked posterior mass, and a
training-summary-density OOD baseline, against the label "the LNA-trained posterior's
90% interval misses the true k_on". Tests C3: does the Fano factor detect failure at
least as well as the OOD baseline? See docs/prereg_diagnostic.md.

Needs torch + sbi + scikit-learn. Reuses run_experiment.train (cached).

Usage:
    python run_detectors.py --n_sims 8000 --n_cells 500 --n_test 2000 --seed 0
"""

import argparse
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import multivariate_normal
from sklearn.metrics import roc_auc_score

import run_experiment as R
from sbi_experiment import make_prior, simulate_batch
from boxflow import LOW, HIGH


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_sims", type=int, default=8000)
    p.add_argument("--n_cells", type=int, default=500)
    p.add_argument("--n_test", type=int, default=2000)
    p.add_argument("--n_post", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    # LNA surrogate of record + its training summaries (for the OOD baseline)
    theta_tr, x_tr = R.cached_training_set("lna", args.n_sims, args.n_cells, args.seed)
    post = R.train("lna", args.n_sims, args.n_cells, args.seed)

    # OOD baseline: Gaussian fit to the surrogate's training summaries
    Xtr = x_tr.numpy()
    mu = Xtr.mean(0); cov = np.cov(Xtr, rowvar=False) + 1e-4 * np.eye(Xtr.shape[1])
    mvn = multivariate_normal(mu, cov, allow_singular=True)

    # exact-CME test set
    prior = make_prior()
    rng = np.random.default_rng(1000 + args.seed)
    tt = prior.sample((args.n_test,)).numpy()
    x_test = torch.as_tensor(simulate_batch(tt, "cme", args.n_cells, rng), dtype=torch.float32)
    Xte = x_test.numpy()

    fano = Xte[:, 2]; zero = Xte[:, 3]
    ood = -mvn.logpdf(Xte)                       # high = out-of-distribution
    leaked = np.empty(args.n_test); label = np.empty(args.n_test)
    ppc = np.empty(args.n_test)                   # posterior-predictive-check discrepancy
    rng_ppc = np.random.default_rng(7)
    n_pp = 40                                     # posterior draws for the PPC
    for i in range(args.n_test):
        s = R._flow_sample(post, x_test[i], args.n_post).numpy()
        leaked[i] = np.mean(np.any((s < LOW) | (s > HIGH), axis=1))
        rank = (s[:, 0] < tt[i, 0]).mean()
        label[i] = 1.0 if (rank < 0.05 or rank > 0.95) else 0.0
        # PPC: draw thetas from the surrogate posterior, simulate from the SURROGATE, and
        # measure how far the observed summaries fall from the predictive summary cloud.
        idx = rng_ppc.choice(len(s), n_pp, replace=False)
        xpp = simulate_batch(np.clip(s[idx], LOW, HIGH), "lna", args.n_cells, rng_ppc)  # (n_pp,6)
        mu_pp = xpp.mean(0); sd_pp = xpp.std(0) + 1e-6
        ppc[i] = np.mean(((Xte[i] - mu_pp) / sd_pp) ** 2)   # mean squared z-score

    print(f"\nfailure rate (LNA misses true k_on): {label.mean():.2f}  "
          f"({int(label.sum())}/{args.n_test})")
    print("\ndetector AUC (predicting LNA failure per observation):")
    for name, score in [("empirical Fano (pre-inf)", fano), ("zero fraction (pre-inf)", zero),
                        ("leaked mass (post-inf)", leaked),
                        ("OOD density (baseline)", ood),
                        ("posterior-pred check (baseline)", ppc)]:
        auc = roc_auc_score(label, score)
        print(f"  {name:34s} AUC = {auc:.3f}")
    print("\nC3 holds if Fano (and/or leaked mass) matches or beats BOTH baselines")
    print("(OOD density and the posterior-predictive check).")


if __name__ == "__main__":
    main()
