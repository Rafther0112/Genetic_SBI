"""
run_ablation.py
---------------
Feature-set ablation. Retrain the three estimators using ONLY the mean and
variance as summary statistics (features="moments") and evaluate on exact CME
data. This tests whether the surrogate failure comes from non-Gaussian shape or
from the specific summary statistics we chose.

Expected reading:
  * CME (control) stays calibrated  -> the feature set is not the culprit.
  * neither surrogate is rescued: enforcing non-negative counts distorts even the
    moments of a Gaussian surrogate in the bursty regime, so the LNA (built to
    match the moments) remains misspecified, and the CLE remains worst.

Usage:
    python run_ablation.py --n_sims 8000 --n_cells 500 --n_test 400
"""

import argparse
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

from sbi.inference import NPE
from sbi_experiment import make_prior, simulate_batch

SIMS = ["cme", "cle", "lna"]


def flow_sample(post, x_row, n_post):
    s = post.posterior_estimator.sample((n_post,), condition=x_row.unsqueeze(0))
    return s.reshape(n_post, -1).detach()


def run(features, n_sims, n_cells, n_test, n_post, seed):
    prior = make_prior()
    out = {}
    for sim in SIMS:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        theta = prior.sample((n_sims,))
        x = torch.as_tensor(
            simulate_batch(theta.numpy(), sim, n_cells, rng, features=features),
            dtype=torch.float32)
        inf = NPE(prior=prior)
        inf.append_simulations(theta, x)
        post = inf.build_posterior(inf.train())

        rng2 = np.random.default_rng(999)
        tt = prior.sample((n_test,))
        xt = torch.as_tensor(
            simulate_batch(tt.numpy(), "cme", n_cells, rng2, features=features),
            dtype=torch.float32)
        ttn = tt.numpy()
        ranks = np.empty((n_test, 3)); means = np.empty((n_test, 3))
        for i in range(n_test):
            s = flow_sample(post, xt[i], n_post).numpy()
            ranks[i] = (s < ttn[i]).sum(0)
            means[i] = s.mean(0)
        u = ranks[:, 0] / n_post
        cov = float(np.mean((u >= 0.05) & (u <= 0.95)))
        bias = float(np.abs(means[:, 0] - ttn[:, 0]).mean())
        out[sim] = (cov, bias)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_sims", type=int, default=8000)
    p.add_argument("--n_cells", type=int, default=500)
    p.add_argument("--n_test", type=int, default=400)
    p.add_argument("--n_post", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    res = run("moments", args.n_sims, args.n_cells, args.n_test, args.n_post, args.seed)
    print("\n=== moments-only ablation (mean + variance features), k_on ===")
    for sim in SIMS:
        cov, bias = res[sim]
        print(f"  {sim.upper()}: 90% coverage = {cov:.2f}   bias = {bias:.2f}")
    print("\nCompare against the full-feature numbers in Table 1.")


if __name__ == "__main__":
    main()
