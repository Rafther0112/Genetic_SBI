"""
run_budget2.py
--------------
E9 (review W2, Gate G2): budget-rule comparison against allocation baselines.

Given a budget = fraction f of the query population served by the exact CME-trained
estimator (the rest by the cheap LNA-trained one), we compare:
  * Fano-guided routing  : send the highest-empirical-Fano queries to the exact one
  * zero-guided routing   : send the highest zero-fraction queries to the exact one
  * random routing        : send a random fraction f to the exact one
  * uniform-mixture       : a SINGLE NPE trained on a training set that is fraction f
                            exact + (1-f) surrogate (mixing simulators, not routing)
Extremes f=0 (all surrogate) and f=1 (all exact) bound the curves.

Success (keeps C4): the Fano-guided curve dominates random and uniform-mixture at
every budget. Reported on k_on 90% coverage.

Not included here: RNPE (Ward 2022) and robust-statistics (Huang 2023) SBI baselines,
which require their reference implementations; they are cited as robust-SBI comparisons
and left for integration.

Needs torch + sbi. Reuses run_experiment.train (cached).

Usage:
    python run_budget2.py --n_sims 8000 --n_cells 500 --n_test 600 --seed 0
"""

import argparse
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

import run_experiment as R
from sbi_experiment import make_prior, simulate_batch
from sbi.inference import NPE


def per_obs_cov(post, x_test, tt, n_post):
    cov = np.empty(len(tt))
    for i in range(len(tt)):
        s = R._flow_sample(post, x_test[i], n_post).numpy()
        rank = (s[:, 0] < tt[i, 0]).mean()
        cov[i] = 1.0 if 0.05 <= rank <= 0.95 else 0.0
    return cov


def train_mixture(f, n_sims, n_cells, seed):
    """One NPE on a training set that is fraction f exact + (1-f) surrogate."""
    prior = make_prior()
    rng = np.random.default_rng(seed)
    n_e = int(round(f * n_sims)); n_c = n_sims - n_e
    thetas, xs = [], []
    if n_e:
        th = prior.sample((n_e,)); thetas.append(th)
        xs.append(torch.as_tensor(simulate_batch(th.numpy(), "cme", n_cells, rng), dtype=torch.float32))
    if n_c:
        th = prior.sample((n_c,)); thetas.append(th)
        xs.append(torch.as_tensor(simulate_batch(th.numpy(), "lna", n_cells, rng), dtype=torch.float32))
    theta = torch.cat(thetas); x = torch.cat(xs)
    inf = NPE(prior=prior); inf.append_simulations(theta, x)
    return inf.build_posterior(inf.train())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_sims", type=int, default=8000)
    p.add_argument("--n_cells", type=int, default=500)
    p.add_argument("--n_test", type=int, default=600)
    p.add_argument("--n_post", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    prior = make_prior()
    rng = np.random.default_rng(999)
    tt = prior.sample((args.n_test,)).numpy()
    x_test = torch.as_tensor(simulate_batch(tt, "cme", args.n_cells, rng), dtype=torch.float32)
    fano = x_test[:, 2].numpy(); zero = x_test[:, 3].numpy()

    post_e = R.train("cme", args.n_sims, args.n_cells, args.seed)
    post_c = R.train("lna", args.n_sims, args.n_cells, args.seed)
    cov_e = per_obs_cov(post_e, x_test, tt, args.n_post)
    cov_c = per_obs_cov(post_c, x_test, tt, args.n_post)

    N = args.n_test
    fracs = [0.0, 0.25, 0.5, 0.75, 1.0]
    order_f = np.argsort(-fano); order_z = np.argsort(-zero)
    rng2 = np.random.default_rng(0)

    def route(order, f):
        k = int(round(f * N)); sel = np.zeros(N, bool); sel[order[:k]] = True
        return np.mean(np.where(sel, cov_e, cov_c))

    def route_random(f):
        k = int(round(f * N)); accs = []
        for _ in range(30):
            s = np.zeros(N, bool); s[rng2.choice(N, k, replace=False)] = True
            accs.append(np.mean(np.where(s, cov_e, cov_c)))
        return np.mean(accs)

    print(f"\n{'budget f':>10s}{'Fano':>9s}{'zero':>9s}{'random':>9s}{'mixture':>9s}")
    for f in fracs:
        mix = per_obs_cov(train_mixture(f, args.n_sims, args.n_cells, args.seed),
                          x_test, tt, args.n_post).mean()
        print(f"{f:>10.2f}{route(order_f, f):>9.2f}{route(order_z, f):>9.2f}"
              f"{route_random(f):>9.2f}{mix:>9.2f}")
    print("\nC4 holds if Fano-guided dominates random and uniform-mixture at every budget.")


if __name__ == "__main__":
    main()
