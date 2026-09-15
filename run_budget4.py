"""
run_budget4.py
--------------
W2 (reviewer): does the negative allocation result survive a FAMILY of Fano-informed
schemes, not just the most aggressive one? At an equal exact budget (fraction f of the
8000 training points simulated exactly, rest by the LNA surrogate), we compare how the
exact fraction is CHOSEN among the same parameters:

  uniform      : f chosen uniformly at random (baseline)
  fano_hard    : top-f by analytic Fano (concentrate all exact on high Fano; starves low Fano)
  fano_soft(a) : sample f WITHOUT replacement with weight 1 + a * fano_rank, so high-Fano
                 points are oversampled but low-Fano points keep nonzero exact probability
                 (a=2 mild, a=6 strong). This is the reviewer's requested scheme that does
                 not zero out low-Fano exact training.
  inverse      : top-f by LOWEST Fano (control; should be worst if Fano matters positively)

If every Fano-informed scheme (hard and soft) loses to uniform, the negative result is
robust: no way of informing allocation with the Fano factor beats a uniform mixture. If a
soft scheme ties or wins, we report that and soften the claim. Either outcome answers W2.

Needs torch + sbi. Usage:
    python run_budget4.py --n_sims 8000 --n_cells 500 --n_test 2000 --seeds 0 1 2
"""

import argparse
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

from sbi.inference import NPE
from sbi_experiment import make_prior, simulate_batch, theta_to_rates
from telegraph import exact_fano
import run_experiment as R

SCHEMES = ["uniform", "fano_hard", "fano_soft2", "fano_soft6", "inverse"]


def select(scheme, fano, k, rng):
    N = len(fano)
    if scheme == "uniform":
        return rng.choice(N, k, replace=False)
    order = np.argsort(-fano)                       # high Fano first
    if scheme == "fano_hard":
        return order[:k]
    if scheme == "inverse":
        return order[::-1][:k]
    a = 2.0 if scheme == "fano_soft2" else 6.0
    r = np.empty(N); r[order] = np.linspace(1.0, 0.0, N)   # rank in [0,1], high Fano -> 1
    w = 1.0 + a * r; w = w / w.sum()
    return rng.choice(N, k, replace=False, p=w)


def cov_kon(post, x_test, tt, n_post):
    c = 0
    for i in range(len(tt)):
        s = R._flow_sample(post, x_test[i], n_post).numpy()
        r = (s[:, 0] < tt[i, 0]).mean()
        c += (0.05 <= r <= 0.95)
    return c / len(tt)


def train_alloc(theta, exact_mask, n_cells, rng):
    prior = make_prior(); thn = theta.numpy()
    x = np.empty((len(thn), 6), np.float32)
    ie = np.where(exact_mask)[0]; ic = np.where(~exact_mask)[0]
    if len(ie): x[ie] = simulate_batch(thn[ie], "cme", n_cells, rng)
    if len(ic): x[ic] = simulate_batch(thn[ic], "lna", n_cells, rng)
    inf = NPE(prior=prior); inf.append_simulations(theta, torch.as_tensor(x))
    return inf.build_posterior(inf.train())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_sims", type=int, default=8000)
    p.add_argument("--n_cells", type=int, default=500)
    p.add_argument("--n_test", type=int, default=2000)
    p.add_argument("--n_post", type=int, default=1000)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = p.parse_args()
    fracs = [0.25, 0.5, 0.75]

    prior = make_prior()
    rng_t = np.random.default_rng(999)
    tt = prior.sample((args.n_test,)).numpy()
    x_test = torch.as_tensor(simulate_batch(tt, "cme", args.n_cells, rng_t), dtype=torch.float32)

    res = {(f, s): [] for f in fracs for s in SCHEMES}
    for seed in args.seeds:
        torch.manual_seed(seed); rng = np.random.default_rng(seed)
        theta = prior.sample((args.n_sims,))
        fano = np.array([exact_fano(*r) for r in theta_to_rates(theta.numpy())])
        rr = np.random.default_rng(seed + 1)
        for f in fracs:
            k = int(round(f * args.n_sims))
            for sc in SCHEMES:
                m = np.zeros(args.n_sims, bool); m[select(sc, fano, k, rr)] = True
                res[(f, sc)].append(cov_kon(train_alloc(theta, m, args.n_cells, rng),
                                            x_test, tt, args.n_post))

    print(f"\nk_on 90% coverage at equal exact budget, mean +/- std over {len(args.seeds)} seeds")
    print(f"{'f':>6s}" + "".join(f"{s:>16s}" for s in SCHEMES))
    for f in fracs:
        print(f"{f:>6.2f}" + "".join(
            f"{np.mean(res[(f,s)]):>10.2f}+-{np.std(res[(f,s)]):<4.2f}" for s in SCHEMES))
    print("\nRead: is any fano_soft cell above uniform by MORE than the seed std? If not,")
    print("the honest statement is 'no reliable advantage over uniform mixing'.")


if __name__ == "__main__":
    main()