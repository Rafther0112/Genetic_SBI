"""
run_oracle.py
-------------
W2 upper bound (reviewer): does an ORACLE allocation, one that places exact simulation
where the surrogate demonstrably fails most, using knowledge a practitioner does not
have, also fail to beat a uniform mixture? If even the oracle cannot beat uniform, the
negative result is structural ("local allocation is futile for an amortized estimator"),
not merely "the Fano is the wrong signal."

The oracle value of a training point theta is the surrogate estimator's actual error
there: train a surrogate (LNA) NPE, and for each training theta measure
||posterior_mean_LNA(x) - theta|| on its exact observation x. This uses the true theta,
which is available only at training time in a controlled study, so it is a legitimate
upper bound. Schemes at equal exact budget f:

  uniform       : random f (baseline)
  oracle_hard   : exact on the top-f highest-surrogate-error theta
  oracle_soft   : soft oversampling by surrogate error, keeping low-error mass nonzero

If oracle_hard/soft do not beat uniform, allocation is structurally futile.

Needs torch + sbi. Usage:
    python run_oracle.py --n_sims 8000 --n_cells 500 --n_test 2000 --seeds 0 1 2
"""

import argparse
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

from sbi.inference import NPE
from sbi_experiment import make_prior, simulate_batch
import run_experiment as R

SCHEMES = ["uniform", "oracle_hard", "oracle_soft"]


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


def oracle_error(theta, n_cells, seed, n_post=500):
    """Surrogate (LNA) posterior-mean error at each training theta on its exact obs."""
    post = R.train("lna", len(theta), n_cells, seed)      # cached
    rng = np.random.default_rng(seed + 7)
    thn = theta.numpy()
    x = torch.as_tensor(simulate_batch(thn, "cme", n_cells, rng), dtype=torch.float32)
    err = np.empty(len(thn))
    for i in range(len(thn)):
        m = R._flow_sample(post, x[i], n_post).numpy().mean(0)
        err[i] = np.linalg.norm(m - thn[i])
    return err


def select(scheme, err, k, rng):
    N = len(err)
    if scheme == "uniform":
        return rng.choice(N, k, replace=False)
    order = np.argsort(-err)                              # highest surrogate error first
    if scheme == "oracle_hard":
        return order[:k]
    r = np.empty(N); r[order] = np.linspace(1.0, 0.0, N)
    w = 1.0 + 4.0 * r; w = w / w.sum()                    # soft oversampling by error
    return rng.choice(N, k, replace=False, p=w)


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
        err = oracle_error(theta, args.n_cells, seed)     # oracle knowledge
        rr = np.random.default_rng(seed + 1)
        for f in fracs:
            k = int(round(f * args.n_sims))
            for sc in SCHEMES:
                m = np.zeros(args.n_sims, bool); m[select(sc, err, k, rr)] = True
                res[(f, sc)].append(cov_kon(train_alloc(theta, m, args.n_cells, rng),
                                            x_test, tt, args.n_post))

    print(f"\nk_on 90% coverage at equal exact budget, mean +/- std over {len(args.seeds)} seeds")
    print(f"{'f':>6s}" + "".join(f"{s:>16s}" for s in SCHEMES))
    for f in fracs:
        print(f"{f:>6.2f}" + "".join(
            f"{np.mean(res[(f,s)]):>10.2f}+-{np.std(res[(f,s)]):<4.2f}" for s in SCHEMES))
    print("\nIf oracle_hard/soft do not exceed uniform beyond the seed std, allocation is")
    print("structurally futile: even knowing exactly where the surrogate fails does not help.")


if __name__ == "__main__":
    main()
