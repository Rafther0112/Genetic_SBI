"""
run_budget3.py
--------------
E9 decisive test (Gate G2): does the Fano factor help ALLOCATE where to spend exact
simulation in the TRAINING set? At a fixed exact budget (fraction f of the training
set simulated exactly, the rest by the LNA surrogate), we compare two ways to choose
WHICH thetas get the exact treatment, using the SAME thetas and the SAME budget:

  * uniform  : a random fraction f of thetas simulated exactly
  * fano     : the fraction f of thetas with the HIGHEST analytic Fano simulated
               exactly (spend exact where the surrogate fails; cover the low-Fano rest
               with the surrogate, which is accurate there)

C4 survives (reformulated as a training-allocation rule) if fano-guided beats uniform
at equal exact budget. If not, C4 is dropped and the paper re-scopes to C1-C3 plus the
finding that mixing beats routing.

Needs torch + sbi. Usage:
    python run_budget3.py --n_sims 8000 --n_cells 500 --n_test 2000 --seeds 0 1 2
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


def cov_kon(post, x_test, tt, n_post):
    c = 0
    for i in range(len(tt)):
        s = R._flow_sample(post, x_test[i], n_post).numpy()
        r = (s[:, 0] < tt[i, 0]).mean()
        c += (0.05 <= r <= 0.95)
    return c / len(tt)


def train_alloc(theta, exact_mask, n_cells, rng):
    """Exact (CME) where exact_mask, LNA elsewhere; one NPE on the mixed set."""
    prior = make_prior()
    thn = theta.numpy()
    x = np.empty((len(thn), 6), dtype=np.float32)
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

    res = {(f, m): [] for f in fracs for m in ["uniform", "fano"]}
    for seed in args.seeds:
        torch.manual_seed(seed); rng = np.random.default_rng(seed)
        theta = prior.sample((args.n_sims,))
        fano = np.array([exact_fano(*r) for r in theta_to_rates(theta.numpy())])
        order = np.argsort(-fano)
        rr = np.random.default_rng(seed + 1)
        for f in fracs:
            k = int(round(f * args.n_sims))
            m_uni = np.zeros(args.n_sims, bool); m_uni[rr.choice(args.n_sims, k, replace=False)] = True
            m_fano = np.zeros(args.n_sims, bool); m_fano[order[:k]] = True
            res[(f, "uniform")].append(cov_kon(train_alloc(theta, m_uni, args.n_cells, rng), x_test, tt, args.n_post))
            res[(f, "fano")].append(cov_kon(train_alloc(theta, m_fano, args.n_cells, rng), x_test, tt, args.n_post))

    print(f"\nk_on 90% coverage, equal exact budget, mean over {len(args.seeds)} seeds")
    print(f"{'budget f':>10s}{'uniform':>10s}{'fano':>10s}{'gain':>8s}")
    for f in fracs:
        u = np.mean(res[(f, "uniform")]); v = np.mean(res[(f, "fano")])
        print(f"{f:>10.2f}{u:>10.2f}{v:>10.2f}{v-u:>+8.2f}")
    print("\nC4 survives if fano beats uniform at equal budget (positive gain).")


if __name__ == "__main__":
    main()
