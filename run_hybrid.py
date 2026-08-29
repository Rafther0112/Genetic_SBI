"""
run_hybrid.py
-------------
E2 (review W4): train an NPE on the hybrid simulator (discrete promoter + CLE mRNA)
and compare it against the full CLE and the CME control, evaluated on exact CME data,
per empirical-Fano bin. This tells whether the CLE failure comes from diffusing the
promoter state (fixed by the hybrid) or from the Gaussian mRNA noise (not fixed).

Needs torch + sbi (+ hybrid.py, numpy). Run on the paper machine.

Usage:
    python run_hybrid.py --n_sims 8000 --n_cells 500 --n_test 2000 --seed 0
"""

import argparse
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

from sbi.inference import NPE
from sbi_experiment import make_prior, theta_to_rates
from telegraph import summary_stats, sample_cle_batched
from hybrid import sample_hybrid_batched

SIMS = ["cme", "cle", "hybrid"]
N_BINS = 6


def simulate(theta_log, sim, n_cells, rng):
    rates = theta_to_rates(theta_log)
    N = rates.shape[0]
    kon = np.repeat(rates[:, 0], n_cells)
    koff = np.repeat(rates[:, 1], n_cells)
    ksyn = np.repeat(rates[:, 2], n_cells)
    if sim == "cme":
        counts = rng.poisson(ksyn * rng.beta(kon, koff))
    elif sim == "cle":
        counts = sample_cle_batched(kon, koff, ksyn, rng)
    elif sim == "hybrid":
        counts = sample_hybrid_batched(kon, koff, ksyn, rng)
    counts = counts.reshape(N, n_cells)
    return np.stack([summary_stats(counts[i]) for i in range(N)])


def flow_sample(post, x_row, n_post):
    s = post.posterior_estimator.sample((n_post,), condition=x_row.unsqueeze(0))
    return s.reshape(n_post, -1).detach()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_sims", type=int, default=8000)
    p.add_argument("--n_cells", type=int, default=500)
    p.add_argument("--n_test", type=int, default=2000)
    p.add_argument("--n_post", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    prior = make_prior()
    rng_t = np.random.default_rng(1000 + args.seed)
    tt = prior.sample((args.n_test,))
    xt = torch.as_tensor(simulate(tt.numpy(), "cme", args.n_cells, rng_t), dtype=torch.float32)
    ttn = tt.numpy()
    emp_fano = xt[:, 2].numpy()
    edges = np.quantile(emp_fano, np.linspace(0, 1, N_BINS + 1)); edges[-1] += 1e-9
    bin_idx = np.clip(np.digitize(emp_fano, edges) - 1, 0, N_BINS - 1)

    cov = {}
    for sim in SIMS:
        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed)
        theta = prior.sample((args.n_sims,))
        x = torch.as_tensor(simulate(theta.numpy(), sim, args.n_cells, rng), dtype=torch.float32)
        inf = NPE(prior=prior); inf.append_simulations(theta, x)
        post = inf.build_posterior(inf.train())
        ranks = np.empty((args.n_test, 3))
        for i in range(args.n_test):
            s = flow_sample(post, xt[i], args.n_post).numpy()
            ranks[i] = (s < ttn[i]).sum(0)
        u = ranks[:, 0] / args.n_post
        in_ci = (u >= 0.05) & (u <= 0.95)
        cov[sim] = in_ci

    print("\nempirical-Fano sextile edges: " + "  ".join(f"{e:.2f}" for e in edges))
    print("\nk_on 90% coverage per Fano bin:")
    print("bin   " + "".join(f"{s.upper():>10s}" for s in SIMS))
    for b in range(N_BINS):
        sel = bin_idx == b
        print(f"{b}     " + "".join(f"{cov[s][sel].mean():>10.2f}" for s in SIMS))
    print("all   " + "".join(f"{cov[s].mean():>10.2f}" for s in SIMS))
    print("\nIf hybrid tracks CME while CLE fails, the CLE failure is promoter diffusion,")
    print("not the Gaussian mRNA noise.")


if __name__ == "__main__":
    main()
