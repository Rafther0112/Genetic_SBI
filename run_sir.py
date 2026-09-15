"""
run_sir.py
----------
W1: does the mechanism transfer OUTSIDE gene expression? Train an NPE to infer the
basic reproduction number R0 of a stochastic SIR epidemic from the distribution of
final outbreak sizes, using the exact SSA vs the CLE surrogate, and check whether the
CLE-trained estimator loses calibration where the empirical Fano is high, as in the
gene-expression systems.

Single parameter theta = log10 R0 (gamma = 1, N = 200, I0 = 1). Each observation is a
snapshot of n_real final sizes; six summaries as elsewhere; empirical Fano is the
diagnostic. This is a demonstration of transfer, so the scale is intentionally modest.

Needs torch + sbi + sir.py. The SSA is the bottleneck (a few minutes).

Usage:
    python run_sir.py --n_sims 3000 --n_real 200 --n_test 1000 --seed 0
"""

import argparse
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

from sbi.inference import NPE
from sbi.utils import BoxUniform
from telegraph import summary_stats
import sir as SIR

N_POP, I0, GAMMA = 200, 1, 1.0
LOW = torch.tensor([np.log10(0.5)]); HIGH = torch.tensor([np.log10(4.0)])   # R0 in [0.5,4]
SIMS = ["ssa", "cle"]
N_BINS = 5


def prior():
    return BoxUniform(low=LOW, high=HIGH)


def simulate(theta_log, sim, n_real, rng):
    R0 = 10.0 ** np.asarray(theta_log, float).ravel()
    X = np.empty((len(R0), 6), np.float32)
    for i, r0 in enumerate(R0):
        fn = SIR.sample_sir_ssa if sim == "ssa" else SIR.sample_sir_cle
        fs = fn(r0 * GAMMA, GAMMA, N_POP, I0, n_real, rng)
        X[i] = summary_stats(fs)
    return X


def train(sim, n_sims, n_real, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    theta = prior().sample((n_sims,))
    x = torch.as_tensor(simulate(theta.numpy(), sim, n_real, rng))
    inf = NPE(prior=prior()); inf.append_simulations(theta, x)
    return inf.build_posterior(inf.train())


def flow_sample(post, x_row, n_post):
    return post.posterior_estimator.sample((n_post,), condition=x_row.unsqueeze(0)).reshape(n_post, -1).detach()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_sims", type=int, default=3000)
    p.add_argument("--n_real", type=int, default=200)
    p.add_argument("--n_test", type=int, default=1000)
    p.add_argument("--n_post", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(1000 + args.seed)
    tt = prior().sample((args.n_test,)).numpy()
    x_test = torch.as_tensor(simulate(tt, "ssa", args.n_real, rng))
    fano = x_test[:, 2].numpy()
    edges = np.quantile(fano, np.linspace(0, 1, N_BINS + 1)); edges[-1] += 1e-9
    bin_idx = np.clip(np.digitize(fano, edges) - 1, 0, N_BINS - 1)

    cov = {}
    for sim in SIMS:
        post = train(sim, args.n_sims, args.n_real, args.seed)
        ranks = np.empty(args.n_test)
        for i in range(args.n_test):
            s = flow_sample(post, x_test[i], args.n_post).numpy()
            ranks[i] = (s[:, 0] < tt[i, 0]).mean()
        cov[sim] = (ranks >= 0.05) & (ranks <= 0.95)

    print("\nempirical-Fano bin edges: " + "  ".join(f"{e:.1f}" for e in edges))
    print("\nR0 90% coverage per Fano bin (stochastic SIR, outside gene expression):")
    print("bin   " + "".join(f"{s:>8s}" for s in SIMS))
    for b in range(N_BINS):
        sel = bin_idx == b
        print(f"{b}     " + "".join(f"{cov[s][sel].mean():>8.2f}" for s in SIMS))
    print("all   " + "".join(f"{cov[s].mean():>8.2f}" for s in SIMS))
    print("\nIf SSA stays calibrated and CLE drops where Fano is high, the mechanism")
    print("transfers outside gene expression (Gaussian/continuum surrogate for a")
    print("discrete low-copy process). If CLE stays close to SSA, the effect is weak")
    print("here and we narrow the generality language instead.")


if __name__ == "__main__":
    main()
