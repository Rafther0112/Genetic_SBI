"""
run_features.py
---------------
E6 (reviewer concern on hand-picked features): replace the six hand-crafted summaries
with the FULL count histogram passed through a small embedding network, and re-check
that the conclusions hold. If the ordering (CME/NB calibrated, LNA/CLE not) survives
under features not chosen to expose a Gaussian, the conclusions are not an artifact of
the summary choice.

Each 500-cell snapshot becomes a normalized histogram over counts 0..K-1 (clipped),
i.e. the empirical count distribution; a 2-layer MLP embeds it before the MAF.

Needs torch + sbi. The histogram featurizer (numpy) is validated separately; the
sbi/embedding driver is syntax-checked here (no torch in build env), so a minor API
tweak on your machine is possible.

Usage:
    python run_features.py --n_sims 8000 --n_cells 500 --n_test 2000 --K 200 --seed 0
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

from sbi.inference import NPE
from sbi.neural_nets import posterior_nn

from sbi_experiment import make_prior, theta_to_rates
from telegraph import exact_fano, sample_cle_batched, sample_lna_batched, sample_nb_batched

SIMS = ["cme", "nb", "lna", "cle"]
N_BINS = 6


def raw_counts(theta_log, sim, n_cells, rng):
    rates = theta_to_rates(theta_log)
    N = rates.shape[0]
    kon = np.repeat(rates[:, 0], n_cells); koff = np.repeat(rates[:, 1], n_cells)
    ksyn = np.repeat(rates[:, 2], n_cells)
    if sim == "cme":
        c = rng.poisson(ksyn * rng.beta(kon, koff))
    elif sim == "cle":
        c = sample_cle_batched(kon, koff, ksyn, rng)
    elif sim == "lna":
        c = sample_lna_batched(kon, koff, ksyn, rng)
    elif sim == "nb":
        c = sample_nb_batched(kon, koff, ksyn, rng)
    return c.reshape(N, n_cells)


def histogram(counts, K):
    """(N, n_cells) integer counts -> (N, K) normalized count histogram (clipped at K-1)."""
    N = counts.shape[0]
    H = np.empty((N, K), dtype=np.float32)
    for i in range(N):
        h = np.bincount(np.clip(counts[i], 0, K - 1), minlength=K)[:K]
        H[i] = h / h.sum()
    return H


class Embed(nn.Module):
    def __init__(self, K, out=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(K, 64), nn.ReLU(), nn.Linear(64, out), nn.ReLU())

    def forward(self, x):
        return self.net(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_sims", type=int, default=8000)
    p.add_argument("--n_cells", type=int, default=500)
    p.add_argument("--n_test", type=int, default=2000)
    p.add_argument("--n_post", type=int, default=1000)
    p.add_argument("--K", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    K = args.K

    prior = make_prior()
    rng_t = np.random.default_rng(1000 + args.seed)
    tt = prior.sample((args.n_test,)).numpy()
    test_counts = raw_counts(tt, "cme", args.n_cells, rng_t)
    x_test = torch.as_tensor(histogram(test_counts, K), dtype=torch.float32)
    emp_fano = np.array([exact_fano(*r) for r in theta_to_rates(tt)])
    edges = np.quantile(emp_fano, np.linspace(0, 1, N_BINS + 1)); edges[-1] += 1e-9
    bin_idx = np.clip(np.digitize(emp_fano, edges) - 1, 0, N_BINS - 1)

    cov = {}
    for sim in SIMS:
        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed)
        theta = prior.sample((args.n_sims,))
        x = torch.as_tensor(histogram(raw_counts(theta.numpy(), sim, args.n_cells, rng), K),
                            dtype=torch.float32)
        net = posterior_nn(model="maf", embedding_net=Embed(K))
        inf = NPE(prior=prior, density_estimator=net)
        inf.append_simulations(theta, x)
        post = inf.build_posterior(inf.train())
        ranks = np.empty((args.n_test, 3))
        for i in range(args.n_test):
            s = post.posterior_estimator.sample((args.n_post,), condition=x_test[i].unsqueeze(0))
            s = s.reshape(args.n_post, -1).detach().numpy()
            ranks[i] = (s < tt[i]).sum(0)
        u = ranks[:, 0] / args.n_post
        cov[sim] = (u >= 0.05) & (u <= 0.95)

    print("\nk_on 90% coverage, FULL-HISTOGRAM features (per Fano bin):")
    print("bin   " + "".join(f"{s.upper():>8s}" for s in SIMS))
    for b in range(N_BINS):
        sel = bin_idx == b
        print(f"{b}     " + "".join(f"{cov[s][sel].mean():>8.2f}" for s in SIMS))
    print("all   " + "".join(f"{cov[s].mean():>8.2f}" for s in SIMS))
    print("\nIf CME/NB stay calibrated and LNA/CLE stay low (as with the 6 summaries),")
    print("the conclusions do not depend on the hand-picked features.")


if __name__ == "__main__":
    main()
