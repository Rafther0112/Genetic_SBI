"""
run_realdata.py
---------------
E11 (stretch): apply the diagnostic to real single-cell data with a capture layer.

There is no ground-truth theta in real data, so we do NOT claim parameter recovery.
The honest, checkable claims are:
  (a) the CME-trained and surrogate-trained posteriors DISAGREE on some genes, and
  (b) the empirical Fano factor of a gene PREDICTS that disagreement.

Both estimators are trained once, amortized, with the capture layer baked in
(simulator -> Binomial(., p) capture -> summaries), so they target the observed-count
posterior. Then, per gene, we compare the two posteriors and correlate the
disagreement with the gene's observed Fano.

Data format: a counts matrix (genes x cells), integer, from .npz (key 'counts') or
.csv. Each row is one gene (or allele) = one snapshot of per-cell mRNA counts. For
Larsson 2019 (allele-resolved) or an smFISH set, export the count matrix to that shape.
Capture efficiency p should ideally be estimated from spike-ins; here it is an argument.

Needs torch + sbi + capture.py + metrics.py. Usage:
    python run_realdata.py --data counts.npz --capture 0.15 --n_sims 8000 --seed 0
"""

import argparse
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

from sbi.inference import NPE
from sbi_experiment import make_prior, theta_to_rates
from telegraph import summary_stats, sample_lna_batched
import capture as C
from metrics import mmd_rbf


def load_counts(path):
    if path.endswith(".npz"):
        m = np.load(path)["counts"]
    elif path.endswith(".csv"):
        m = np.loadtxt(path, delimiter=",")
    else:
        raise ValueError("use .npz (key 'counts') or .csv, genes x cells")
    m = np.rint(np.asarray(m)).astype(int)
    keep = (m.mean(1) > 0.05) & (m.shape[1] - (m == 0).sum(1) >= 20)   # expressed genes
    return m[keep]


def simulate_captured(theta_log, sim, n_cells, p, rng):
    rates = theta_to_rates(theta_log); N = rates.shape[0]
    kon = np.repeat(rates[:, 0], n_cells); koff = np.repeat(rates[:, 1], n_cells)
    ksyn = np.repeat(rates[:, 2], n_cells)
    if sim == "cme":
        counts = rng.poisson(ksyn * rng.beta(kon, koff))
    else:
        counts = sample_lna_batched(kon, koff, ksyn, rng)
    counts = C.apply_capture(counts, p, rng).reshape(N, n_cells)      # capture layer
    return np.stack([summary_stats(counts[i]) for i in range(N)]).astype(np.float32)


def train(sim, n_sims, n_cells, p, seed):
    prior = make_prior()
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    theta = prior.sample((n_sims,))
    x = torch.as_tensor(simulate_captured(theta.numpy(), sim, n_cells, p, rng))
    inf = NPE(prior=prior); inf.append_simulations(theta, x)
    return inf.build_posterior(inf.train())


def flow_sample(post, x_row, n_post):
    return post.posterior_estimator.sample((n_post,), condition=x_row.unsqueeze(0)).reshape(n_post, -1).detach().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--capture", type=float, default=0.15)
    p.add_argument("--n_sims", type=int, default=8000)
    p.add_argument("--n_post", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    counts = load_counts(args.data)
    n_genes, n_cells = counts.shape
    print(f"[data] {n_genes} expressed genes x {n_cells} cells  (capture p={args.capture})")

    post_e = train("cme", args.n_sims, n_cells, args.capture, args.seed)
    post_c = train("lna", args.n_sims, n_cells, args.capture, args.seed)

    fano = np.empty(n_genes); disc = np.empty(n_genes)
    for g in range(n_genes):
        x = torch.as_tensor(summary_stats(counts[g]), dtype=torch.float32)
        fano[g] = x[2].item()
        se = flow_sample(post_e, x, args.n_post); sc = flow_sample(post_c, x, args.n_post)
        disc[g] = mmd_rbf(se, sc)                    # posterior disagreement
        disc[g] = max(disc[g], 0.0)

    # does the Fano predict the disagreement?
    from scipy.stats import spearmanr
    rho, pval = spearmanr(fano, disc)
    print(f"\nSpearman corr(observed Fano, exact-vs-surrogate disagreement) = {rho:.3f} (p={pval:.1e})")
    hi = disc > np.median(disc)
    print(f"median observed Fano: disagree>median genes = {np.median(fano[hi]):.2f}, "
          f"others = {np.median(fano[~hi]):.2f}")
    order = np.argsort(-disc)[:10]
    print("\ntop-10 most disagreeing genes (row idx, Fano, disagreement):")
    for g in order:
        print(f"  gene {g:5d}   Fano={fano[g]:6.2f}   MMD={disc[g]:.3f}")
    print("\nClaim (C5): high-Fano genes are where exact- and surrogate-trained posteriors")
    print("disagree, so the observable Fano flags where a surrogate cannot be trusted on")
    print("real data. No ground-truth theta is used or claimed.")


if __name__ == "__main__":
    main()
