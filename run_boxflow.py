"""
run_boxflow.py
--------------
E3 (review W6): train each estimator with the flow reparametrized onto the prior box
(boxflow.py), so posterior samples cannot leak outside it, and re-measure coverage.

We move theta to an unbounded space u = logit(scaled theta), train the NPE there with
the matching (logistic) prior, and map samples back with theta = to_box(u). Leakage is
then 0 by construction. The honest question E3 answers: does constraining the support
RESCUE the surrogates, or does the bias simply stay inside the box? If coverage does
not improve over the direct-sampling numbers (run_reconcile), the leakage was a symptom
of misspecification, not its cause.

Needs torch + sbi. NOTE: the numpy transforms in boxflow.py are validated; this
sbi/torch driver is syntax-checked but not run here (no torch in the build env), so a
minor API tweak on your machine is possible. If the logistic prior gives trouble,
swap PRIOR_U for the commented Normal(0, 1.81) approximation.

Usage:
    python run_boxflow.py --n_sims 8000 --n_cells 500 --n_test 2000 --seed 0
"""

import argparse
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

from torch.distributions import Uniform, Independent, TransformedDistribution
from torch.distributions.transforms import SigmoidTransform
from sbi.inference import NPE

from sbi_experiment import make_prior, theta_to_rates
from telegraph import summary_stats, sample_cle_batched, sample_lna_batched, sample_nb_batched
from boxflow import to_box_torch, to_unbounded, LOW, HIGH

SIMS = ["cme", "nb", "lna", "cle"]
N_BINS = 6
LOW_T = torch.tensor(LOW, dtype=torch.float32)
HIGH_T = torch.tensor(HIGH, dtype=torch.float32)


def logistic_prior():
    # u ~ Logistic(0,1) per dim  <=>  to_box(u) ~ Uniform(box). Exact reparametrization.
    base = Uniform(torch.zeros(3), torch.ones(3))
    return Independent(TransformedDistribution(base, [SigmoidTransform().inv]), 1)
    # Fallback if the above misbehaves in your sbi version:
    # from torch.distributions import Normal
    # return Independent(Normal(torch.zeros(3), 1.81 * torch.ones(3)), 1)


PRIOR_U = logistic_prior()


def sim_counts(rates_rep, sim, rng):
    kon, koff, ksyn = rates_rep
    if sim == "cme":
        return rng.poisson(ksyn * rng.beta(kon, koff))
    if sim == "cle":
        return sample_cle_batched(kon, koff, ksyn, rng)
    if sim == "lna":
        return sample_lna_batched(kon, koff, ksyn, rng)
    if sim == "nb":
        return sample_nb_batched(kon, koff, ksyn, rng)


def simulate_u(u, sim, n_cells, rng):
    """u: (N,3) unbounded -> theta in box -> counts -> summaries (N,6)."""
    theta = to_box_torch(u, LOW_T, HIGH_T).numpy()
    rates = theta_to_rates(theta)
    N = rates.shape[0]
    rep = (np.repeat(rates[:, 0], n_cells), np.repeat(rates[:, 1], n_cells),
           np.repeat(rates[:, 2], n_cells))
    counts = sim_counts(rep, sim, rng).reshape(N, n_cells)
    return np.stack([summary_stats(counts[i]) for i in range(N)])


def flow_sample_u(post, x_row, n_post):
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

    # shared exact-CME test set (theta drawn from the ORIGINAL uniform box prior)
    prior_box = make_prior()
    rng_t = np.random.default_rng(1000 + args.seed)
    tt = prior_box.sample((args.n_test,)).numpy()                 # theta in box (truth)
    rates = theta_to_rates(tt)
    rep = (np.repeat(rates[:, 0], args.n_cells), np.repeat(rates[:, 1], args.n_cells),
           np.repeat(rates[:, 2], args.n_cells))
    x_test = np.stack([summary_stats(c) for c in
                       rng_t.poisson(rep[2] * rng_t.beta(rep[0], rep[1])).reshape(args.n_test, args.n_cells)])
    x_test = torch.as_tensor(x_test, dtype=torch.float32)
    emp_fano = x_test[:, 2].numpy()
    edges = np.quantile(emp_fano, np.linspace(0, 1, N_BINS + 1)); edges[-1] += 1e-9
    bin_idx = np.clip(np.digitize(emp_fano, edges) - 1, 0, N_BINS - 1)

    cov = {}; leaked = {}
    for sim in SIMS:
        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed)
        u = PRIOR_U.sample((args.n_sims,))                        # train in u-space
        x = torch.as_tensor(simulate_u(u, sim, args.n_cells, rng), dtype=torch.float32)
        inf = NPE(prior=PRIOR_U)
        inf.append_simulations(u, x)
        post = inf.build_posterior(inf.train())

        ranks = np.empty((args.n_test, 3)); leak = np.empty(args.n_test)
        for i in range(args.n_test):
            us = flow_sample_u(post, x_test[i], args.n_post)
            th = to_box_torch(us, LOW_T, HIGH_T).numpy()          # back to box
            ranks[i] = (th < tt[i]).sum(0)
            leak[i] = np.mean(np.any((th < LOW) | (th > HIGH), axis=1))
        u_r = ranks[:, 0] / args.n_post
        cov[sim] = (u_r >= 0.05) & (u_r <= 0.95)
        leaked[sim] = leak.mean()

    print("\nempirical-Fano sextile edges: " + "  ".join(f"{e:.2f}" for e in edges))
    print("\nk_on 90% coverage per Fano bin (box-constrained flow):")
    print("bin   " + "".join(f"{s.upper():>8s}" for s in SIMS))
    for b in range(N_BINS):
        sel = bin_idx == b
        print(f"{b}     " + "".join(f"{cov[s][sel].mean():>8.2f}" for s in SIMS))
    print("all   " + "".join(f"{cov[s].mean():>8.2f}" for s in SIMS))
    print("\nleaked mass (should be ~0 for all): " +
          "  ".join(f"{s}={leaked[s]:.3f}" for s in SIMS))
    print("Compare coverage against run_reconcile (direct sampling). If the surrogates")
    print("stay low, constraining the support does not rescue them: leakage was a")
    print("symptom of misspecification, not its cause (answers W6/Q3).")


if __name__ == "__main__":
    main()
