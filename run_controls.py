"""
run_controls.py
---------------
E1 (review W3): train an NPE on each LNA variant in the MOMENTS-ONLY setting and
evaluate on exact CME data. The variants differ only in how the Gaussian is mapped
to counts (see lna_variants.py). This isolates whether the moments-only failure of
the LNA comes from Gaussianity or from the non-negativity enforcement.

Reading:
  * CME control stays calibrated.
  * lna_unclamped / lna_rounded preserve the exact moments, so they should calibrate
    in moments-only -> the moments-only failure is the clamping, not Gaussianity.
    (They do so only by emitting physically impossible negative counts, which is why
    a Gaussian remains the wrong family: in the FULL-feature setting, where the zero
    fraction and tail enter, every Gaussian variant fails, as shown in the main
    experiment. That is Gaussianity proper.)
  * lna_truncated / lna_clamped distort the moments in the bursty regime -> they stay
    miscalibrated even here.

Needs torch + sbi. Run on the paper machine.

Usage:
    python run_controls.py --n_sims 8000 --n_cells 500 --n_test 2000 --seeds 0 1 2
"""

import argparse
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

from sbi.inference import NPE
from sbi_experiment import make_prior, theta_to_rates
from telegraph import summary_stats, sample_nb_batched
import lna_variants as V

SIMS = ["cme", "nb", "lna_clamped", "lna_truncated", "lna_rounded", "lna_unclamped"]


def simulate(theta_log, sim, n_cells, rng, features="moments"):
    rates = theta_to_rates(theta_log)
    N = rates.shape[0]
    kon = np.repeat(rates[:, 0], n_cells)
    koff = np.repeat(rates[:, 1], n_cells)
    ksyn = np.repeat(rates[:, 2], n_cells)
    if sim == "cme":
        counts = rng.poisson(ksyn * rng.beta(kon, koff)).astype(float)
    elif sim == "nb":
        counts = sample_nb_batched(kon, koff, ksyn, rng).astype(float)
    else:
        counts = np.asarray(V.VARIANTS[sim](kon, koff, ksyn, rng), dtype=float)
    counts = counts.reshape(N, n_cells)
    return np.stack([summary_stats(counts[i], features=features) for i in range(N)])


def flow_sample(post, x_row, n_post):
    s = post.posterior_estimator.sample((n_post,), condition=x_row.unsqueeze(0))
    return s.reshape(n_post, -1).detach()


def run_one(sim, n_sims, n_cells, n_test, n_post, seed):
    prior = make_prior()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    theta = prior.sample((n_sims,))
    x = torch.as_tensor(simulate(theta.numpy(), sim, n_cells, rng), dtype=torch.float32)
    inf = NPE(prior=prior)
    inf.append_simulations(theta, x)
    post = inf.build_posterior(inf.train())

    rng2 = np.random.default_rng(1000 + seed)
    tt = prior.sample((n_test,))
    xt = torch.as_tensor(simulate(tt.numpy(), "cme", n_cells, rng2), dtype=torch.float32)
    ttn = tt.numpy()
    ranks = np.empty((n_test, 3)); means = np.empty((n_test, 3))
    for i in range(n_test):
        s = flow_sample(post, xt[i], n_post).numpy()
        ranks[i] = (s < ttn[i]).sum(0)
        means[i] = s.mean(0)
    u = ranks[:, 0] / n_post
    cov = float(np.mean((u >= 0.05) & (u <= 0.95)))
    bias = float(np.abs(means[:, 0] - ttn[:, 0]).mean())
    return cov, bias


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_sims", type=int, default=8000)
    p.add_argument("--n_cells", type=int, default=500)
    p.add_argument("--n_test", type=int, default=2000)
    p.add_argument("--n_post", type=int, default=1000)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = p.parse_args()

    res = {s: [] for s in SIMS}
    for seed in args.seeds:
        for sim in SIMS:
            res[sim].append(run_one(sim, args.n_sims, args.n_cells,
                                    args.n_test, args.n_post, seed))

    print("\n=== E1: moments-only, k_on 90% coverage (mean +/- std over seeds) ===")
    print(f"{'simulator':16s} {'coverage':>16s} {'|bias|':>14s}")
    for sim in SIMS:
        c = np.array([r[0] for r in res[sim]]); b = np.array([r[1] for r in res[sim]])
        print(f"{sim:16s} {c.mean():.2f} +/- {c.std():.2f}     {b.mean():.2f} +/- {b.std():.2f}")
    print("\nExpected: unclamped/rounded calibrate (moments preserved) -> the moments-only")
    print("failure is clamping, not Gaussianity; truncated/clamped stay miscalibrated.")


if __name__ == "__main__":
    main()
