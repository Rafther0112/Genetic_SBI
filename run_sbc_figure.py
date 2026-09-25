"""
run_sbc_figure.py
-----------------
Regenerates the SBC rank histograms of Appendix B (Figure "sbc_histograms") on the full
exact test set of 2,000 snapshots, one seed, for the four estimators (CME, NB, LNA, CLE),
and prints the "outer tenth" fractions the appendix quotes.

Training data: if your pipeline's cached training sets exist
(cache/train_{sim}_{n_sims}_{n_cells}_{seed}.npz, the run_experiment.py format), they are
reused so the estimators see exactly the paper's training pairs; otherwise the script
simulates new ones (the CLE takes about 7 minutes for 8,000 simulations).

Usage
  python run_sbc_figure.py --seed 0
  python run_sbc_figure.py --seed 0 --test_seed 999    # if the paper's test set used 999

Outputs sbc_histograms.pdf / .png, results_lastmile/sbc/ranks_seed{seed}.npz and the
fractions to paste into Appendix B.
"""

import argparse, os
import numpy as np
import common as C

SIMS = ["cme", "nb", "lna", "cle"]


def training_set(sim, n_sims, n_cells, seed, spec):
    path = f"cache/train_{sim}_{n_sims}_{n_cells}_{seed}.npz"
    if os.path.exists(path):
        d = np.load(path)
        print(f"[cache] reusing {path}", flush=True)
        return d["theta"], d["x"]
    rng = np.random.default_rng(seed)
    theta = C.sample_prior(n_sims, rng, spec)
    x = C.cached_sims(f"sbc{seed}_{n_sims}", theta, sim, n_cells, seed, "telegraph")
    return theta, x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--test_seed", type=int, default=None, help="default 1000 + seed")
    p.add_argument("--n_sims", type=int, default=8000)
    p.add_argument("--n_test", type=int, default=2000)
    p.add_argument("--n_cells", type=int, default=500)
    p.add_argument("--n_post", type=int, default=1000)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--out", default="sbc_histograms")
    a = p.parse_args()

    spec = C.TELEGRAPH
    test_seed = a.test_seed if a.test_seed is not None else 1000 + a.seed
    th_t, x_t = C.exact_test_set(spec, a.n_test, a.n_cells, test_seed, "telegraph")

    ranks = {}
    for sim in SIMS:
        th, x = training_set(sim, a.n_sims, a.n_cells, a.seed, spec)
        print(f"[train] {sim} on {len(th)} pairs", flush=True)
        est = C.train_npe(th, x, spec, seed=a.seed, threads=a.threads)
        s = C.sample_posterior(est, x_t, a.n_post)              # (n_test, n_post, 3)
        ranks[sim] = (s < th_t[:, None, :]).mean(1)             # normalized rank in [0, 1]
        cov = ((ranks[sim][:, 0] >= 0.05) & (ranks[sim][:, 0] <= 0.95)).mean()
        print(f"   k_on coverage {cov:.3f} (compare with Table 1)", flush=True)

    os.makedirs("results_lastmile/sbc", exist_ok=True)
    np.savez(f"results_lastmile/sbc/ranks_seed{a.seed}.npz", theta_test=th_t,
             **{s: r for s, r in ranks.items()})

    # outer tenth: rank below 0.05 or above 0.95 (a calibrated estimator gives 0.10)
    print("\nouter-tenth fraction (calibrated = 0.10), for Appendix B:")
    for j, name in enumerate(spec["names"]):
        vals = [((ranks[s][:, j] < 0.05) | (ranks[s][:, j] > 0.95)).mean() for s in SIMS]
        print(f"  {name:6s} " + "  ".join(f"{s.upper()}={v:.2f}" for s, v in zip(SIMS, vals)))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import binom
    nb = 10
    n = a.n_test
    lo, hi = binom.ppf(0.005, n, 1 / nb), binom.ppf(0.995, n, 1 / nb)   # 99% band per bin
    labels = {"cme": "CME", "nb": "NB", "lna": "LNA", "cle": "CLE"}
    fig, ax = plt.subplots(len(SIMS), 3, figsize=(7.0, 1.5 * len(SIMS)),
                           sharex=True, sharey=True)
    for r, sim in enumerate(SIMS):
        for c, name in enumerate(spec["names"]):
            A = ax[r, c]
            A.axhspan(lo, hi, color="0.85", zorder=0)
            A.hist(ranks[sim][:, c], bins=nb, range=(0, 1), color="#444", alpha=0.9)
            if r == 0:
                A.set_title(name.replace("k_", "$k_{\\mathrm{") + "}}$", fontsize=9)
            if c == 0:
                A.set_ylabel(labels[sim], fontsize=9)
            A.tick_params(labelsize=7)
    fig.supxlabel("normalized SBC rank", fontsize=9)
    fig.tight_layout()
    fig.savefig(a.out + ".pdf"); fig.savefig(a.out + ".png", dpi=200)
    print(f"[plot] saved {a.out}.pdf / .png  (band = binomial 99%, n = {n})")


if __name__ == "__main__":
    main()
