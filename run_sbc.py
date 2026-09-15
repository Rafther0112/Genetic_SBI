"""
run_sbc.py
----------
SBC rank histograms for the appendix (contribution C1 announces SBC; the draft has no
figure for it). Trains or loads the four telegraph estimators (cached training sets),
evaluates all of them on ONE shared exact test set, and writes sbc_histograms.png/.pdf.

Unlike run_experiment.py this does NOT write money_plot.png, so it cannot overwrite the
Figure 2 produced by make_fig2.py.

Needs torch + sbi.   Usage:
    python run_sbc.py --n_sims 8000 --n_cells 500 --n_test 2000 --seed 0
"""
import argparse
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

import run_experiment as R
from sbi_experiment import make_prior, simulate_batch

SIMS = ["cme", "nb", "lna", "cle"]
PARAMS = [r"$k_{on}$", r"$k_{off}$", r"$k_{syn}$"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_sims", type=int, default=8000)
    ap.add_argument("--n_cells", type=int, default=500)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--n_post", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bins", type=int, default=10)
    a = ap.parse_args()

    torch.manual_seed(1000 + a.seed)
    tt = make_prior().sample((a.n_test,)).numpy()
    x_test = torch.as_tensor(simulate_batch(tt, "cme", a.n_cells, np.random.default_rng(1000 + a.seed)),
                             dtype=torch.float32)

    ranks = {}
    for sim in SIMS:
        post = R.train(sim, a.n_sims, a.n_cells, a.seed)
        r = np.empty((a.n_test, 3))
        for i in range(a.n_test):
            s = R._flow_sample(post, x_test[i], a.n_post).numpy()
            r[i] = (s < tt[i]).mean(0)
        ranks[sim] = r
        print(f"[{sim}] done", flush=True)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    exp = a.n_test / a.bins
    band = 2.58 * np.sqrt(exp * (1 - 1.0 / a.bins))          # 99% band
    fig, ax = plt.subplots(len(SIMS), 3, figsize=(9, 2.1 * len(SIMS)), sharex=True, sharey=True)
    for i, sim in enumerate(SIMS):
        for j in range(3):
            A = ax[i, j]
            A.axhspan(exp - band, exp + band, color="0.85", zorder=0)
            A.hist(ranks[sim][:, j], bins=a.bins, range=(0, 1), color="#444", alpha=0.85)
            if i == 0: A.set_title(PARAMS[j])
            if j == 0: A.set_ylabel(sim.upper())
    fig.supxlabel("normalized SBC rank"); fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"sbc_histograms.{ext}", dpi=150, bbox_inches="tight")
    print("[plot] sbc_histograms.png/.pdf  (grey band = 99% interval for a uniform histogram)")

    print("\nfraction of ranks in the outer 10% of the unit interval (uniform = 0.10):")
    for sim in SIMS:
        u = ranks[sim]
        print(f"  {sim.upper():4s} " + "  ".join(
            f"{PARAMS[j]}={np.mean((u[:, j] < 0.05) | (u[:, j] > 0.95)):.2f}" for j in range(3)))


if __name__ == "__main__":
    main()
