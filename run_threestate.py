"""
run_threestate.py
-----------------
E10: repeat the misspecification diagnostic on the three-state refractory promoter,
where there is NO closed form, so the exact simulator (FSP/SSA) is genuinely expensive.
This exercises the cost trade-off the telegraph sandbox could not (Gate G0b) and shows
the diagnostic transfers to a system without an analytic Fano factor (we use the
empirical Fano of each snapshot throughout).

Estimators, evaluated on exact (FSP) test data, binned by empirical Fano:
    exact  : FSP-sampled (the expensive control)
    cle    : CLE diffusing all three promoter states (the straw-man)
    hybrid : discrete 3-state promoter + CLE mRNA (the count-appropriate surrogate)

Success: the same qualitative ordering (CLE fails, hybrid tracks exact) and a working
empirical rule where no closed form exists.

Cost note: the exact training set is 8000 FSP solves (~minutes); the CLE 3-state set is
the wall-clock bottleneck. Everything is cached in cache/three_*.npz after first build.

Needs torch + sbi + threestate.py. Usage:
    python run_threestate.py --n_sims 8000 --n_cells 500 --n_test 2000 --seed 0
"""

import argparse
import os
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

from sbi.inference import NPE
from sbi.utils import BoxUniform
from telegraph import summary_stats
import threestate as TS

CACHE = "cache"
os.makedirs(CACHE, exist_ok=True)
# log10 prior over (k01, k10, k12, k21, k_syn)
LOW = torch.tensor([-2.0, -1.0, -2.0, -1.0, 0.0])
HIGH = torch.tensor([1.3, 1.3, 1.3, 1.3, 2.3])
SIMS = ["exact", "cle", "hybrid", "nb"]
N_BINS = 6
PARAM = 4          # report coverage for k_syn


def prior():
    return BoxUniform(low=LOW, high=HIGH)


def simulate(theta_log, sim, n_cells, rng):
    rates = 10.0 ** np.asarray(theta_log, float)          # (N,5)
    N = rates.shape[0]
    if sim == "exact":
        X = np.empty((N, 6), np.float32)
        for i in range(N):
            mv, pmf = TS.fsp_stationary_pmf(rates[i])
            pmf = np.clip(pmf, 0, None); pmf /= pmf.sum()
            X[i] = summary_stats(rng.choice(mv, size=n_cells, p=pmf))
        return X
    if sim == "nb":
        X = np.empty((N, 6), np.float32)
        for i in range(N):
            X[i] = summary_stats(TS.sample_nb_3state(rates[i], n_cells, rng))
        return X
    cols = [np.repeat(rates[:, j], n_cells) for j in range(5)]
    fn = TS.sample_cle_3state_batched if sim == "cle" else TS.sample_hybrid_3state_batched
    counts = fn(*cols, rng).reshape(N, n_cells)
    return np.stack([summary_stats(counts[i]) for i in range(N)]).astype(np.float32)


def build(sim, n_sims, n_cells, seed):
    path = f"{CACHE}/three_{sim}_{n_sims}_{n_cells}_{seed}.npz"
    if os.path.exists(path):
        d = np.load(path); print(f"[cache] {path}")
        return torch.as_tensor(d["theta"]), torch.as_tensor(d["x"])
    print(f"[sim] {sim} ({n_sims}) ...")
    rng = np.random.default_rng(seed)
    theta = prior().sample((n_sims,))
    x = simulate(theta.numpy(), sim, n_cells, rng)
    np.savez(path, theta=theta.numpy(), x=x)
    return theta, torch.as_tensor(x)


def train(sim, n_sims, n_cells, seed):
    theta, x = build(sim, n_sims, n_cells, seed)
    inf = NPE(prior=prior()); inf.append_simulations(theta, x)
    return inf.build_posterior(inf.train())


def flow_sample(post, x_row, n_post):
    return post.posterior_estimator.sample((n_post,), condition=x_row.unsqueeze(0)).reshape(n_post, -1).detach()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_sims", type=int, default=8000)
    p.add_argument("--n_cells", type=int, default=500)
    p.add_argument("--n_test", type=int, default=2000)
    p.add_argument("--n_post", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    # exact test set
    rng = np.random.default_rng(1000 + args.seed)
    tt = prior().sample((args.n_test,)).numpy()
    x_test = torch.as_tensor(simulate(tt, "exact", args.n_cells, rng))
    fano = x_test[:, 2].numpy()
    edges = np.quantile(fano, np.linspace(0, 1, N_BINS + 1)); edges[-1] += 1e-9
    bin_idx = np.clip(np.digitize(fano, edges) - 1, 0, N_BINS - 1)

    cov = {}
    for sim in SIMS:
        post = train(sim, args.n_sims, args.n_cells, args.seed)
        ranks = np.empty((args.n_test, 5))
        for i in range(args.n_test):
            s = flow_sample(post, x_test[i], args.n_post).numpy()
            ranks[i] = (s < tt[i]).sum(0)
        u = ranks[:, PARAM] / args.n_post
        cov[sim] = (u >= 0.05) & (u <= 0.95)

    print("\nempirical-Fano sextile edges: " + "  ".join(f"{e:.2f}" for e in edges))
    print("\nk_syn 90% coverage per Fano bin (3-state model):")
    print("bin   " + "".join(f"{s:>9s}" for s in SIMS))
    for b in range(N_BINS):
        sel = bin_idx == b
        print(f"{b}     " + "".join(f"{cov[s][sel].mean():>9.2f}" for s in SIMS))
    print("all   " + "".join(f"{cov[s].mean():>9.2f}" for s in SIMS))
    print("\nExpected: exact calibrated, CLE fails (diffuses 3 discrete states),")
    print("hybrid tracks exact; NB (count family) transfers the count-shape mechanism.")

    # W3: coverage-vs-Fano figure for the second system
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    centers = np.sqrt(edges[:-1] * np.clip(edges[1:], 1e-6, None))
    col = {"exact": "#0f6e56", "cle": "#ba7517", "hybrid": "#2b6cb0", "nb": "#7e4ea8"}
    lab = {"exact": "exact (FSP, control)", "cle": "CLE (diffuses 3 states)",
           "hybrid": "hybrid (discrete promoter)", "nb": "NB (count surrogate)"}
    mk = {"exact": "o", "cle": "s", "hybrid": "^", "nb": "D"}
    plt.figure(figsize=(5, 3.6))
    for s in SIMS:
        y = [cov[s][bin_idx == b].mean() for b in range(N_BINS)]
        plt.plot(centers, y, mk[s] + "-", color=col[s], label=lab[s], lw=1.5)
    plt.axhline(0.9, ls=":", color="gray", label="nominal 90%")
    plt.xscale("log"); plt.ylim(0, 1)
    plt.xlabel("empirical Fano factor of snapshot")
    plt.ylabel("$k_{syn}$ 90% CI coverage")
    plt.title("Three-state refractory promoter (no closed form)")
    plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig("threestate_coverage.png", dpi=130, bbox_inches="tight")
    print("[plot] saved threestate_coverage.png")


if __name__ == "__main__":
    main()