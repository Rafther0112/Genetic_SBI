"""
run_hybrid_seeds.py
-------------------
Hybrid column of Tables 1 and 3 over the same five seeds as the other telegraph
estimators. Trains ONLY the discrete-promoter hybrid (the CME and CLE numbers already
come from run_reconcile.py), evaluates k_on coverage on one exact test set per seed,
binned in equal-count sextiles of the empirical Fano factor. Reuses simulate and
flow_sample from run_hybrid.py (run from the repo root).

Needs torch + sbi.   Usage:
    python run_hybrid_seeds.py --n_sims 8000 --n_cells 500 --n_test 2000 --seeds 0 1 2 3 4
"""
import argparse, os
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

from sbi.inference import NPE
from sbi_experiment import make_prior
import run_hybrid as H

N_BINS = 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_sims", type=int, default=8000)
    ap.add_argument("--n_cells", type=int, default=500)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--n_post", type=int, default=1000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    prior = make_prior()

    per_bin, overall, bias, edges_all = [], [], [], []
    for seed in a.seeds:
        torch.manual_seed(1000 + seed)
        tt = prior.sample((a.n_test,))
        xt = torch.as_tensor(H.simulate(tt.numpy(), "cme", a.n_cells, np.random.default_rng(1000 + seed)),
                             dtype=torch.float32)
        ttn = tt.numpy(); fano = xt[:, 2].numpy()
        edges = np.quantile(fano, np.linspace(0, 1, N_BINS + 1)); edges[-1] += 1e-9
        b = np.clip(np.digitize(fano, edges) - 1, 0, N_BINS - 1)

        torch.manual_seed(seed); rng = np.random.default_rng(seed)
        theta = prior.sample((a.n_sims,))
        x = torch.as_tensor(H.simulate(theta.numpy(), "hybrid", a.n_cells, rng), dtype=torch.float32)
        inf = NPE(prior=prior); inf.append_simulations(theta, x)
        post = inf.build_posterior(inf.train())

        inside = np.empty(a.n_test, bool); mean_kon = np.empty(a.n_test)
        for i in range(a.n_test):
            s = H.flow_sample(post, xt[i], a.n_post).numpy()
            u = (s[:, 0] < ttn[i, 0]).mean(); inside[i] = 0.05 <= u <= 0.95
            mean_kon[i] = s[:, 0].mean()
        pb = [inside[b == k].mean() for k in range(N_BINS)]
        per_bin.append(pb); overall.append(inside.mean()); edges_all.append(edges)
        bias.append(np.mean(mean_kon - ttn[:, 0]))
        np.savez(f"results/hybrid_seed{seed}.npz", inside=inside, bin_idx=b, edges=edges)
        print(f"[seed {seed}] hybrid per sextile " + " ".join(f"{v:.2f}" for v in pb)
              + f" | all {inside.mean():.2f} | signed k_on bias {bias[-1]:+.2f}", flush=True)

    P = np.array(per_bin); E = np.mean(edges_all, axis=0)
    print(f"\n=== hybrid, mean +/- std over seeds {a.seeds} ===")
    for k in range(N_BINS):
        print(f"  sextile {k} [{E[k]:.1f},{E[k+1]:.1f}]  ${P[:, k].mean():.2f}{{\\pm}}.{int(round(100*P[:, k].std())):02d}$")
    print(f"  overall coverage {np.mean(overall):.2f} +/- {np.std(overall):.2f}")
    print(f"  signed k_on bias {np.mean(bias):+.2f}")
    print("  (C2ST and k_off contraction for the hybrid need run_score.py / run_reconcile.py support)")


if __name__ == "__main__":
    main()
