"""
run_score.py
------------
E4: score each NPE posterior against the gold-standard reference posterior with
C2ST and MMD, per empirical-Fano bin. This closes the metrics thread (W5): coverage
alone cannot tell an informative posterior from a near-prior one, but a distance to
the exact posterior can.

Requires that references already exist (run run_reference.py first) on the SAME
shared test set (same seed, n_test, n_cells). Trains / loads the four NPEs via
run_experiment.train (cached), queries each on the summaries of every reference
observation, and compares the two posterior sample clouds in log10 space.

Interpretation: the CME-trained NPE is NOT expected to hit C2ST = 0.5, because it
conditions on 6 summaries while the reference conditions on the full 500-cell
snapshot; its C2ST is the floor set by summary-statistic information loss plus NPE
approximation. The surrogate-minus-CME gap is the misspecification signal.

Writes results/scores.csv with, per seed x estimator x bin:
    n, C2ST (mean + bootstrap 95% CI), MMD (mean).

Needs torch + sbi (queries the flows). Run on the paper machine.

Usage:
    python run_score.py --seed 0 --n_test 400 --n_cells 500 --n_ref 400 --n_sims 8000
"""

import argparse
import csv
import os

import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

import run_experiment as R
import testset as TS
from metrics import c2st, mmd_rbf

SIMS = ["cme", "nb", "lna", "cle"]
N_BINS = 6


def boot_ci(vals, rng, n_boot=2000):
    if len(vals) < 4:
        return np.nan, np.nan
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    b = np.asarray(vals)[idx].mean(axis=1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_test", type=int, default=400)
    p.add_argument("--n_cells", type=int, default=500)
    p.add_argument("--n_ref", type=int, default=400)
    p.add_argument("--n_sims", type=int, default=8000)
    p.add_argument("--n_post", type=int, default=1000)
    p.add_argument("--outdir", type=str, default="results")
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    ts = TS.build_or_load(args.seed, args.n_test, args.n_cells)
    ref = np.load(os.path.join("cache", f"references_{args.seed}_{args.n_ref}.npz"))
    ref_samples = ref["ref_samples"]                    # (n_ref, n_samples, 3)
    emp_fano = ref["emp_fano"]
    n_ref = ref_samples.shape[0]
    summaries = torch.as_tensor(ts["summaries"][:n_ref], dtype=torch.float32)

    edges = np.quantile(emp_fano, np.linspace(0, 1, N_BINS + 1)); edges[-1] += 1e-9
    bin_idx = np.clip(np.digitize(emp_fano, edges) - 1, 0, N_BINS - 1)
    print("empirical-Fano sextile edges: " + "  ".join(f"{e:.2f}" for e in edges))
    print("n per bin: " + "  ".join(str(int((bin_idx == b).sum())) for b in range(N_BINS)))

    rng = np.random.default_rng(0)
    rows = []
    per_sim_c2 = {}
    for sim in SIMS:
        post = R.train(sim, args.n_sims, args.n_cells, args.seed)
        c2 = np.empty(n_ref); mm = np.empty(n_ref)
        for i in range(n_ref):
            s_npe = R._flow_sample(post, summaries[i], args.n_post).numpy()
            s_ref = ref_samples[i]
            k = min(len(s_npe), len(s_ref))
            if len(s_ref) > k:
                s_ref = s_ref[rng.choice(len(s_ref), k, replace=False)]
            if len(s_npe) > k:
                s_npe = s_npe[rng.choice(len(s_npe), k, replace=False)]
            c2[i] = c2st(s_npe, s_ref, seed=i)
            mm[i] = mmd_rbf(s_npe, s_ref)
        per_sim_c2[sim] = c2

        for b in range(N_BINS + 1):
            sel = np.ones(n_ref, bool) if b == N_BINS else (bin_idx == b)
            clo, chi = boot_ci(c2[sel], rng)
            rows.append(dict(
                seed=args.seed, sim=sim, bin=("all" if b == N_BINS else b),
                n=int(sel.sum()),
                c2st=round(float(c2[sel].mean()), 4),
                c2st_lo=round(clo, 4), c2st_hi=round(chi, 4),
                mmd=round(float(mm[sel].mean()), 6)))
        r_all = next(x for x in rows if x["sim"] == sim and x["bin"] == "all")
        print(f"  {sim.upper():3s}  overall C2ST = {r_all['c2st']:.3f} "
              f"[{r_all['c2st_lo']:.3f},{r_all['c2st_hi']:.3f}]   MMD = {r_all['mmd']:.4f}")

    # misspecification signal = surrogate C2ST minus the CME floor
    floor = float(np.mean(per_sim_c2["cme"]))
    print(f"\n  CME floor (summary loss + approx): C2ST = {floor:.3f}")
    for sim in ["nb", "lna", "cle"]:
        print(f"  {sim.upper()} misspecification gap over CME floor: "
              f"{np.mean(per_sim_c2[sim]) - floor:+.3f}")

    path = os.path.join(args.outdir, "scores.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n[written] {path}")


if __name__ == "__main__":
    main()
