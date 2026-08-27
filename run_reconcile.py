"""
run_reconcile.py
----------------
P0-2 / Gate G0(a): reconcile the text-vs-figure conflict flagged in review W1/Q1.

Trains the four NPEs (reusing the cached training sets from run_experiment.py) and
evaluates all of them on a SHARED exact-CME test set, then emits a machine-readable
per-bin table:

  for each  seed x estimator x parameter x empirical-Fano bin:
      n, 90% CI coverage, bootstrap 95% CI on coverage, MAE, SIGNED bias,
      posterior contraction (posterior std / prior std)

The signed bias supports the "directional" narrative (W5); the contraction shows
whether an estimator is actually informative or just returns a near-prior posterior
(the k_off identifiability point in W5). A second CSV reports the leaked-mass
fraction per estimator and bin (W6 / Q3): the share of posterior samples that fall
outside the prior box on exact-CME data.

Binning rule: equal-count sextiles of the EMPIRICAL Fano factor (the observable the
budget rule uses), the same rule as the money plot. Bin edges and per-bin n are
printed and written to the CSV, addressing the "state the binning" ask in W7.

Answers Q1 directly: is the LNA k_on coverage flat across Fano (figure) or a
monotone 0.75 -> 0.27 decline (old text)? Read the LNA rows of the printed table.

Usage:
    python run_reconcile.py --n_sims 8000 --n_cells 500 --n_test 2000 --seeds 0
    python run_reconcile.py ... --seeds 0 1 2 3 4      # per-seed, for the full Q1 answer

Needs torch + sbi (run on the paper machine). Uses run_experiment.train, which
loads cached training sets when present, so re-runs are cheap.
"""

import argparse
import csv
import os

import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

import run_experiment as R
from sbi_experiment import make_prior, simulate_batch, LOW, HIGH
from telegraph import exact_fano

PARAM_NAMES = ["k_on", "k_off", "k_syn"]
SIMS = ["cme", "nb", "lna", "cle"]
N_BINS = 6

LOW_NP = np.asarray(LOW, dtype=np.float64)
HIGH_NP = np.asarray(HIGH, dtype=np.float64)
PRIOR_STD = (HIGH_NP - LOW_NP) / np.sqrt(12.0)   # std of a uniform on [low, high]


def eval_full(posterior, tt, x_test, n_post):
    """Per test point: posterior mean, std, SBC rank, and leaked-mass fraction."""
    N = len(tt)
    means = np.empty((N, 3))
    stds = np.empty((N, 3))
    ranks = np.empty((N, 3))
    leaked = np.empty(N)
    for i in range(N):
        s = R._flow_sample(posterior, x_test[i], n_post).numpy()
        means[i] = s.mean(0)
        stds[i] = s.std(0)
        ranks[i] = (s < tt[i]).sum(0)
        outside = np.any((s < LOW_NP) | (s > HIGH_NP), axis=1)
        leaked[i] = outside.mean()
    return means, stds, ranks, leaked


def boot_cov_ci(in_ci, rng, n_boot=2000):
    """Bootstrap 95% interval for a coverage (mean of a 0/1 array)."""
    n = len(in_ci)
    if n < 4:
        return np.nan, np.nan
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = in_ci[idx].mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def sextile_bins(fano):
    edges = np.quantile(fano, np.linspace(0, 1, N_BINS + 1))
    edges[-1] += 1e-9
    idx = np.clip(np.digitize(fano, edges) - 1, 0, N_BINS - 1)
    return edges, idx


def run_seed(seed, n_sims, n_cells, n_test, n_post):
    # shared exact-CME test set for this seed
    rng = np.random.default_rng(1000 + seed)
    prior = make_prior()
    theta_test = prior.sample((n_test,))
    tt = theta_test.numpy()
    x_test = torch.as_tensor(simulate_batch(tt, "cme", n_cells, rng), dtype=torch.float32)
    emp_fano = x_test[:, 2].numpy()
    edges, bin_idx = sextile_bins(emp_fano)

    rows, leak_rows = [], []
    boot_rng = np.random.default_rng(7)
    print(f"\n================ seed {seed} ================")
    print("empirical-Fano sextile edges: " +
          "  ".join(f"{e:.2f}" for e in edges))
    print("n per bin: " + "  ".join(str(int((bin_idx == b).sum())) for b in range(N_BINS)))

    ev = {}
    for sim in SIMS:
        post = R.train(sim, n_sims, n_cells, seed)
        ev[sim] = eval_full(post, tt, x_test, n_post)

    for sim in SIMS:
        means, stds, ranks, leaked = ev[sim]
        u = ranks / n_post
        in_ci = ((u >= 0.05) & (u <= 0.95)).astype(float)   # (N,3)
        err = means - tt                                    # signed, (N,3)

        for b in range(N_BINS + 1):                         # last iter = "all"
            sel = np.ones(len(tt), bool) if b == N_BINS else (bin_idx == b)
            lo = float(emp_fano[sel].min()); hi = float(emp_fano[sel].max())
            bname = "all" if b == N_BINS else b
            for p in range(3):
                cov = float(in_ci[sel, p].mean())
                clo, chi = boot_cov_ci(in_ci[sel, p], boot_rng)
                rows.append(dict(
                    seed=seed, sim=sim, param=PARAM_NAMES[p], bin=bname,
                    fano_lo=round(lo, 4), fano_hi=round(hi, 4), n=int(sel.sum()),
                    coverage=round(cov, 4), cov_lo=round(clo, 4), cov_hi=round(chi, 4),
                    mae=round(float(np.abs(err[sel, p]).mean()), 4),
                    signed_bias=round(float(err[sel, p].mean()), 4),
                    contraction=round(float(stds[sel, p].mean() / PRIOR_STD[p]), 4)))
            leak_rows.append(dict(
                seed=seed, sim=sim, bin=bname,
                fano_lo=round(lo, 4), fano_hi=round(hi, 4),
                n=int(sel.sum()), leaked_frac=round(float(leaked[sel].mean()), 4)))

    # focused read-out for the W1 question: k_on coverage per bin, all estimators
    print("\n  k_on 90% coverage per empirical-Fano bin (the W1/Q1 question):")
    print("  " + "bin".ljust(6) + "".join(f"{s.upper():>16s}" for s in SIMS))
    for b in range(N_BINS):
        cells = []
        for sim in SIMS:
            r = next(x for x in rows if x["seed"] == seed and x["sim"] == sim
                     and x["param"] == "k_on" and x["bin"] == b)
            cells.append(f"{r['coverage']:.2f}[{r['cov_lo']:.2f},{r['cov_hi']:.2f}]")
        print("  " + str(b).ljust(6) + "".join(f"{c:>16s}" for c in cells))
    return rows, leak_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_sims", type=int, default=8000)
    p.add_argument("--n_cells", type=int, default=500)
    p.add_argument("--n_test", type=int, default=2000)
    p.add_argument("--n_post", type=int, default=1000)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--outdir", type=str, default="results")
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    all_rows, all_leak = [], []
    for seed in args.seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        r, lr = run_seed(seed, args.n_sims, args.n_cells, args.n_test, args.n_post)
        all_rows += r; all_leak += lr

    rec_path = os.path.join(args.outdir, "reconcile.csv")
    with open(rec_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)
    leak_path = os.path.join(args.outdir, "leakage.csv")
    with open(leak_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_leak[0].keys()))
        w.writeheader(); w.writerows(all_leak)

    # overall per-parameter reconciliation against Table 1, averaged over seeds
    print("\n=== overall (bin=all) vs Table 1, mean over seeds ===")
    print("  sim   param     coverage      MAE   signed_bias   contraction   leaked")
    for sim in SIMS:
        for pname in PARAM_NAMES:
            sub = [x for x in all_rows if x["sim"] == sim and x["param"] == pname and x["bin"] == "all"]
            cov = np.mean([x["coverage"] for x in sub])
            mae = np.mean([x["mae"] for x in sub])
            sb = np.mean([x["signed_bias"] for x in sub])
            ct = np.mean([x["contraction"] for x in sub])
            lk = np.mean([x["leaked_frac"] for x in all_leak if x["sim"] == sim and x["bin"] == "all"])
            print(f"  {sim.upper():4s}  {pname:6s}   {cov:6.2f}     {mae:6.2f}      {sb:+6.2f}"
                  f"        {ct:6.2f}     {lk:5.2f}")
    print(f"\n[written] {rec_path}\n[written] {leak_path}")


if __name__ == "__main__":
    main()
