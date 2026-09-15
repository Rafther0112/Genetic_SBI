"""
run_exact_only.py
-----------------
Exact-only baseline for Table 2 (review W2). At each exact budget f, trains
  * uniform     : f*N exact + (1-f)*N LNA simulations, exact ones placed at random
                  (same scheme as run_budget4.py, rerun here so the pair shares
                  training parameters, seeds and test set)
  * exact_only  : ONLY the same f*N exact simulations, no LNA simulations
and reports k_on 90% coverage on one shared exact test set, mean +/- std over seeds,
plus |coverage - 0.90|. Decides whether the LNA simulations add anything at equal
exact budget.

Reuses select, cov_kon and train_alloc from run_budget4.py (run from the repo root).
Needs torch + sbi.   Usage:
    python run_exact_only.py --n_sims 8000 --n_cells 500 --n_test 2000 --seeds 0 1 2 3 4
"""
import argparse, csv, os
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

from sbi.inference import NPE
from sbi_experiment import make_prior, simulate_batch, theta_to_rates
from telegraph import exact_fano
from run_budget4 import select, cov_kon, train_alloc


def train_exact_only(theta, exact_mask, n_cells, rng):
    idx = np.where(exact_mask)[0]
    th = theta[idx]
    x = torch.as_tensor(simulate_batch(th.numpy(), "cme", n_cells, rng), dtype=torch.float32)
    inf = NPE(prior=make_prior()); inf.append_simulations(th, x)
    return inf.build_posterior(inf.train())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_sims", type=int, default=8000)
    ap.add_argument("--n_cells", type=int, default=500)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--n_post", type=int, default=1000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    ap.add_argument("--out", default="results/exact_only.csv")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    prior = make_prior()
    torch.manual_seed(999)                                   # fixed, shared test set
    tt = prior.sample((a.n_test,)).numpy()
    x_test = torch.as_tensor(simulate_batch(tt, "cme", a.n_cells, np.random.default_rng(999)),
                             dtype=torch.float32)

    rows = []
    for seed in a.seeds:
        torch.manual_seed(seed); rng = np.random.default_rng(seed)
        theta = prior.sample((a.n_sims,))
        fano = np.array([exact_fano(*r) for r in theta_to_rates(theta.numpy())])
        rr = np.random.default_rng(seed + 1)
        for f in a.fracs:
            k = int(round(f * a.n_sims))
            m = np.zeros(a.n_sims, bool); m[select("uniform", fano, k, rr)] = True
            torch.manual_seed(seed)
            c_mix = cov_kon(train_alloc(theta, m, a.n_cells, rng), x_test, tt, a.n_post)
            torch.manual_seed(seed)
            c_ex = cov_kon(train_exact_only(theta, m, a.n_cells, rng), x_test, tt, a.n_post)
            rows.append(dict(seed=seed, f=f, n_exact=k, uniform=c_mix, exact_only=c_ex))
            print(f"[seed {seed} f={f:.2f}] uniform={c_mix:.3f}  exact_only={c_ex:.3f}", flush=True)
            with open(a.out, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    print(f"\n=== k_on coverage, mean +/- std over seeds {a.seeds} (nominal 0.90) ===")
    print("   f   n_exact   uniform          exact_only       |u-0.9|  |e-0.9|  paired diff (u-e)")
    for f in a.fracs:
        u = np.array([r["uniform"] for r in rows if r["f"] == f])
        e = np.array([r["exact_only"] for r in rows if r["f"] == f])
        d = u - e
        print(f"{f:5.2f}  {int(round(f*a.n_sims)):7d}   {u.mean():.2f} +/- {u.std():.2f}   "
              f"{e.mean():.2f} +/- {e.std():.2f}   {abs(u.mean()-.9):.2f}    {abs(e.mean()-.9):.2f}    "
              f"{d.mean():+.3f} +/- {d.std():.3f}")
    print("\nReading: if |exact_only - 0.9| <= |uniform - 0.9| within the seed spread at every f,")
    print("the LNA simulations add nothing at equal exact budget. Paste this table to Claude.")
    print(f"[written] {a.out}")


if __name__ == "__main__":
    main()
