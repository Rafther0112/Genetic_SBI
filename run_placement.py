"""
run_placement.py
----------------
Item 5: the concentrated-surrogate placement experiment (Section 4.3, Figure 4) with more
seeds. Given an exact fraction f of n_total training parameters, each scheme decides WHICH
parameters get the exact simulator; the rest get the surrogate (NB by default).

  uniform      random subset                                   (reference)
  exact_only   the uniform subset alone, no surrogate pairs
  fano_hard    the k parameters with the highest analytic Fano
  fano_soft2   sampled without replacement, weight 1 + 2 * Fano-rank (rank in [0, 1])
  fano_soft6   same with a = 6
  inverse      the k parameters with the lowest analytic Fano (control)

All schemes share theta, the simulations and the exact test set within a seed, so
collect.py can pair each scheme against uniform. Score k_syn for the NB (the rate it damages).

IMPORTANT: run ALL seeds (0-4) with this script and report the five-seed numbers from it,
rather than mixing new seeds with the two old ones. Seeds 0-1 should land near the paper's
values (fano_soft2 about +0.012, reversed about -0.057 in distance to nominal); if they are
far off, the old implementation differed and the new five-seed numbers replace the old ones.

Example
  python run_placement.py --system telegraph --surrogate nb --f 0.25 --seed 0
"""

import argparse
import numpy as np
import common as C

SCHEMES = ["uniform", "exact_only", "fano_hard", "fano_soft2", "fano_soft6", "inverse"]


def choose(scheme, fano, k, rng):
    n = len(fano)
    order = np.argsort(fano)                           # ascending Fano
    if scheme in ("uniform", "exact_only"):
        return rng.choice(n, k, replace=False)
    if scheme == "fano_hard":
        return order[-k:]
    if scheme == "inverse":
        return order[:k]
    if scheme.startswith("fano_soft"):
        a = float(scheme.replace("fano_soft", ""))
        rank = np.empty(n); rank[order] = np.arange(n) / (n - 1)
        w = 1.0 + a * rank
        return rng.choice(n, k, replace=False, p=w / w.sum())
    raise ValueError(scheme)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--system", default="telegraph", choices=["telegraph", "threestate"])
    p.add_argument("--surrogate", default="nb")
    p.add_argument("--f", type=float, nargs="+", default=[0.25])
    p.add_argument("--schemes", nargs="+", default=SCHEMES)
    p.add_argument("--n_total", type=int, default=8000)
    p.add_argument("--n_test", type=int, default=2000)
    p.add_argument("--n_cells", type=int, default=500)
    p.add_argument("--n_post", type=int, default=1000)
    p.add_argument("--target", default="k_syn")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--test_seed", type=int, default=None, help="default 1000 + seed")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", default="results_lastmile/placement")
    a = p.parse_args()

    spec = C.system_spec(a.system)
    j = spec["names"].index(a.target)
    test_seed = a.test_seed if a.test_seed is not None else 1000 + a.seed

    rng = np.random.default_rng(a.seed)
    theta = C.sample_prior(a.n_total, rng, spec)
    tag = f"train{a.seed}_{a.n_total}"
    x_ex = C.cached_sims(tag, theta, "cme", a.n_cells, a.seed, a.system)
    x_su = C.cached_sims(tag, theta, a.surrogate, a.n_cells, a.seed, a.system)
    fano = C.analytic_fano(theta, a.system)
    th_t, x_t = C.exact_test_set(spec, a.n_test, a.n_cells, test_seed, a.system)

    for f in a.f:
        k = int(round(f * a.n_total))
        res = dict(system=a.system, surrogate=a.surrogate, target=a.target, seed=a.seed,
                   test_seed=test_seed, f=f, k_exact=k, n_total=a.n_total)
        uniform_idx = None
        for sch in a.schemes:
            srng = np.random.default_rng([a.seed, int(f * 1e4), SCHEMES.index(sch)])
            if sch in ("uniform", "exact_only"):
                if uniform_idx is None:                 # same subset for both
                    uniform_idx = choose("uniform", fano, k,
                                         np.random.default_rng([a.seed, int(f * 1e4), 0]))
                idx = uniform_idx
            else:
                idx = choose(sch, fano, k, srng)
            mask = np.zeros(a.n_total, bool); mask[idx] = True
            if sch == "exact_only":
                th, xx = theta[idx], x_ex[idx]
            else:
                th, xx = theta, np.where(mask[:, None], x_ex, x_su)
            print(f"[f={f}] {sch}: {len(th)} pairs, median Fano of exact subset "
                  f"{np.median(fano[idx]):.2f}", flush=True)
            est = C.train_npe(th, xx, spec, seed=a.seed, threads=a.threads)
            s = C.sample_posterior(est, x_t, a.n_post)
            res[sch] = C.metrics(s, th_t, j, spec["low"], spec["high"])
            res[sch]["median_fano_exact"] = float(np.median(fano[idx]))
            print("   " + ", ".join(f"{m}={v:.3f}" for m, v in res[sch].items()), flush=True)
        C.save_json(f"{a.out}/{a.system}_{a.surrogate}_f{f}_seed{a.seed}.json", res)


if __name__ == "__main__":
    main()
