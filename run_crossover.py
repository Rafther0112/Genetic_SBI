"""
run_crossover.py
----------------
Item 4: push the mixture-vs-exact-only comparison into the scarce-exact regime, where
Proposition 1 predicts a crossover (a biased but plentiful training set should beat a tiny
exact one).

For each exact fraction f, a nested uniform subset of k = round(f * n_total) training
parameters gets the exact simulator. Two arms share those k exact simulations:
  exact_only : train on the k exact pairs alone
  mixture    : the same k exact pairs + surrogate pairs for the other n_total - k
Both arms are evaluated on one exact test set per seed.

Include f = 0.02 and 0.05 as overlap points: they must reproduce the paper's numbers
(f = 0.02: exact-only coverage ~0.91, mixture ~0.70) before the new points are trusted.

Examples
  python run_crossover.py --system telegraph --surrogate lna --seed 0
  python run_crossover.py --system threestate --surrogate nb --seed 0 --target k_syn \
         --fracs 0.005 0.01 0.02 0.05
"""

import argparse
import numpy as np
import common as C


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--system", default="telegraph", choices=["telegraph", "threestate"])
    p.add_argument("--surrogate", default="lna")
    p.add_argument("--fracs", type=float, nargs="+",
                   default=[0.0025, 0.005, 0.01, 0.02, 0.05])
    p.add_argument("--n_total", type=int, default=8000)
    p.add_argument("--n_test", type=int, default=2000)
    p.add_argument("--n_cells", type=int, default=500)
    p.add_argument("--n_post", type=int, default=1000)
    p.add_argument("--target", default=None, help="parameter to score (default k_on / k_syn)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--test_seed", type=int, default=None, help="default 1000 + seed")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", default="results_lastmile/crossover")
    a = p.parse_args()

    spec = C.system_spec(a.system)
    target = a.target or ("k_on" if a.system == "telegraph" else "k_syn")
    j = spec["names"].index(target)
    test_seed = a.test_seed if a.test_seed is not None else 1000 + a.seed

    rng = np.random.default_rng(a.seed)
    theta = C.sample_prior(a.n_total, rng, spec)
    tag = f"train{a.seed}_{a.n_total}"
    x_ex = C.cached_sims(tag, theta, "cme", a.n_cells, a.seed, a.system)
    x_su = C.cached_sims(tag, theta, a.surrogate, a.n_cells, a.seed, a.system)
    perm = np.random.default_rng(10_000 + a.seed).permutation(a.n_total)   # nested subsets

    th_t, x_t = C.exact_test_set(spec, a.n_test, a.n_cells, test_seed, a.system)

    for f in sorted(a.fracs):
        k = max(1, int(round(f * a.n_total)))
        idx = perm[:k]
        mask = np.zeros(a.n_total, bool); mask[idx] = True
        res = dict(system=a.system, surrogate=a.surrogate, target=target, seed=a.seed,
                   test_seed=test_seed, f=f, k_exact=k, n_total=a.n_total)
        arms = {
            "exact_only": (theta[idx], x_ex[idx]),
            "mixture": (theta, np.where(mask[:, None], x_ex, x_su)),
        }
        for arm, (th, xx) in arms.items():
            print(f"[f={f} k={k}] training {arm} on {len(th)} pairs", flush=True)
            est = C.train_npe(th, xx, spec, seed=a.seed, threads=a.threads)
            s = C.sample_posterior(est, x_t, a.n_post)
            res[arm] = C.metrics(s, th_t, j, spec["low"], spec["high"])
            print(f"   {arm}: " + ", ".join(f"{m}={v:.3f}" for m, v in res[arm].items()),
                  flush=True)
        C.save_json(f"{a.out}/{a.system}_{a.surrogate}_f{f}_seed{a.seed}.json", res)


if __name__ == "__main__":
    main()
