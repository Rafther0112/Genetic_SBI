"""
run_experiment.py
-----------------
End-to-end driver for the misspecification experiment.

  1. Build & CACHE training sets from the CME and CLE simulators (one-time cost).
  2. Train one NPE per simulator.
  3. Evaluate both on EXACT CME test data (bias + calibration).
  4. Produce the "money plot" with bootstrap error bars, binned by the OBSERVABLE
     empirical Fano factor of each snapshot (the quantity the budget rule uses),
     and report its agreement with the analytic Fano.

Usage:
    python run_experiment.py --n_sims 8000 --n_cells 500 --n_test 400
    python run_experiment.py ... --bin_by analytic     # oracle binning instead

Sampling note: a CLE-trained posterior queried on CME data is out-of-distribution
and places mass outside the prior box, so sbi's default rejection sampler accepts
~0% and hangs. We sample the trained flow DIRECTLY (no prior rejection). This is
faithful for the diagnostic and is itself a symptom of misspecification.
"""

import argparse, os, time
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

from sbi.inference import NPE

from sbi_experiment import make_prior, simulate_batch, theta_to_rates
from telegraph import exact_fano

CACHE = "cache"
os.makedirs(CACHE, exist_ok=True)
PARAM_NAMES = ["k_on", "k_off", "k_syn"]


def cached_training_set(simulator, n_sims, n_cells, seed):
    path = f"{CACHE}/train_{simulator}_{n_sims}_{n_cells}_{seed}.npz"
    if os.path.exists(path):
        d = np.load(path)
        print(f"[cache] loaded {path}")
        return torch.as_tensor(d["theta"], dtype=torch.float32), \
               torch.as_tensor(d["x"], dtype=torch.float32)
    print(f"[sim] generating {n_sims} '{simulator}' sims (cached after) ...")
    rng = np.random.default_rng(seed)
    prior = make_prior()
    theta = prior.sample((n_sims,))
    t = time.time()
    x = simulate_batch(theta.numpy(), simulator, n_cells, rng)
    print(f"      done in {time.time()-t:.0f}s")
    np.savez(path, theta=theta.numpy(), x=x)
    return theta, torch.as_tensor(x, dtype=torch.float32)


def train(simulator, n_sims, n_cells, seed):
    theta, x = cached_training_set(simulator, n_sims, n_cells, seed)
    inf = NPE(prior=make_prior())
    inf.append_simulations(theta, x)
    de = inf.train()
    return inf.build_posterior(de)


def _flow_sample(posterior, x_row, n_post):
    """Sample the trained flow directly, bypassing prior-rejection sampling."""
    s = posterior.posterior_estimator.sample((n_post,), condition=x_row.unsqueeze(0))
    return s.reshape(n_post, -1).detach()


def evaluate(posterior, n_test, n_cells, seed, n_post=1000):
    """
    Confront the trained posterior with EXACT CME test data.
    Returns true theta, posterior-mean bias (log10 units), SBC ranks, n_post,
    and the empirical Fano factor of each test snapshot (an OBSERVABLE quantity,
    summary-stat index 2).
    """
    rng = np.random.default_rng(seed)
    prior = make_prior()
    theta_test = prior.sample((n_test,))
    tt = theta_test.numpy()
    x_test = torch.as_tensor(
        simulate_batch(tt, "cme", n_cells, rng), dtype=torch.float32)

    post_means = np.empty((n_test, 3))
    ranks = np.empty((n_test, 3))
    for i in range(n_test):
        s = _flow_sample(posterior, x_test[i], n_post).numpy()
        post_means[i] = s.mean(0)
        ranks[i] = (s < tt[i]).sum(0)
    bias = post_means - tt
    emp_fano = x_test[:, 2].numpy()
    return tt, bias, ranks, n_post, emp_fano


def sbc_calibration_error(ranks, n_post):
    """Mean |empirical CDF - uniform| of SBC ranks (0 = perfectly calibrated)."""
    u = (ranks + 0.5) / (n_post + 1)
    grid = np.linspace(0, 1, 50)
    errs = []
    for j in range(u.shape[1]):
        ecdf = np.searchsorted(np.sort(u[:, j]), grid) / len(u)
        errs.append(np.mean(np.abs(ecdf - grid)))
    return np.array(errs)


def coverage_90(ranks, n_post):
    """Empirical coverage of the central 90% credible interval, per parameter."""
    u = ranks / n_post
    return np.mean((u >= 0.05) & (u <= 0.95), axis=0)


def _bootstrap_bin(vals, idx, n_bins, fn, n_boot=1000, seed=1):
    """Per-bin statistic with a bootstrap 95% interval over the points in the bin."""
    rng = np.random.default_rng(seed)
    m = np.full(n_bins, np.nan); lo = np.full(n_bins, np.nan); hi = np.full(n_bins, np.nan)
    for b in range(n_bins):
        v = vals[idx == b]
        if len(v) < 4:
            continue
        m[b] = fn(v)
        boot = [fn(v[rng.integers(0, len(v), len(v))]) for _ in range(n_boot)]
        lo[b], hi[b] = np.percentile(boot, [2.5, 97.5])
    return m, lo, hi


def money_plot(results, theta_test, emp_fano, bin_by="empirical", out="money_plot.png"):
    """
    results[sim] = (bias, ranks, n_post). Binned by the empirical Fano factor
    (observable at inference time) or the analytic Fano factor (oracle).
    Bootstrap 95% error bars over test points within each bin.
    """
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if bin_by == "analytic":
        fano = np.array([exact_fano(*r) for r in theta_to_rates(theta_test)])
        xlab = "true burstiness  (analytic Fano factor)"
    else:
        fano = emp_fano
        xlab = "empirical burstiness  (Fano factor of snapshot)"

    edges = np.quantile(fano, np.linspace(0, 1, 7)); edges[-1] += 1e-6
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(fano, edges) - 1, 0, len(centers) - 1)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    col = {"cme": "#0f6e56", "cle": "#ba7517", "lna": "#7e4ea8", "nb": "#2b6cb0"}
    lab = {"cme": "NPE trained on CME (control)",
           "cle": "NPE trained on CLE (misspecified)",
           "lna": "NPE trained on LNA (misspecified)",
           "nb": "NPE trained on NB (non-Gaussian control)"}
    cov_fn = lambda v: np.mean((v >= 0.05) & (v <= 0.95))
    mk = {"cme": "o", "cle": "s", "lna": "^", "nb": "D"}

    for sim, (bias, ranks, n_post) in results.items():
        err = np.linalg.norm(bias, axis=1)
        m, lo, hi = _bootstrap_bin(err, idx, len(centers), np.mean)
        ax[0].errorbar(centers, m, yerr=[m - lo, hi - m], fmt=mk[sim] + "-",
                       color=col[sim], label=lab[sim], capsize=2, lw=1.5)
        u = ranks[:, 0] / n_post
        m, lo, hi = _bootstrap_bin(u, idx, len(centers), cov_fn)
        ax[1].errorbar(centers, m, yerr=[m - lo, hi - m], fmt=mk[sim] + "-",
                       color=col[sim], label=lab[sim], capsize=2, lw=1.5)

    ax[1].axhline(0.9, ls=":", color="gray", label="nominal 90%")
    ax[0].set_title("A. Parameter bias vs burstiness")
    ax[0].set_ylabel("posterior-mean error  ||Δθ||  (log10 units)")
    ax[1].set_title("B. 90% CI coverage vs burstiness")
    ax[1].set_ylabel("empirical coverage of 90% CI (k_on)")
    ax[1].set_ylim(0, 1)
    for a in ax:
        a.set_xscale("log"); a.set_xlabel(xlab); a.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[plot] saved {out}  (binned by {bin_by} Fano)")


def print_table(results):
    """Print a LaTeX-ready summary of bias and 90% coverage per parameter."""
    print("\n--- summary table (bias / 90% coverage per parameter) ---")
    for sim, (bias, ranks, n_post) in results.items():
        ab = np.abs(bias).mean(0); cov = coverage_90(ranks, n_post)
        print(f"  {sim.upper():3s}  bias " +
              " ".join(f"{n}={ab[i]:.2f}" for i, n in enumerate(PARAM_NAMES)) +
              "  | cov90 " +
              " ".join(f"{n}={cov[i]:.2f}" for i, n in enumerate(PARAM_NAMES)))


def sbc_histograms(results, out="sbc_histograms.png"):
    """Grid of SBC rank histograms (rows = simulators, cols = parameters).
    A calibrated estimator gives uniform ranks (flat within the grey band)."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sims = [s for s in ["cme", "nb", "lna", "cle"] if s in results]
    nb_bins = 10
    fig, ax = plt.subplots(len(sims), 3, figsize=(9, 2.1 * len(sims)),
                           sharex=True, sharey=True)
    ax = np.atleast_2d(ax)
    for r, sim in enumerate(sims):
        _, ranks, n_post = results[sim]
        n_test = ranks.shape[0]
        exp = n_test / nb_bins                       # expected count per bin if uniform
        band = 2.0 * np.sqrt(exp * (1 - 1.0 / nb_bins))
        for c in range(3):
            a = ax[r, c]
            a.axhspan(exp - band, exp + band, color="0.85", zorder=0)
            a.hist(ranks[:, c] / n_post, bins=nb_bins, range=(0, 1),
                   color="#444", alpha=0.85)
            if r == 0:
                a.set_title(PARAM_NAMES[c])
            if c == 0:
                a.set_ylabel(sim.upper())
    fig.supxlabel("normalized SBC rank"); fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[plot] saved {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_sims", type=int, default=8000)
    p.add_argument("--n_cells", type=int, default=500)
    p.add_argument("--n_test", type=int, default=400)
    p.add_argument("--n_post", type=int, default=1000)
    p.add_argument("--bin_by", choices=["empirical", "analytic"], default="empirical")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    results = {}
    theta_test_ref = emp_fano_ref = None
    for sim in ["cme", "cle", "lna", "nb"]:
        print(f"\n===== {sim.upper()} =====")
        post = train(sim, args.n_sims, args.n_cells, args.seed)
        tt, bias, ranks, n_post, emp_fano = evaluate(
            post, args.n_test, args.n_cells, seed=999, n_post=args.n_post)
        theta_test_ref, emp_fano_ref = tt, emp_fano
        results[sim] = (bias, ranks, n_post)
        ce = sbc_calibration_error(ranks, n_post)
        cov = coverage_90(ranks, n_post)
        print("  mean |bias| (log10): " +
              "  ".join(f"{n}={np.abs(bias)[:,i].mean():.3f}" for i, n in enumerate(PARAM_NAMES)))
        print("  SBC calib error:     " +
              "  ".join(f"{n}={ce[i]:.3f}" for i, n in enumerate(PARAM_NAMES)))
        print("  90% CI coverage:     " +
              "  ".join(f"{n}={cov[i]:.2f}" for i, n in enumerate(PARAM_NAMES)))

    # agreement between the observable empirical Fano and the oracle analytic Fano
    ana = np.array([exact_fano(*r) for r in theta_to_rates(theta_test_ref)])
    corr = np.corrcoef(np.log(ana), np.log(np.clip(emp_fano_ref, 1e-3, None)))[0, 1]
    print(f"\n[diagnostic] corr(log analytic Fano, log empirical Fano) = {corr:.3f}")

    print_table(results)
    money_plot(results, theta_test_ref, emp_fano_ref, bin_by=args.bin_by)
    sbc_histograms(results)


if __name__ == "__main__":
    main()