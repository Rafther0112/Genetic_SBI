"""
common.py
---------
Shared utilities for the last-mile experiments (crossover sweep, NB placement with more
seeds, diagnostic cost curve). Everything that touches the paper's pipeline goes through
four functions, so that it stays identical to the runs behind the tables:

    sample_prior, simulate, train_npe, sample_posterior

Simulation reuses your own sbi_experiment.simulate_batch (telegraph) and adapters.py
(three-state). NPE settings follow Appendix B: MAF, 5 transforms, 50 hidden features,
Adam 5e-4, batch 200, 10% validation, early stopping after 20 epochs. Posteriors are
sampled directly from the flow, as in the paper.

Set MOCK_NPE=1 to replace the NPE by a nearest-neighbour stand-in. That mode exists only
to dry-run the scripts end to end in a few seconds; never report its numbers.
"""

import json, os, subprocess, time, zlib
import numpy as np

MOCK = os.environ.get("MOCK_NPE", "0") == "1"

# ---------------------------------------------------------------------------
# Priors (log10 rates), Appendix B
# ---------------------------------------------------------------------------
TELEGRAPH = dict(
    names=["k_on", "k_off", "k_syn"],
    low=np.array([-2.0, -1.0, 0.0]),
    high=np.array([1.3, 1.3, 2.3]),
)


def system_spec(system):
    if system == "telegraph":
        return TELEGRAPH
    if system == "threestate":
        import adapters
        return adapters.THREESTATE
    raise ValueError(system)


def sample_prior(n, rng, spec):
    return rng.uniform(spec["low"], spec["high"], size=(n, len(spec["low"])))


# ---------------------------------------------------------------------------
# Simulators
# ---------------------------------------------------------------------------
def simulate(theta_log, sim, n_cells, rng, system="telegraph"):
    """Return the (N, 6) summary array for simulator `sim` ('cme' is exact)."""
    if system == "threestate":
        import adapters
        return adapters.simulate_threestate(theta_log, sim, n_cells, rng)
    try:
        from sbi_experiment import simulate_batch          # your pipeline, preferred
    except Exception:                                        # e.g. no torch in MOCK mode
        simulate_batch = _simulate_batch_local
    return simulate_batch(theta_log, sim, n_cells, rng)


def _simulate_batch_local(theta_log, simulator, n_cells, rng):
    """Numpy-only copy of sbi_experiment.simulate_batch (used only if it can't import)."""
    from telegraph import (sample_cle_batched, sample_lna_batched,
                           sample_nb_batched, summary_stats)
    rates = 10.0 ** np.asarray(theta_log, dtype=np.float64)
    N = rates.shape[0]
    kon, koff, ksyn = (np.repeat(rates[:, i], n_cells) for i in range(3))
    if simulator == "cme":
        counts = rng.poisson(ksyn * rng.beta(kon, koff))
    elif simulator == "cle":
        counts = sample_cle_batched(kon, koff, ksyn, rng)
    elif simulator == "lna":
        counts = sample_lna_batched(kon, koff, ksyn, rng)
    elif simulator == "nb":
        counts = sample_nb_batched(kon, koff, ksyn, rng)
    else:
        raise ValueError(simulator)
    counts = counts.reshape(N, n_cells)
    return np.stack([summary_stats(c) for c in counts])


def cached_sims(tag, theta_log, sim, n_cells, seed, system, cache="cache_lastmile"):
    """Simulate once per (tag, sim, seed) and cache; the tag must identify theta."""
    os.makedirs(cache, exist_ok=True)
    path = f"{cache}/{system}_{tag}_{sim}_{n_cells}_{seed}.npz"
    if os.path.exists(path):
        d = np.load(path)
        if d["theta"].shape == theta_log.shape and np.allclose(d["theta"], theta_log):
            return d["x"]
    rng = np.random.default_rng([seed, zlib.crc32(sim.encode()), zlib.crc32(tag.encode())])
    t = time.time()
    x = simulate(theta_log, sim, n_cells, rng, system)
    print(f"  [sim] {system}/{sim} x{len(theta_log)} in {time.time()-t:.0f}s", flush=True)
    np.savez(path, theta=theta_log, x=x)
    return x


# ---------------------------------------------------------------------------
# Analytic Fano factors (used to rank training parameters and to bin the diagnostic)
# ---------------------------------------------------------------------------
def analytic_fano(theta_log, system="telegraph"):
    r = 10.0 ** np.asarray(theta_log, dtype=np.float64)
    if system == "telegraph":
        kon, koff, ksyn = r[:, 0], r[:, 1], r[:, 2]
        s = kon + koff
        return 1.0 + ksyn * koff / (s * (s + 1.0))
    # three-state, theta = (k01, k10, k12, k21, k_syn); Fano = 1 + ksyn([(I-Q)^-1]_AA - pi_A)
    out = np.empty(len(r))
    for i, (k01, k10, k12, k21, ksyn) in enumerate(r):
        Q = np.array([[-k01, k01, 0.0],
                      [k10, -(k10 + k12), k12],
                      [0.0, k21, -k21]])
        pi = np.array([1.0, k01 / k10, k01 * k12 / (k10 * k21)]); pi /= pi.sum()
        R = np.linalg.solve(np.eye(3) - Q, np.eye(3))
        out[i] = 1.0 + ksyn * (R[2, 2] - pi[2])
    return out


# ---------------------------------------------------------------------------
# NPE
# ---------------------------------------------------------------------------
def train_npe(theta_log, x, spec, seed, threads=None):
    if MOCK:
        return _MockNPE(theta_log, x)
    import torch
    from sbi.inference import NPE
    from sbi.neural_nets import posterior_nn
    from sbi.utils import BoxUniform
    if threads:
        torch.set_num_threads(threads)
    torch.manual_seed(seed); np.random.seed(seed)
    prior = BoxUniform(low=torch.as_tensor(spec["low"], dtype=torch.float32),
                       high=torch.as_tensor(spec["high"], dtype=torch.float32))
    de = posterior_nn(model="maf", hidden_features=50, num_transforms=5)
    inf = NPE(prior=prior, density_estimator=de)
    inf.append_simulations(torch.as_tensor(theta_log, dtype=torch.float32),
                           torch.as_tensor(x, dtype=torch.float32))
    est = inf.train(training_batch_size=200, learning_rate=5e-4,
                    validation_fraction=0.1, stop_after_epochs=20,
                    show_train_summary=False)
    return est


def sample_posterior(est, x_test, n_post=1000, chunk=100):
    """(n_test, n_post, d) samples, drawn directly from the flow (no prior rejection)."""
    if MOCK:
        return est.sample(x_test, n_post)
    import torch
    out = []
    X = torch.as_tensor(x_test, dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, len(X), chunk):
            xb = X[i:i + chunk]
            s = est.sample((n_post,), condition=xb)          # (n_post, B, d) in sbi 0.26
            s = s.reshape(n_post, len(xb), -1).permute(1, 0, 2)
            out.append(s.cpu().numpy())
    return np.concatenate(out, 0)


class _MockNPE:
    """Nearest-neighbour stand-in for dry runs only."""
    def __init__(self, theta, x):
        self.t = theta; self.mu = x.mean(0); self.sd = x.std(0) + 1e-9
        self.x = (x - self.mu) / self.sd
    def sample(self, x_test, n_post):
        rng = np.random.default_rng(0)
        k = max(5, min(50, len(self.t) // 10))
        z = (x_test - self.mu) / self.sd
        out = np.empty((len(z), n_post, self.t.shape[1]))
        for i, zi in enumerate(z):
            nn = np.argsort(((self.x - zi) ** 2).sum(1))[:k]
            pick = rng.choice(nn, n_post)
            out[i] = self.t[pick] + 0.05 * rng.standard_normal((n_post, self.t.shape[1]))
        return out


# ---------------------------------------------------------------------------
# Metrics (same conventions as the paper: rank u = P(sample < truth), covered if
# 0.05 <= u <= 0.95; bias of the posterior mean in log10 units)
# ---------------------------------------------------------------------------
def metrics(samples, theta_true, j, low=None, high=None):
    s = samples[:, :, j]; t = theta_true[:, j]
    u = (s < t[:, None]).mean(1)
    cov = ((u >= 0.05) & (u <= 0.95)).astype(float)
    mean = s.mean(1)
    q05, q95 = np.percentile(s, [5, 95], axis=1)
    m = dict(
        coverage=float(cov.mean()),
        dist_nominal=float(abs(cov.mean() - 0.90)),
        signed_bias=float((mean - t).mean()),
        abs_bias=float(np.abs(mean - t).mean()),
        post_std=float(s.std(1).mean()),
        ci90_width=float((q95 - q05).mean()),
    )
    if low is not None:
        out = ((samples < low) | (samples > high)).any(2).mean(1)
        m["leaked_mass"] = float(out.mean())
    return m


# ---------------------------------------------------------------------------
# Test sets and bookkeeping
# ---------------------------------------------------------------------------
def exact_test_set(spec, n_test, n_cells, test_seed, system):
    rng = np.random.default_rng(test_seed)
    theta = sample_prior(n_test, rng, spec)
    x = cached_sims(f"test{test_seed}_{n_test}", theta, "cme", n_cells, test_seed, system)
    return theta, x


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    obj = dict(obj, git_commit=git_commit(), mock=MOCK,
               time=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
    print(f"[saved] {path}", flush=True)
