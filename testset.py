"""
testset.py
----------
Single source of truth for the exact-CME evaluation snapshots, built with numpy
only (no torch / sbi) and cached. Both the reference-posterior sampler and the
future C2ST/MMD scoring step load the SAME test set, so reference[i] and the NPE
posterior queried on observation i refer to the identical snapshot (required for a
valid per-observation C2ST).

Usage (as a module):
    import testset
    ts = testset.build_or_load(seed=0, n_test=400, n_cells=500)
    ts["theta_log"]   # (n_test, 3) true log10 params
    ts["counts"]      # (n_test, n_cells) integer snapshots
    ts["summaries"]   # (n_test, 6) summary stats (telegraph.summary_stats)
    ts["emp_fano"]    # (n_test,) empirical Fano of each snapshot
"""

import os
import numpy as np

from telegraph import summary_stats

LOW = np.array([-2.0, -1.0, 0.0])     # log10 prior box, matches sbi_experiment.py
HIGH = np.array([1.3, 1.3, 2.3])
CACHE = "cache"


def build_or_load(seed=0, n_test=400, n_cells=500, cache=CACHE):
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, f"testset_{seed}_{n_test}_{n_cells}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return {k: d[k] for k in d.files}

    rng = np.random.default_rng(1000 + seed)          # same offset as run_seeds test sets
    theta_log = rng.uniform(LOW, HIGH, size=(n_test, 3))
    rates = 10.0 ** theta_log
    counts = np.empty((n_test, n_cells), dtype=np.int64)
    for i in range(n_test):
        x = rng.beta(rates[i, 0], rates[i, 1], size=n_cells)
        counts[i] = rng.poisson(rates[i, 2] * x)
    summaries = np.stack([summary_stats(counts[i]) for i in range(n_test)])
    emp_fano = summaries[:, 2].copy()

    np.savez(path, theta_log=theta_log, counts=counts,
             summaries=summaries, emp_fano=emp_fano)
    return dict(theta_log=theta_log, counts=counts,
                summaries=summaries, emp_fano=emp_fano)
