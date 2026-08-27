"""
tests/test_simulators.py
------------------------
Correctness tests for the telegraph forward simulators (workplan P0-3).

These make the pipeline trustworthy before any new experiment is run, and they
pin down several details the review flagged as unspecified (W8):

  * closed-form moments Eq. (2) vs Monte Carlo (exact_mean / exact_fano)
  * Poisson-Beta zero fraction vs the hypergeometric 1F1 identity
  * NB surrogate reproduces the EXACT mean and Fano it is built to match
  * NB parametrization stays well-behaved as F -> 1 (near-Poisson limit)
  * LNA reproduces (mean, var) where clamping is negligible
  * summary_stats on an all-zero snapshot (the log-median / frac_zero question)
  * CLE Euler-Maruyama step-size convergence

Run:
    pytest -q tests/test_simulators.py

Only needs numpy + scipy (no torch / sbi).
"""

import numpy as np
import pytest
from scipy.special import hyp1f1

import telegraph as T


# ----------------------------------------------------------------------
# Representative parameter sets, chosen to span the burstiness axis.
# rates = (k_on, k_off, k_syn); gamma = 1.
# ----------------------------------------------------------------------
SAFE = [                      # fast-switching / high-mean, low Fano
    (20.0, 1.0, 40.0),
    (10.0, 2.0, 80.0),
    (30.0, 5.0, 120.0),
]
BURSTY = [                    # rare switching / low-copy, high Fano, zero-inflated
    (0.05, 1.0, 60.0),
    (0.1, 2.0, 100.0),
    (0.03, 1.0, 40.0),
]
ALL = SAFE + BURSTY


def _mc_mean_fano(counts):
    m = counts.mean()
    v = counts.var()
    return m, (v / m if m > 0 else 0.0)


# ----------------------------------------------------------------------
# 1. Closed-form moments vs Monte Carlo from the exact Poisson-Beta.
# ----------------------------------------------------------------------
@pytest.mark.parametrize("rates", ALL)
def test_exact_moments_match_montecarlo(rates):
    kon, koff, ksyn = rates
    rng = np.random.default_rng(0)
    n = 500_000
    x = rng.beta(kon, koff, size=n)
    counts = rng.poisson(ksyn * x)
    m, f = _mc_mean_fano(counts)
    assert m == pytest.approx(T.exact_mean(*rates), rel=0.03)
    assert f == pytest.approx(T.exact_fano(*rates), rel=0.08)


# ----------------------------------------------------------------------
# 2. Poisson-Beta zero fraction  P(m=0) = E[e^{-ksyn x}] = 1F1(kon; kon+koff; -ksyn).
# ----------------------------------------------------------------------
@pytest.mark.parametrize("rates", ALL)
def test_zero_fraction_matches_1F1(rates):
    kon, koff, ksyn = rates
    rng = np.random.default_rng(1)
    n = 500_000
    x = rng.beta(kon, koff, size=n)
    counts = rng.poisson(ksyn * x)
    frac_zero = np.mean(counts == 0)
    analytic = hyp1f1(kon, kon + koff, -ksyn)
    # tolerance loosens for tiny probabilities (MC noise on rare zeros)
    tol = max(0.01, 0.05 * analytic)
    assert frac_zero == pytest.approx(analytic, abs=tol)


# ----------------------------------------------------------------------
# 3. NB surrogate reproduces the EXACT mean and Fano it is constructed to carry
#    (no clamping: this must hold across the whole grid where F > 1).
# ----------------------------------------------------------------------
@pytest.mark.parametrize("rates", ALL)
def test_nb_matches_target_moments(rates):
    kon, koff, ksyn = rates
    rng = np.random.default_rng(2)
    n = 500_000
    counts = T.sample_nb_batched(np.full(n, kon), np.full(n, koff), np.full(n, ksyn), rng)
    m, f = _mc_mean_fano(counts)
    assert m == pytest.approx(T.exact_mean(*rates), rel=0.04)
    assert f == pytest.approx(T.exact_fano(*rates), rel=0.08)


# ----------------------------------------------------------------------
# 4. NB stays well-behaved as F -> 1 (the r = mu/(F-1) question, W8):
#    a near-Poisson target must run and return Fano close to 1.
# ----------------------------------------------------------------------
def test_nb_near_poisson_limit():
    # very fast switching -> Fano barely above 1
    kon, koff, ksyn = 100.0, 0.1, 50.0
    assert T.exact_fano(kon, koff, ksyn) < 1.02
    rng = np.random.default_rng(3)
    n = 500_000
    counts = T.sample_nb_batched(np.full(n, kon), np.full(n, koff), np.full(n, ksyn), rng)
    assert np.all(np.isfinite(counts))
    _, f = _mc_mean_fano(counts)
    assert f == pytest.approx(1.0, abs=0.05)


# ----------------------------------------------------------------------
# 5. LNA reproduces (mean, var) where clamping is negligible (SAFE regime only).
#    In the bursty regime clamping distorts the moments on purpose; that is a
#    documented finding, not a bug, so it is not asserted here.
# ----------------------------------------------------------------------
@pytest.mark.parametrize("rates", SAFE)
def test_lna_matches_moments_when_unclamped(rates):
    kon, koff, ksyn = rates
    rng = np.random.default_rng(4)
    n = 500_000
    counts = T.sample_lna_batched(np.full(n, kon), np.full(n, koff), np.full(n, ksyn), rng)
    mean_exact = T.exact_mean(*rates)
    var_exact = mean_exact * T.exact_fano(*rates)
    # confirm we are actually in the negligible-clamping regime
    assert mean_exact / np.sqrt(var_exact) > 3.0
    assert counts.mean() == pytest.approx(mean_exact, rel=0.03)
    assert counts.var() == pytest.approx(var_exact, rel=0.10)


# ----------------------------------------------------------------------
# 6. summary_stats on an all-zero snapshot: no NaN, sensible values.
#    (Answers the "log-median when the median is zero" question in W8.)
# ----------------------------------------------------------------------
def test_summary_stats_all_zero_snapshot():
    counts = np.zeros(500, dtype=int)
    full = T.summary_stats(counts, features="full")
    assert full.shape == (6,)
    assert np.all(np.isfinite(full))
    log_mean, log_var, fano, frac_zero, log_q50, log_q90 = full
    assert frac_zero == 1.0
    assert fano == 0.0            # guarded against divide-by-zero
    assert log_mean == 0.0        # log1p(0)
    assert log_q50 == 0.0         # log1p(median=0)
    mom = T.summary_stats(counts, features="moments")
    assert mom.shape == (2,)
    assert np.all(np.isfinite(mom))


# ----------------------------------------------------------------------
# 7. Batched samplers return non-negative integer counts of the right length.
# ----------------------------------------------------------------------
@pytest.mark.parametrize("sampler", [T.sample_cle_batched, T.sample_lna_batched, T.sample_nb_batched])
def test_outputs_nonneg_integer(sampler):
    rng = np.random.default_rng(5)
    n = 5_000
    kon = np.full(n, 0.1); koff = np.full(n, 1.0); ksyn = np.full(n, 50.0)
    counts = sampler(kon, koff, ksyn, rng)
    assert counts.shape == (n,)
    assert np.all(np.isfinite(counts))
    assert np.all(counts >= 0)
    assert np.issubdtype(counts.dtype, np.integer)


# ----------------------------------------------------------------------
# 8. CLE Euler-Maruyama step-size convergence: the integrated moments should
#    stabilize as dt shrinks (dt=1e-2 vs 5e-3 agree; the coarse dt=1e-1 need not).
# ----------------------------------------------------------------------
@pytest.mark.parametrize("rates", [SAFE[0], BURSTY[0]])
def test_cle_stepsize_convergence(rates):
    kon, koff, ksyn = rates
    n = 40_000

    def cle_mean(dt):
        rng = np.random.default_rng(6)   # same seed -> compare the integrator, not noise
        c = T.sample_cle_batched(np.full(n, kon), np.full(n, koff), np.full(n, ksyn),
                                 rng, dt=dt)
        return c.mean()

    m_fine = cle_mean(0.005)
    m_med = cle_mean(0.010)
    assert m_med == pytest.approx(m_fine, rel=0.10)
