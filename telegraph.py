"""
telegraph.py
------------
Single-gene stochastic transcription (telegraph model) with two simulators:

  * sample_cme  : EXACT steady-state via the Poisson-Beta representation.
                  This is the "ground truth" / high-fidelity generator.
  * sample_cle  : Chemical Langevin Equation (Euler-Maruyama). This is the
                  CHEAP, MISSPECIFIED simulator whose Gaussian/continuum
                  approximation breaks in the bursty (low-copy) regime.

Time is measured in units of the mRNA lifetime, i.e. degradation rate gamma = 1.
Parameters theta = (k_on, k_off, k_syn), all relative to gamma.

Reactions (single gene copy, g_on + g_off = 1):
    off -> on         rate k_on * g_off
    on  -> off        rate k_off * g_on
    on  -> on + mRNA  rate k_syn * g_on      (transcription)
    mRNA -> 0         rate gamma * m         (degradation, gamma = 1)

Useful reparametrization for the "burstiness" sweep:
    burst frequency  f = k_on            (how often the gene switches on)
    burst size       b = k_syn / k_off   (mean transcripts per ON episode)
Low f + high b  ==>  very bursty, bimodal, zero-inflated  ==>  CLE fails hardest.
"""

import numpy as np


# ----------------------------------------------------------------------
# Exact steady-state moments (closed form) -- used for validation
# ----------------------------------------------------------------------
def exact_mean(k_on, k_off, k_syn):
    return k_syn * k_on / (k_on + k_off)


def exact_fano(k_on, k_off, k_syn):
    s = k_on + k_off
    return 1.0 + k_syn * k_off / (s * (s + 1.0))


# ----------------------------------------------------------------------
# EXACT CME steady state  (Poisson-Beta)   -- the "reality" generator
# ----------------------------------------------------------------------
def sample_cme(theta, n_cells, rng):
    """
    Exact stationary mRNA counts for the telegraph model.

    theta : (k_on, k_off, k_syn)
    returns integer array of shape (n_cells,)
    """
    k_on, k_off, k_syn = theta
    # x = fraction of time gene is ON, distributed Beta(k_on, k_off)
    x = rng.beta(k_on, k_off, size=n_cells)
    # mRNA | x  ~  Poisson(k_syn * x)
    return rng.poisson(k_syn * x)


# ----------------------------------------------------------------------
# CHEAP, MISSPECIFIED CLE   (Euler-Maruyama)   -- the imperfect simulator
# ----------------------------------------------------------------------
def sample_cle(theta, n_cells, rng, t_max=20.0, dt=0.01, burn_in=0.5):
    """
    Chemical Langevin Equation for the telegraph model, integrated to
    (approximate) steady state, sampled across n_cells independent runs.

    Both the gene state g_on and mRNA m are treated as CONTINUOUS with
    Gaussian reaction noise -- this is exactly the approximation that is
    wrong when molecule numbers are small (bursty regime).

    g_on is clamped to [0, 1] and m to >= 0 (reflecting the fact that the
    diffusion approximation leaks outside the physical state space -- one of
    the concrete symptoms of misspecification we want to expose).
    """
    k_on, k_off, k_syn = theta
    n_steps = int(t_max / dt)
    keep_from = int(burn_in * n_steps)

    sqrt_dt = np.sqrt(dt)
    # Initialize at the deterministic ON fraction and its mean expression
    p_on0 = k_on / (k_on + k_off)
    g = np.full(n_cells, p_on0, dtype=np.float64)
    m = np.full(n_cells, k_syn * p_on0, dtype=np.float64)

    # We time-average m over the retained window to emulate a snapshot read-out;
    # here we simply take the final state (snapshot). Keeping a running last state.
    for step in range(n_steps):
        # propensities
        a1 = k_on * (1.0 - g)      # off -> on
        a2 = k_off * g            # on  -> off
        a3 = k_syn * g            # transcription
        a4 = m                    # degradation (gamma = 1)
        a1 = np.clip(a1, 0, None)
        a2 = np.clip(a2, 0, None)
        a3 = np.clip(a3, 0, None)
        a4 = np.clip(a4, 0, None)

        # independent Wiener increments per reaction channel
        dW1 = rng.standard_normal(n_cells) * sqrt_dt
        dW2 = rng.standard_normal(n_cells) * sqrt_dt
        dW3 = rng.standard_normal(n_cells) * sqrt_dt
        dW4 = rng.standard_normal(n_cells) * sqrt_dt

        dg = (a1 - a2) * dt + np.sqrt(a1) * dW1 - np.sqrt(a2) * dW2
        dm = (a3 - a4) * dt + np.sqrt(a3) * dW3 - np.sqrt(a4) * dW4

        g = np.clip(g + dg, 0.0, 1.0)
        m = np.clip(m + dm, 0.0, None)

    # Round to integer counts so summary stats are comparable to CME/data
    return np.round(m).astype(int)


# ----------------------------------------------------------------------
# BATCHED CLE  -- integrate every (theta, cell) trajectory in parallel.
# This is the version to use for NPE training-set generation: the only
# Python-level loop is over time steps, everything else is vectorized.
# ----------------------------------------------------------------------
def sample_cle_batched(k_on, k_off, k_syn, rng, t_max=20.0, dt=0.02, burn_in=0.5):
    """
    k_on, k_off, k_syn : arrays of shape (M,) -- one entry per trajectory
                         (e.g. M = n_sims * n_cells, with rates broadcast
                          so that all cells of a given theta share its rates).
    returns integer mRNA counts, shape (M,).
    """
    k_on = np.asarray(k_on, dtype=np.float64)
    k_off = np.asarray(k_off, dtype=np.float64)
    k_syn = np.asarray(k_syn, dtype=np.float64)
    M = k_on.shape[0]
    n_steps = int(t_max / dt)
    sqrt_dt = np.sqrt(dt)

    p_on0 = k_on / (k_on + k_off)
    g = p_on0.copy()
    m = k_syn * p_on0

    for _ in range(n_steps):
        a1 = np.clip(k_on * (1.0 - g), 0, None)
        a2 = np.clip(k_off * g, 0, None)
        a3 = np.clip(k_syn * g, 0, None)
        a4 = np.clip(m, 0, None)
        noise = rng.standard_normal((4, M)) * sqrt_dt
        dg = (a1 - a2) * dt + np.sqrt(a1) * noise[0] - np.sqrt(a2) * noise[1]
        dm = (a3 - a4) * dt + np.sqrt(a3) * noise[2] - np.sqrt(a4) * noise[3]
        g = np.clip(g + dg, 0.0, 1.0)
        m = np.clip(m + dm, 0.0, None)

    return np.round(m).astype(int)


# ----------------------------------------------------------------------
# CHEAP, MISSPECIFIED LNA   (linear noise approximation)
# ----------------------------------------------------------------------
def _lna_moments(k_on, k_off, k_syn):
    """Mean and variance of mRNA under the LNA for the telegraph model.
    (These equal the EXACT telegraph mean and variance; the LNA error is
    purely in the SHAPE, since it forces a Gaussian.)"""
    a = k_on + k_off
    g = k_on / a
    mean = k_syn * g
    var = mean + k_syn**2 * k_on * k_off / (a**2 * (a + 1.0))
    return mean, var


def sample_lna(theta, n_cells, rng):
    """
    Linear noise approximation: a Gaussian with the exact telegraph mean and
    variance, clamped to >= 0 and rounded. Matches the first two moments of the
    CME exactly but cannot represent zero-inflation or bimodality -- a second,
    distinct flavor of Gaussian-surrogate misspecification.
    """
    k_on, k_off, k_syn = theta
    mean, var = _lna_moments(k_on, k_off, k_syn)
    s = rng.normal(mean, np.sqrt(max(var, 1e-12)), size=n_cells)
    return np.clip(np.round(s), 0, None).astype(int)


def sample_lna_batched(k_on, k_off, k_syn, rng):
    """Vectorized LNA over (M,) rate arrays; returns integer counts (M,)."""
    k_on = np.asarray(k_on, float); k_off = np.asarray(k_off, float); k_syn = np.asarray(k_syn, float)
    a = k_on + k_off
    mean = k_syn * k_on / a
    var = mean + k_syn**2 * k_on * k_off / (a**2 * (a + 1.0))
    s = rng.normal(mean, np.sqrt(np.clip(var, 1e-12, None)))
    return np.clip(np.round(s), 0, None).astype(int)


# ----------------------------------------------------------------------
# NON-GAUSSIAN surrogate (negative binomial) -- POSITIVE CONTROL
# Uses the SAME analytic mean and Fano as the LNA, but wraps them in a
# negative binomial (which represents overdispersion and zero-inflation)
# instead of a Gaussian. Isolates whether the failure is Gaussianity itself
# or the moment information the surrogate carries.
# ----------------------------------------------------------------------
def sample_nb_batched(k_on, k_off, k_syn, rng):
    """Vectorized NB surrogate matched to the exact telegraph mean and Fano."""
    k_on = np.asarray(k_on, float); k_off = np.asarray(k_off, float); k_syn = np.asarray(k_syn, float)
    a = k_on + k_off
    mean = k_syn * k_on / a
    fano = 1.0 + k_syn * k_off / (a * (a + 1.0))
    fano = np.clip(fano, 1.0 + 1e-6, None)          # NB needs Fano > 1
    p = 1.0 / fano                                   # Var/Mean = 1/p
    r = mean / (fano - 1.0)                           # dispersion (n successes)
    r = np.clip(r, 1e-6, None)
    return rng.negative_binomial(r, p).astype(int)


# ----------------------------------------------------------------------
# Optional: exact Gillespie SSA (slow) -- for independent validation only
# ----------------------------------------------------------------------
def sample_gillespie(theta, n_cells, rng, t_max=30.0):
    """Exact SSA; returns mRNA count at t_max for each of n_cells runs."""
    k_on, k_off, k_syn = theta
    out = np.empty(n_cells, dtype=int)
    for c in range(n_cells):
        t = 0.0
        g_on = 1 if rng.random() < k_on / (k_on + k_off) else 0
        m = 0
        while True:
            a1 = k_on * (1 - g_on)
            a2 = k_off * g_on
            a3 = k_syn * g_on
            a4 = m
            a0 = a1 + a2 + a3 + a4
            if a0 <= 0:
                break
            t += rng.exponential(1.0 / a0)
            if t > t_max:
                break
            r = rng.random() * a0
            if r < a1:
                g_on = 1
            elif r < a1 + a2:
                g_on = 0
            elif r < a1 + a2 + a3:
                m += 1
            else:
                m -= 1
        out[c] = m
    return out


# ----------------------------------------------------------------------
# Summary statistics of a single-cell snapshot (the "x" fed to NPE)
# ----------------------------------------------------------------------
def summary_stats(counts, features="full"):
    """
    Low-dimensional features of a snapshot distribution of mRNA counts.

    features="full"    : 6 features, including the fraction of zeros and the tail
                         (the non-Gaussian shape a Gaussian surrogate cannot match).
    features="moments" : 2 features, log mean and log variance only. An ablation:
                         a moment-matched Gaussian surrogate (LNA) is well-specified
                         for these, so restricting to them tests whether the failure
                         comes from non-Gaussian shape or from the feature choice.
    """
    counts = np.asarray(counts, dtype=np.float64)
    mean = counts.mean()
    var = counts.var()
    if features == "moments":
        return np.array([np.log1p(mean), np.log1p(var)], dtype=np.float64)
    fano = var / mean if mean > 1e-8 else 0.0
    frac_zero = np.mean(counts == 0)
    q50, q90 = np.percentile(counts, [50, 90])
    # log1p on the scale features stabilizes NPE training
    return np.array([np.log1p(mean), np.log1p(var), fano, frac_zero,
                     np.log1p(q50), np.log1p(q90)], dtype=np.float64)


SUMMARY_NAMES = ["log_mean", "log_var", "fano", "frac_zero", "log_q50", "log_q90"]