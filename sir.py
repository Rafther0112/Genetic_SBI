"""
sir.py
------
W1: a system OUTSIDE gene expression that exercises the SAME mechanism, to test the
generality claim. The stochastic SIR epidemic has a discrete, low-copy infected count
I; the chemical Langevin (diffusion) approximation is the standard cheap surrogate and
is known to fail near the epidemic threshold, where whether an outbreak takes off or
dies out is a discrete stochastic event that a continuum cannot represent.

Observable analogous to a single-cell snapshot: the distribution of FINAL OUTBREAK
SIZES R(inf) = N - S(inf) over many independent outbreaks with the same (beta, gamma).
Its Fano factor spikes near R0 = beta/gamma = 1, where the final-size law is bimodal
(fizzle vs take-off). The exact simulator is Gillespie SSA; the surrogate is the CLE.

Needs numpy only.
"""

import numpy as np


def sample_sir_ssa(beta, gamma, N, I0, n_real, rng):
    """Exact final sizes R(inf) for n_real independent outbreaks."""
    finals = np.empty(n_real, dtype=np.int64)
    for k in range(n_real):
        S = N - I0; I = I0
        while I > 0:
            a_inf = beta * S * I / N
            a0 = a_inf + gamma * I
            if a0 <= 0:
                break
            if rng.random() * a0 < a_inf:
                S -= 1; I += 1          # infection
            else:
                I -= 1                  # recovery
        finals[k] = N - S              # total ever infected
    return finals


def sample_sir_cle(beta, gamma, N, I0, n_real, rng, dt=0.05, max_steps=5000):
    """CLE (diffusion) final sizes: S, I evolved as continuous diffusions."""
    S = np.full(n_real, float(N - I0)); I = np.full(n_real, float(I0))
    active = np.ones(n_real, bool); sdt = np.sqrt(dt)
    for _ in range(max_steps):
        if not active.any():
            break
        a_inf = np.clip(beta * S * I / N, 0, None)
        a_rec = np.clip(gamma * I, 0, None)
        w1 = rng.standard_normal(n_real) * sdt
        w2 = rng.standard_normal(n_real) * sdt
        dS = -a_inf * dt - np.sqrt(a_inf) * w1
        dI = (a_inf - a_rec) * dt + np.sqrt(a_inf) * w1 - np.sqrt(a_rec) * w2
        S = np.where(active, np.clip(S + dS, 0, N), S)
        I = np.where(active, np.clip(I + dI, 0, None), I)
        active = I > 0.5               # outbreak over when I drops below one individual
    return np.round(N - S).astype(int)
