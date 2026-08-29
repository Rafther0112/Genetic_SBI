"""
hybrid.py
---------
E2 (review W4): a hybrid simulator that keeps the promoter switching DISCRETE and
exact (a two-state Markov chain, g in {0,1}) and applies the CLE only to the mRNA.
This is the standard cheap-but-correct treatment (Bokes et al. 2012; Lin & Doering
2016) and the missing baseline: it separates whether the CLE failure comes from the
Gaussian noise on the mRNA (the paper's thesis) or from diffusing the single-copy
promoter state g in [0,1] (the straw-man the full CLE uses).

Contrast with telegraph.sample_cle_batched, which relaxes g to a continuous variable
in [0,1]. Here g flips between 0 and 1 with the exact switching rates each step, and
only m follows Euler-Maruyama Gaussian dynamics.

Needs numpy only. Same (M,) -> (M,) interface as the other batched samplers.
"""

import numpy as np


def sample_hybrid_batched(k_on, k_off, k_syn, rng, t_max=20.0, dt=0.02):
    k_on = np.asarray(k_on, float); k_off = np.asarray(k_off, float)
    k_syn = np.asarray(k_syn, float)
    M = k_on.shape[0]
    n_steps = int(t_max / dt)
    sqrt_dt = np.sqrt(dt)

    p_on0 = k_on / (k_on + k_off)
    g = (rng.random(M) < p_on0).astype(np.float64)     # DISCRETE promoter {0,1}
    m = k_syn * p_on0                                    # mRNA near its mean

    # per-step switch probabilities (exact two-state chain)
    p01 = 1.0 - np.exp(-k_on * dt)                       # off -> on
    p10 = 1.0 - np.exp(-k_off * dt)                      # on  -> off

    for _ in range(n_steps):
        p_switch = np.where(g == 0.0, p01, p10)
        flip = rng.random(M) < p_switch
        g = np.where(flip, 1.0 - g, g)                  # stays in {0,1}

        a3 = k_syn * g                                   # transcription (0 when off)
        a4 = np.clip(m, 0.0, None)                       # degradation
        dW3 = rng.standard_normal(M) * sqrt_dt
        dW4 = rng.standard_normal(M) * sqrt_dt
        dm = (a3 - a4) * dt + np.sqrt(a3) * dW3 - np.sqrt(a4) * dW4
        m = np.clip(m + dm, 0.0, None)

    return np.round(m).astype(int)
