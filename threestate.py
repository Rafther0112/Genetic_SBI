"""
threestate.py
-------------
E10: a three-state refractory promoter (OFF <-> POISED <-> ACTIVE), transcribing only
in ACTIVE. A linear reversible chain, the standard refractory architecture in
single-cell bursting studies, and the natural next step past the telegraph model. It
has NO closed-form stationary distribution, so the exact simulator (FSP / SSA) is
genuinely expensive here, which is what lets E10 exercise the cost trade-off that the
telegraph sandbox could not (Gate G0b).

Promoter states g in {0,1,2} = {OFF, POISED, ACTIVE}. Reactions (gamma = 1):
    OFF   -> POISED    k01
    POISED-> OFF       k10
    POISED-> ACTIVE    k12
    ACTIVE-> POISED    k21
    ACTIVE-> ACTIVE+m  k_syn      (transcription, only in ACTIVE)
    m     -> 0         m          (degradation)

theta = (k01, k10, k12, k21, k_syn). Time in mRNA lifetimes.

Needs numpy + scipy.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


def prom_stationary(k01, k10, k12, k21):
    """Stationary distribution of the 3-state promoter (detailed balance on a chain)."""
    r1 = k01 / k10                       # POISED / OFF
    r2 = r1 * k12 / k21                  # ACTIVE / OFF
    w = np.array([1.0, r1, r2])
    return w / w.sum()


def exact_mean(theta):
    k01, k10, k12, k21, k_syn = theta
    return k_syn * prom_stationary(k01, k10, k12, k21)[2]   # E[m] = k_syn * P(ACTIVE)


def _choose_M(theta):
    m = exact_mean(theta); ksyn = theta[4]
    return int(min(5000, 80 + m + ksyn + 20 * np.sqrt(m + ksyn + 1.0)))


def fsp_stationary_pmf(theta, M=None):
    """Exact stationary mRNA marginal via FSP over (promoter x mRNA)."""
    k01, k10, k12, k21, k_syn = [float(v) for v in theta]
    if M is None:
        M = _choose_M(theta)
    K = M + 1
    N = 3 * K
    rows, cols, data = [], [], []

    def add(i, j, r):
        rows.append(i); cols.append(j); data.append(r)

    for m in range(K):
        i0, i1, i2 = m, K + m, 2 * K + m
        add(i0, i1, k01)                       # OFF -> POISED
        add(i1, i0, k10)                       # POISED -> OFF
        add(i1, i2, k12)                       # POISED -> ACTIVE
        add(i2, i1, k21)                       # ACTIVE -> POISED
        if m < M:
            add(i2, i2 + 1, k_syn)             # transcription (ACTIVE only)
        if m > 0:
            add(i0, i0 - 1, m); add(i1, i1 - 1, m); add(i2, i2 - 1, m)   # degradation

    Q = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tolil()
    Q.setdiag(-np.asarray(Q.sum(axis=1)).ravel())
    A = Q.transpose().tolil()
    A[0, :] = 1.0
    b = np.zeros(N); b[0] = 1.0
    pi = spsolve(A.tocsr(), b)
    pi = np.clip(pi, 0.0, None); pi /= pi.sum()
    pmf = pi[:K] + pi[K:2 * K] + pi[2 * K:]    # marginalize over promoter
    return np.arange(K), pmf


def sample_ssa(theta, n_cells, rng, t_max=40.0):
    """Exact Gillespie SSA; mRNA count at t_max per cell."""
    k01, k10, k12, k21, k_syn = [float(v) for v in theta]
    pi = prom_stationary(k01, k10, k12, k21)
    out = np.empty(n_cells, dtype=int)
    for c in range(n_cells):
        g = rng.choice(3, p=pi); m = 0; t = 0.0
        while True:
            if g == 0:
                props = [(k01, "01"), (m, "deg")]
            elif g == 1:
                props = [(k10, "10"), (k12, "12"), (m, "deg")]
            else:
                props = [(k21, "21"), (k_syn, "syn"), (m, "deg")]
            a0 = sum(p for p, _ in props)
            if a0 <= 0:
                break
            t += rng.exponential(1.0 / a0)
            if t > t_max:
                break
            r = rng.random() * a0; acc = 0.0
            for p, name in props:
                acc += p
                if r < acc:
                    if name == "01": g = 1
                    elif name == "10": g = 0
                    elif name == "12": g = 2
                    elif name == "21": g = 1
                    elif name == "syn": m += 1
                    else: m -= 1
                    break
        out[c] = m
    return out


# ----------------------------------------------------------------------
# Surrogates for the 3-state model
# ----------------------------------------------------------------------
def _prom_init(k01, k10, k12, k21, M, rng):
    r1 = k01 / k10; r2 = r1 * k12 / k21
    w = np.stack([np.ones(M), r1, r2], axis=1)
    w /= w.sum(1, keepdims=True)
    cdf = np.cumsum(w, axis=1)
    return (rng.random(M)[:, None] < cdf).argmax(axis=1)


def sample_cle_3state_batched(k01, k10, k12, k21, k_syn, rng, t_max=25.0, dt=0.02):
    """CLE that diffuses ALL THREE promoter states to a continuous simplex (the
    straw-man, now worse than the telegraph: three single-copy states diffused)."""
    k01, k10, k12, k21, k_syn = [np.asarray(v, float) for v in (k01, k10, k12, k21, k_syn)]
    M = k01.shape[0]
    pi = np.stack([prom_stationary(k01[i], k10[i], k12[i], k21[i]) for i in range(M)])
    g = pi.copy()                                   # (M,3) continuous occupancies
    m = k_syn * g[:, 2]
    n_steps = int(t_max / dt); sdt = np.sqrt(dt)

    def rt(a):                                       # sqrt of clipped propensity
        return np.sqrt(np.clip(a, 0, None))

    for _ in range(n_steps):
        a01 = k01 * g[:, 0]; a10 = k10 * g[:, 1]
        a12 = k12 * g[:, 1]; a21 = k21 * g[:, 2]
        n = rng.standard_normal((4, M)) * sdt
        d01 = a01 * dt + rt(a01) * n[0]; d10 = a10 * dt + rt(a10) * n[1]
        d12 = a12 * dt + rt(a12) * n[2]; d21 = a21 * dt + rt(a21) * n[3]
        g[:, 0] += (-d01 + d10)
        g[:, 1] += (d01 - d10 - d12 + d21)
        g[:, 2] += (d12 - d21)
        g = np.clip(g, 0.0, None); g /= g.sum(1, keepdims=True)
        a_syn = k_syn * g[:, 2]; a_deg = np.clip(m, 0, None)
        nm = rng.standard_normal((2, M)) * sdt
        m = np.clip(m + (a_syn - a_deg) * dt + rt(a_syn) * nm[0] - rt(a_deg) * nm[1], 0, None)
    return np.round(m).astype(int)


def sample_hybrid_3state_batched(k01, k10, k12, k21, k_syn, rng, t_max=25.0, dt=0.02):
    """Discrete 3-state promoter (exact switching) + CLE mRNA. Analog of the E2 hybrid."""
    k01, k10, k12, k21, k_syn = [np.asarray(v, float) for v in (k01, k10, k12, k21, k_syn)]
    M = k01.shape[0]
    g = _prom_init(k01, k10, k12, k21, M, rng)      # discrete {0,1,2}
    pi = np.stack([prom_stationary(k01[i], k10[i], k12[i], k21[i]) for i in range(M)])
    m = k_syn * pi[:, 2]
    n_steps = int(t_max / dt); sdt = np.sqrt(dt)

    for _ in range(n_steps):
        u = rng.random(M)
        # exact per-step transitions of the discrete promoter
        in0, in1, in2 = g == 0, g == 1, g == 2
        p01 = 1 - np.exp(-k01 * dt)
        p10 = 1 - np.exp(-k10 * dt); p12 = 1 - np.exp(-k12 * dt)
        p21 = 1 - np.exp(-k21 * dt)
        g = np.where(in0 & (u < p01), 1, g)
        g = np.where(in1 & (u < p10), 0, np.where(in1 & (u < p10 + p12), 2, g))
        g = np.where(in2 & (u < p21), 1, g)
        a_syn = k_syn * (g == 2); a_deg = np.clip(m, 0, None)
        nm = rng.standard_normal((2, M)) * sdt
        m = np.clip(m + (a_syn - a_deg) * dt
                    + np.sqrt(a_syn) * nm[0] - np.sqrt(a_deg) * nm[1], 0, None)
    return np.round(m).astype(int)
