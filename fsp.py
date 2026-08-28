"""
fsp.py
------
Finite State Projection (FSP) for the telegraph model: build the truncated CME
generator on the joint (promoter, mRNA) state space and solve for the stationary
distribution, then read off the mRNA marginal.

This is a second EXACT method (besides the Poisson-Beta closed form). It matters
for two things:
  * the cost table (review W2): FSP is the general-purpose exact solver one would
    actually use when no closed form exists, so its wall-clock is the honest
    "exact" cost to compare against the CLE.
  * E10: the second system will have no Poisson-Beta form, and FSP / SSA become the
    only exact references.

States are indexed  idx = g * (M+1) + m  with promoter g in {0,1} and mRNA m in
{0..M}. Reactions (gamma = 1):
    off -> on          k_on
    on  -> off         k_off
    on  -> on + mRNA   k_syn
    mRNA -> 0          m

Needs numpy + scipy.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


def _choose_M(a, b, lam):
    mean = lam * a / (a + b)
    s = a + b
    var = mean + lam ** 2 * a * b / (s ** 2 * (s + 1.0))
    M = int(60 + mean + lam + 15 * np.sqrt(var + 1.0))
    return int(min(M, 4000))


def fsp_stationary_pmf(k_on, k_off, k_syn, M=None):
    """Return (m_values, pmf) of the exact stationary mRNA marginal via FSP."""
    a, b, lam = float(k_on), float(k_off), float(k_syn)
    if M is None:
        M = _choose_M(a, b, lam)
    K = M + 1
    N = 2 * K
    rows, cols, data = [], [], []

    def add(i, j, r):
        rows.append(i); cols.append(j); data.append(r)

    for m in range(K):
        i0 = m           # g = 0
        i1 = K + m       # g = 1
        add(i0, i1, a)                      # off -> on
        add(i1, i0, b)                      # on -> off
        if m < M:
            add(i1, i1 + 1, lam)           # transcription (only while on)
        if m > 0:
            add(i0, i0 - 1, m)             # degradation
            add(i1, i1 - 1, m)             # degradation

    Q = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tolil()
    Q.setdiag(-np.asarray(Q.sum(axis=1)).ravel())

    # stationary distribution: pi Q = 0, i.e. Q^T pi = 0, with sum(pi) = 1.
    A = Q.transpose().tolil()
    A[0, :] = 1.0                          # replace one balance eq with normalization
    bvec = np.zeros(N); bvec[0] = 1.0
    pi = spsolve(A.tocsr(), bvec)
    pi = np.clip(pi, 0.0, None)
    pi = pi / pi.sum()

    pmf = pi[:K] + pi[K:]                  # marginalize over the promoter state
    return np.arange(K), pmf


def sample_fsp(theta, n_cells, rng, M=None):
    """Draw n_cells exact mRNA counts by sampling the FSP stationary marginal."""
    m_vals, pmf = fsp_stationary_pmf(*theta, M=M)
    pmf = np.clip(pmf, 0.0, None); pmf /= pmf.sum()
    return rng.choice(m_vals, size=n_cells, p=pmf)
