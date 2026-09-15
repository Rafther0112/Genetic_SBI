"""
check_moments.py
----------------
Verifies Eq. (4) / Appendix C: for a promoter with generator Q transcribing only in state
A (mRNA degradation rate 1),

    E[m]    = ksyn * pi_A
    Var[m]  = E[m] + ksyn^2 * ( pi_A * [(I - Q)^-1]_AA  -  pi_A^2 )

against (i) the telegraph closed form and (ii) FSP moments of the three-state promoter,
on a grid drawn from the paper's prior. Prints the maximum relative difference, which is
the number Appendix C asks for, and times the 3x3 solve for the cost table.

Uses threestate.fsp_stationary_pmf if importable, otherwise its own sparse FSP.
No torch/sbi needed.   Usage:
    python check_moments.py --n_grid 200 --M 600
"""
import argparse, time
import numpy as np
from scipy.linalg import inv
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def promoter_Q(k01, k10, k12, k21):
    return np.array([[-k01, k01, 0.], [k10, -(k10 + k12), k12], [0., k21, -k21]])


def moments_from_Q(Q, ksyn, iA):
    n = Q.shape[0]
    M = np.vstack([Q.T[:-1], np.ones(n)]); b = np.zeros(n); b[-1] = 1.0
    pi = np.linalg.solve(M, b)
    R = inv(np.eye(n) - Q)
    mean = ksyn * pi[iA]
    var = mean + ksyn ** 2 * (pi[iA] * R[iA, iA] - pi[iA] ** 2)
    return mean, var


def fsp_pmf(th, M):
    k01, k10, k12, k21, ksyn = th
    Qp = promoter_Q(k01, k10, k12, k21)
    N = 3 * (M + 1); idx = lambda g, m: g * (M + 1) + m
    rows, cols, vals = [], [], []
    def add(i, j, r): rows.append(j); cols.append(i); vals.append(r)
    for g in range(3):
        for m in range(M + 1):
            i = idx(g, m); tot = 0.0
            for h in range(3):
                if h != g and Qp[g, h] > 0:
                    add(i, idx(h, m), Qp[g, h]); tot += Qp[g, h]
            if g == 2 and m < M:
                add(i, idx(g, m + 1), ksyn); tot += ksyn
            if m > 0:
                add(i, idx(g, m - 1), float(m)); tot += m
            add(i, i, -tot)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N)).tolil()
    A[0, :] = 1.0; b = np.zeros(N); b[0] = 1.0
    p = np.clip(spla.spsolve(A.tocsr(), b), 0, None); p /= p.sum()
    return p.reshape(3, M + 1).sum(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_grid", type=int, default=200)
    ap.add_argument("--M", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    print("=== telegraph check (Eq. 4 vs the closed form of Section 3) ===")
    rng = np.random.default_rng(a.seed)
    err = 0.0
    for _ in range(2000):
        kon = 10 ** rng.uniform(-2, 1.3); koff = 10 ** rng.uniform(-1, 1.3); ksyn = 10 ** rng.uniform(0, 2.3)
        m, v = moments_from_Q(np.array([[-kon, kon], [koff, -koff]]), ksyn, 1)
        s = kon + koff
        F_exact = 1 + ksyn * koff / (s * (s + 1))
        err = max(err, abs(v / m - F_exact) / F_exact)
    print(f"  max relative difference in Fano over 2000 prior draws: {err:.2e}")

    print(f"\n=== three-state check (Eq. 4 vs FSP, M={a.M}) ===")
    worst = 0.0; worst_th = None; tails = []
    for _ in range(a.n_grid):
        th = (10 ** rng.uniform(-2, 1.3), 10 ** rng.uniform(-1, 1.3),
              10 ** rng.uniform(-2, 1.3), 10 ** rng.uniform(-1, 1.3), 10 ** rng.uniform(0, 2.3))
        pm = fsp_pmf(th, a.M); ms = np.arange(a.M + 1)
        mean = (pm * ms).sum(); var = (pm * ms ** 2).sum() - mean ** 2
        m1, v1 = moments_from_Q(promoter_Q(*th[:4]), th[4], 2)
        d = max(abs(m1 - mean) / mean, abs(v1 / m1 - var / mean) / (var / mean))
        tails.append(pm[-1])
        if d > worst: worst, worst_th = d, th
    print(f"  max relative difference (mean or Fano): {worst:.2e}")
    print(f"  at theta = {tuple(round(x, 4) for x in worst_th)}")
    print(f"  max FSP truncation tail mass: {max(tails):.2e}   (raise --M if above ~1e-8)")
    print(f"\n  -> Appendix C: 'agrees with the FSP moments to {worst:.0e} relative difference")
    print(f"     over {a.n_grid} parameter draws from the prior (FSP truncation M = {a.M})'")

    t = time.perf_counter()
    for _ in range(8000):
        moments_from_Q(promoter_Q(1., 1., 1., 1.), 40., 2)
    print(f"\n  cost of the NB moments for 8,000 parameters: {time.perf_counter()-t:.2f} s (Table 5)")


if __name__ == "__main__":
    main()
