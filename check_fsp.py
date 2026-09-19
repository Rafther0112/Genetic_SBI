"""
check_fsp.py
------------
The one number the paper still needs from code: how closely the FSP stationary
distribution matches the closed-form Poisson-Beta pmf of Eq. (2). Reports, over
parameters drawn from the prior, the maximum total-variation distance and the maximum
relative error in the mean and the Fano factor, plus the truncation M the adaptive rule
chose. Fills the two "tolerance" markers (Section 3 and Appendix D).

Uses fsp.fsp_stationary_pmf and the closed form, so it checks your code, not a copy.
No torch/sbi needed.   Usage:
    python check_fsp.py --n_grid 200
"""
import argparse
import numpy as np
from scipy.special import gammaln, hyp1f1

from fsp import fsp_stationary_pmf
from sbi_experiment import make_prior, theta_to_rates


def pb_pmf(kon, koff, ksyn, n):
    """Poisson-Beta pmf of Eq. (2), in log space via Kummer's transform."""
    n = np.asarray(n)
    logB1 = gammaln(kon + n) + gammaln(koff) - gammaln(kon + koff + n)
    logB0 = gammaln(kon) + gammaln(koff) - gammaln(kon + koff)
    log_p = n * np.log(ksyn) - gammaln(n + 1.0) + logB1 - logB0
    # 1F1(a;c;-x) = e^{-x} 1F1(c-a;c;x), positive-term series
    a, c = kon + n, kon + koff + n
    log_p = log_p - ksyn + np.log(np.maximum(hyp1f1(c - a, c, ksyn), 1e-300))
    return np.exp(log_p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_grid", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    th = theta_to_rates(make_prior().sample((a.n_grid,)).numpy())
    tv, e_mean, e_fano, Ms, tails = [], [], [], [], []
    for kon, koff, ksyn in th:
        m_vals, pmf = fsp_stationary_pmf(kon, koff, ksyn)
        m_vals = np.asarray(m_vals); pmf = np.asarray(pmf, float)
        exact = pb_pmf(kon, koff, ksyn, m_vals)
        tv.append(0.5 * np.abs(pmf - exact).sum())
        mu = (pmf * m_vals).sum(); var = (pmf * m_vals ** 2).sum() - mu ** 2
        mu_x = ksyn * kon / (kon + koff)
        s = kon + koff
        f_x = 1 + ksyn * koff / (s * (s + 1.0))
        e_mean.append(abs(mu - mu_x) / mu_x)
        e_fano.append(abs(var / mu - f_x) / f_x)
        Ms.append(m_vals[-1]); tails.append(pmf[-1])

    for name, v in [("total-variation distance to Eq. (2)", tv),
                    ("relative error in the mean", e_mean),
                    ("relative error in the Fano factor", e_fano),
                    ("truncated tail mass", tails)]:
        v = np.array(v)
        print(f"  {name:38s} max {v.max():.2e}   median {np.median(v):.2e}")
    Ms = np.array(Ms)
    print(f"  {'adaptive truncation M':38s} min {Ms.min():.0f}  median {np.median(Ms):.0f}  max {Ms.max():.0f}")
    print(f"\n  -> paper: 'the FSP probabilities match Eq. (2) to a total-variation distance")
    print(f"     below {np.max(tv):.0e} over {a.n_grid} parameter draws from the prior'")


if __name__ == "__main__":
    main()
