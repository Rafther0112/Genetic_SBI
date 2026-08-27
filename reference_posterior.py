"""
reference_posterior.py
----------------------
P0-5: gold-standard posteriors from the EXACT telegraph likelihood.

The telegraph stationary law is Poisson-Beta, whose marginal PMF has closed form
(Peccoud & Ycart 1995; Raj et al. 2006):

    P(m=n | a, b, lam) = lam^n / n!  *  B(a+n, b)/B(a, b)  *  1F1(a+n; a+b+n; -lam)

with a = k_on, b = k_off, lam = k_syn (gamma = 1). We evaluate log P(n) stably via
Kummer's transform 1F1(a+n; a+b+n; -lam) = e^{-lam} 1F1(b; a+b+n; lam), whose
argument is positive, so the series has only positive terms (no cancellation) and
1F1 >= 1 (its log is finite and non-negative).

Because the likelihood is tractable, we can obtain the TRUE posterior for each test
snapshot by MCMC (emcee, gradient-free) and use it as the reference against which any
NPE posterior is scored with C2ST / MMD. This is far more informative than coverage
alone (workplan P0-5, review W5).

Note on interpretation: the reference conditions on the FULL 500-cell snapshot, while
an NPE conditions on the 6 summaries. C2ST(NPE, reference) therefore blends NPE
approximation error, summary-statistic information loss, and (for a surrogate-trained
NPE) misspecification. The CME-trained NPE is the control that carries everything
EXCEPT misspecification, so the surrogate-vs-CME gap against the same reference still
isolates the misspecification effect.

Needs numpy + scipy + emcee (no torch / sbi). Runs on CPU.
"""

import numpy as np
from scipy.special import gammaln, hyp1f1

import emcee

# prior box on log10 rates, matching sbi_experiment.py
LOW = np.array([-2.0, -1.0, 0.0])
HIGH = np.array([1.3, 1.3, 2.3])


# ----------------------------------------------------------------------
# Exact log-PMF of the telegraph stationary distribution
# ----------------------------------------------------------------------
def telegraph_logpmf(n, k_on, k_off, k_syn):
    """log P(m=n) for scalar rates; n is an int scalar or array."""
    n = np.asarray(n, dtype=np.float64)
    a, b, lam = float(k_on), float(k_off), float(k_syn)
    logM = np.log(hyp1f1(b, a + b + n, lam))          # Kummer form, positive args
    return (n * np.log(lam) - gammaln(n + 1.0)
            + gammaln(a + n) + gammaln(a + b) - gammaln(a + b + n) - gammaln(a)
            - lam + logM)


def _logpmf_grid(uniq, a, b, lam):
    """log P(uniq | rates) broadcast over walkers.

    uniq : (1, U) float counts;  a, b, lam : (W, 1) rate columns  ->  (W, U)
    """
    n = uniq
    logM = np.log(hyp1f1(b, a + b + n, lam))
    return (n * np.log(lam) - gammaln(n + 1.0)
            + gammaln(a + n) + gammaln(a + b) - gammaln(a + b + n) - gammaln(a)
            - lam + logM)


# ----------------------------------------------------------------------
# Vectorized log-posterior for a single snapshot (flat prior on the box)
# ----------------------------------------------------------------------
def make_log_prob(counts):
    """Return a vectorized log-posterior over log10 params for one snapshot."""
    uniq, mult = np.unique(np.asarray(counts), return_counts=True)
    uniq = uniq.astype(np.float64)[None, :]           # (1, U)
    mult = mult.astype(np.float64)                    # (U,)

    def log_prob(theta):                              # theta: (W, 3) log10 params
        theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
        inbox = np.all((theta >= LOW) & (theta <= HIGH), axis=1)
        rates = 10.0 ** theta
        a = rates[:, 0:1]; b = rates[:, 1:2]; lam = rates[:, 2:3]
        lp = _logpmf_grid(uniq, a, b, lam) @ mult     # (W,)
        lp = np.where(inbox & np.isfinite(lp), lp, -np.inf)
        return lp

    return log_prob


# ----------------------------------------------------------------------
# Reference posterior via emcee
# ----------------------------------------------------------------------
def sample_reference(counts, n_samples=2000, nwalkers=32,
                     burn=1000, thin=5, seed=0):
    """Gold-standard posterior samples (in log10 space) for one snapshot."""
    rng = np.random.default_rng(seed)
    ndim = 3
    steps = burn + thin * int(np.ceil(n_samples / nwalkers)) + thin
    p0 = rng.uniform(LOW, HIGH, size=(nwalkers, ndim))
    sampler = emcee.EnsembleSampler(nwalkers, ndim, make_log_prob(counts),
                                    vectorize=True)
    sampler.run_mcmc(p0, steps, progress=False)
    chain = sampler.get_chain(discard=burn, thin=thin, flat=True)
    if len(chain) > n_samples:
        chain = chain[rng.choice(len(chain), n_samples, replace=False)]
    return chain
