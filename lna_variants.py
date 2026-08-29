"""
lna_variants.py
---------------
E1 (review W3): decompose the LNA to separate Gaussianity from the non-negativity
enforcement. All four draw the same Gaussian with the exact telegraph mean and
variance and differ only in how they map it to counts:

    unclamped : raw real-valued sample            (no round, no clamp)
    rounded   : round to nearest integer          (allows negatives; isolates discretization)
    truncated : resample the negative tail to >=0  (isolates truncation)
    clamped   : round then clip to >=0  == the LNA used in the main experiments
                (piles the negative mass as a delta at zero: maximal moment distortion)

If the unclamped LNA calibrates in the moments-only setting while the clamped one
does not, the moments-only lesson is about clamping, not Gaussianity. In the
full-feature setting every Gaussian variant still fails, because a Gaussian cannot
produce zero-inflation or a super-Poissonian tail, which is Gaussianity proper.

Needs numpy + scipy. Same (M,) -> (M,) interface as telegraph.sample_*_batched.
"""

import numpy as np
from scipy.stats import truncnorm


def _moments(k_on, k_off, k_syn):
    a = np.asarray(k_on, float) + np.asarray(k_off, float)
    mean = np.asarray(k_syn, float) * np.asarray(k_on, float) / a
    var = mean + np.asarray(k_syn, float) ** 2 * np.asarray(k_on, float) * \
        np.asarray(k_off, float) / (a ** 2 * (a + 1.0))
    return mean, np.clip(var, 1e-12, None)


def sample_lna_unclamped_batched(k_on, k_off, k_syn, rng):
    mean, var = _moments(k_on, k_off, k_syn)
    return rng.normal(mean, np.sqrt(var))                      # raw real values


def sample_lna_rounded_batched(k_on, k_off, k_syn, rng):
    mean, var = _moments(k_on, k_off, k_syn)
    return np.round(rng.normal(mean, np.sqrt(var)))            # integer, may be < 0


def sample_lna_truncated_batched(k_on, k_off, k_syn, rng):
    mean, var = _moments(k_on, k_off, k_syn)
    sd = np.sqrt(var)
    a = (0.0 - mean) / sd                                       # lower bound at 0
    return truncnorm.rvs(a, np.inf, loc=mean, scale=sd, random_state=rng)


def sample_lna_clamped_batched(k_on, k_off, k_syn, rng):
    mean, var = _moments(k_on, k_off, k_syn)
    return np.clip(np.round(rng.normal(mean, np.sqrt(var))), 0, None)


VARIANTS = {
    "lna_unclamped": sample_lna_unclamped_batched,
    "lna_rounded": sample_lna_rounded_batched,
    "lna_truncated": sample_lna_truncated_batched,
    "lna_clamped": sample_lna_clamped_batched,
}
