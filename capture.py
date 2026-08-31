"""
capture.py
----------
E11 groundwork: a capture-efficiency (binomial thinning) layer for scRNA-seq.

Each true mRNA molecule is detected independently with probability p, so the observed
count is  obs ~ Binomial(true, p). Applied on top of any simulator, this makes the
training pipeline  simulator -> capture -> summaries  match the real generative
process, so an NPE trained with it targets the observed-count posterior.

The key consequence for the diagnostic: capture pushes the Fano factor toward 1,
    Fano_obs = 1 + p * (Fano_true - 1),
so a bursty gene looks less bursty than it is, and dropout inflates the zero fraction.
The pre-registered Fano* is therefore NOT directly comparable across capture levels;
with a known/estimated p one can de-bias via  Fano_true = 1 + (Fano_obs - 1) / p, or
re-calibrate the threshold on captured data. This is the honest caveat E11 must state.

Needs numpy.
"""

import numpy as np


def apply_capture(counts, p, rng):
    """obs ~ Binomial(true, p). p is a scalar or a per-cell array matching counts."""
    counts = np.asarray(counts)
    p = np.asarray(p, float)
    return rng.binomial(counts, p)


def fano_observed(fano_true, p):
    """Exact map of the Fano factor under binomial thinning."""
    return 1.0 + p * (fano_true - 1.0)


def fano_debias(fano_obs, p):
    """Recover the true Fano from the observed one when p is known."""
    return 1.0 + (fano_obs - 1.0) / p
