"""
boxflow.py
----------
E3 (review W6): guarantee that a trained flow cannot place posterior mass outside the
prior box, by reparametrizing theta.

Instead of learning q(theta | x) on the bounded box [low, high] and sampling directly
(which lets a misspecified flow leak mass outside the box, up to ~58% for the CLE), we
map theta to an unbounded space with a logit and learn the flow there:

    u = logit((theta - low) / (high - low))            in R^3   (train the flow on u)
    theta = low + (high - low) * sigmoid(u)            back in the box  (sample)

By construction every sample lands strictly inside the box, so leakage is zero. The
scientific point of E3 is then honest: if the surrogate coverage does NOT improve once
the support is constrained, the leakage was a symptom of misspecification, not its
cause, and constraining the support does not rescue the surrogate (answers W6/Q3).

Needs numpy (torch versions provided for use inside the sbi pipeline).
"""

import numpy as np

LOW = np.array([-2.0, -1.0, 0.0])
HIGH = np.array([1.3, 1.3, 2.3])
_EPS = 1e-6


def to_unbounded(theta_log, low=LOW, high=HIGH):
    """box -> R^3 (logit of min-max scaled theta)."""
    z = (np.asarray(theta_log, float) - low) / (high - low)
    z = np.clip(z, _EPS, 1.0 - _EPS)
    return np.log(z / (1.0 - z))


def to_box(u, low=LOW, high=HIGH):
    """R^3 -> box (sigmoid then rescale). Output always strictly inside [low, high]."""
    s = 1.0 / (1.0 + np.exp(-np.asarray(u, float)))
    return low + (high - low) * s


# --- torch versions (for training/sampling inside the sbi flow) ---
def to_unbounded_torch(theta_log, low, high):
    import torch
    z = (theta_log - low) / (high - low)
    z = torch.clamp(z, _EPS, 1.0 - _EPS)
    return torch.log(z / (1.0 - z))


def to_box_torch(u, low, high):
    import torch
    return low + (high - low) * torch.sigmoid(u)
