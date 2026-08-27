"""
metrics.py
----------
Distances between an NPE posterior and the gold-standard reference posterior for
the same observation (used in E4, and as detector labels in E8).

  c2st(X, Y) : classifier two-sample test accuracy. 0.5 = indistinguishable,
               1.0 = perfectly separable. Small MLP with 5-fold CV, following the
               convention in the SBI benchmark (Lueckmann et al. 2021).
  mmd_rbf(X, Y) : unbiased squared MMD with an RBF kernel (median-heuristic
                  bandwidth). 0 = identical distributions.

Both operate in log10 parameter space, the space of both NPE and reference samples.
"""

import numpy as np
from scipy.spatial.distance import cdist, pdist
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score


def c2st(X, Y, seed=0, n_folds=5, hidden=(16, 16)):
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    Z = np.vstack([X, Y])
    y = np.r_[np.zeros(len(X)), np.ones(len(Y))]
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-8)
    clf = MLPClassifier(hidden_layer_sizes=hidden, max_iter=1000,
                        early_stopping=True, random_state=seed)
    return float(cross_val_score(clf, Z, y, cv=n_folds, scoring="accuracy").mean())


def mmd_rbf(X, Y, gamma=None):
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    if gamma is None:
        med = np.median(pdist(np.vstack([X, Y])))
        gamma = 1.0 / (2.0 * med ** 2 + 1e-12)
    Kxx = np.exp(-gamma * cdist(X, X, "sqeuclidean"))
    Kyy = np.exp(-gamma * cdist(Y, Y, "sqeuclidean"))
    Kxy = np.exp(-gamma * cdist(X, Y, "sqeuclidean"))
    m, n = len(X), len(Y)
    sxx = (Kxx.sum() - np.trace(Kxx)) / (m * (m - 1))
    syy = (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1))
    return float(sxx + syy - 2.0 * Kxy.mean())
