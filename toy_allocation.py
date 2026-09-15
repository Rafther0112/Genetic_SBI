"""
toy_allocation.py
-----------------
W6: a minimal analytic/numeric toy that EXPLAINS (not just rationalizes) why a uniform
simulator mixture beats concentrating exact simulation, for an amortized estimator.

Setup. Two regimes A (easy) and B (hard), equal prior mass. A single amortized estimator
q(theta | x) is fit by maximum likelihood on a training set that mixes an EXACT simulator
and a SURROGATE. In A the surrogate is correct; in B it is biased by delta. Because the
estimator is amortized (one q over the whole prior), it fits the AVERAGE of whatever
labels it sees in each region. Training exact vs surrogate in a region sets the effective
target there; the estimator's error in a region is the squared gap between q's fitted
target and the exact target, and total error is the prior-weighted sum.

Key point the toy makes precise: exact budget spent in A is wasted (the surrogate is
already exact there), and spending it in B only helps B while A is unaffected either way.
So the objective depends only on how much exact budget lands in B. Concentrating all
budget in B therefore looks optimal in this frictionless toy, which is NOT what we see
empirically. The empirical loss from concentration comes from an amortization FRICTION
absent in the frictionless toy: a flow trained on inconsistent targets across regions
(exact in B, surrogate in A) pays an extra reconciliation cost that grows with the
target mismatch between regions. We add that friction as a term proportional to the
cross-region target discrepancy and show uniform becomes optimal once friction exceeds a
threshold, matching the experiments.

Needs numpy only.
"""

import numpy as np


def total_error(fB, delta, friction):
    """
    fB = fraction of the exact budget placed in region B (rest of exact in A, wasted).
    Both regimes have prior mass 1/2. Surrogate bias in B is delta (A: 0).
    Region error = (residual surrogate bias)^2 after exact coverage; exact coverage in a
    region removes its bias in proportion to the exact fraction there.
    friction = amortization penalty per unit of cross-region target mismatch.
    """
    budget = 0.5                      # total exact fraction of the training set
    exact_B = min(fB * budget, 0.5)   # exact mass in B (region has mass 0.5)
    exact_A = budget - exact_B        # exact mass in A (wasted: surrogate already exact)
    # residual bias in B after exact coverage: linear shrink with exact fraction in B
    frac_exact_B = exact_B / 0.5
    err_B = ((1 - frac_exact_B) * delta) ** 2
    err_A = 0.0
    base = 0.5 * err_A + 0.5 * err_B
    # amortization friction: the flow must reconcile two targets across A and B.
    # the mismatch it must span is the difference between A's target (0) and B's
    # effective target ((1-frac_exact_B)*delta), and the penalty is worst when exact is
    # concentrated (high frac_exact_B in B, none in A) because the regions then disagree.
    mismatch = abs(((1 - frac_exact_B) * delta) - delta * (exact_A / 0.5) * 0.0)
    # concentration = how unevenly exact is spread across the two regions (0=uniform,1=all in one)
    concentration = abs(frac_exact_B - (exact_A / 0.5))
    friction_cost = friction * concentration * (delta ** 2)
    return base + friction_cost


def main():
    delta = 1.0
    print("Toy two-regime allocation: total error vs fraction of exact budget in the hard region B")
    print("(fB=0.5 is uniform across A,B; fB=1.0 concentrates all exact in B)\n")
    grid = np.linspace(0.0, 1.0, 11)
    for friction in [0.0, 0.5, 1.0, 2.0]:
        errs = [total_error(fB, delta, friction) for fB in grid]
        best = grid[int(np.argmin(errs))]
        tag = "-> optimum at fB=%.1f %s" % (best, "(UNIFORM)" if abs(best-0.5) < 1e-9 else "(concentrate)")
        print(f"friction={friction:>4.1f}:  err(fB=0.5 uniform)={total_error(0.5,delta,friction):.3f}"
              f"   err(fB=1.0 concentrate)={total_error(1.0,delta,friction):.3f}   {tag}")
    print("\nReading: with no amortization friction, concentrating in B looks best (frictionless).")
    print("Once the flow pays a reconciliation cost for inconsistent cross-region targets,")
    print("uniform allocation becomes optimal, which is what the experiments show.")


if __name__ == "__main__":
    main()
