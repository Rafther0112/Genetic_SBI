"""
adapters.py
-----------
The ONLY file you need to edit: it connects the last-mile scripts to your existing
three-state code (threestate.py, fsp.py, hybrid.py). The telegraph model needs nothing here.

Fill in `simulate_threestate` so that it returns exactly the (N, 6) summaries your
three-state runs used for Table 3 / Table 6. Keep the parameter order below; if your code
uses a different order, permute inside the function.
"""

import numpy as np

# theta = (log10 k01, log10 k10, log10 k12, log10 k21, log10 k_syn), Appendix B ranges
THREESTATE = dict(
    names=["k01", "k10", "k12", "k21", "k_syn"],
    low=np.array([-2.0, -1.0, -2.0, -1.0, 0.0]),
    high=np.array([1.3, 1.3, 1.3, 1.3, 2.3]),
)


def simulate_threestate(theta_log, sim, n_cells, rng):
    """
    theta_log : (N, 5) array, log10 rates in the order above
    sim       : 'cme' (exact, FSP), 'nb', 'cle', or 'hybrid'
    returns   : (N, 6) summary array, same summary_stats as the paper

    Example wiring (adapt the names to your modules):

        from threestate import simulate_batch_threestate
        return simulate_batch_threestate(theta_log, sim, n_cells, rng)

    or, if your functions return raw counts (N, n_cells):

        from threestate import sample_fsp, sample_nb, sample_cle, sample_hybrid
        from telegraph import summary_stats
        fn = dict(cme=sample_fsp, nb=sample_nb, cle=sample_cle, hybrid=sample_hybrid)[sim]
        counts = fn(10.0 ** theta_log, n_cells, rng)
        return np.stack([summary_stats(c) for c in counts])
    """
    raise NotImplementedError(
        "Wire adapters.simulate_threestate to your three-state simulators first.")
