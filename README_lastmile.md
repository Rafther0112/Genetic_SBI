# Last-mile experiments (ICLR 2027)

Copy these files into the root of `Genetic_SBI` (next to `telegraph.py` and
`sbi_experiment.py`; the scripts import your versions of both). Do not copy any
telegraph.py from elsewhere.

| file | item | what it does |
|---|---|---|
| `run_placement.py` | 5 | NB placement at f = 0.25, six schemes, one seed per process |
| `run_crossover.py` | 4 | mixture vs exact-only down to 20 exact simulations |
| `run_diag_cost.py` | 3 | classifier TV and uncovered mass vs number of simulations |
| `collect.py` | all | paired mean ± std over seeds, ready for the paper |
| `common.py` | all | prior, simulation, NPE (Appendix B settings), metrics |
| `adapters.py` | three-state only | the one file to edit: wire your three-state simulators |
| `launch.sh` | all | parallel launcher (`dryrun`, `telegraph`, `threestate`) |

## Steps
1. `pip install scikit-learn` if it is missing (for the diagnostic).
2. `bash launch.sh dryrun` (about a minute, mock NPE) to check the plumbing, then `rm -rf results_dryrun`.
3. `THREADS=4 bash launch.sh telegraph` launches 13 processes. Keep THREADS x 13 at or below the node's cores.
4. Optional: wire `adapters.simulate_threestate`, then `bash launch.sh threestate`.
5. Tomorrow: `python collect.py placement`, `python collect.py crossover`, `python collect.py diag`.

## Sanity checks before using any number
- Crossover, f = 0.02 (160 exact): exact-only about 0.91, mixture about 0.70 (paper values).
- Placement, seeds 0-1: fano_soft2 about +0.012, inverse about -0.057 in distance to nominal.
  If they differ, report the five seeds from this script, not a mix with the old runs.
- Diagnostic, N = 12000: telegraph NB about 0.37, LNA 0.94, CLE 0.98 (Table 6). If not,
  match `--n_bins`, `--eps` and the classifier to your original implementation first.
- `collect.py` warns if any file came from a mock run.

Test sets use `test_seed = 1000 + seed`. If your paper runs used a different mapping, pass
`--test_seed` so the new numbers share the paper's test sets.
