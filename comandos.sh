Para correr Run_cost

python run_cost.py --n_theta 200 --n_cells 500
python run_cost.py --n_theta 500 --ssa_theta 64 --ssa_cells 200

Para correr reconcile

python run_reconcile.py --n_sims 8000 --n_cells 500 --n_test 2000 --seeds 0
python run_reconcile.py --n_sims 8000 --n_cells 500 --n_test 2000 --seeds 0 1 2 3 4

Para correr Reference
python run_reference.py --seed 0 --n_test 400 --n_ref 400 --n_samples 2000 --workers 16


python run_controls.py --n_sims 8000 --n_cells 500 --n_test 2000 --seeds 0 1 2