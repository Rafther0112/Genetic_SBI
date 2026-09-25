#!/usr/bin/env bash
# launch.sh: run the last-mile experiments in parallel, one process per seed.
# Usage:  bash launch.sh telegraph      # items 3-5 on the telegraph model (no edits needed)
#         bash launch.sh threestate     # after wiring adapters.py
#         bash launch.sh dryrun         # 1-minute mock run to check the plumbing
set -u
THREADS=${THREADS:-4}          # torch threads per process; THREADS x processes <= cores
mkdir -p logs
export PYTHONUNBUFFERED=1

case "${1:-telegraph}" in
dryrun)
  export MOCK_NPE=1
  python run_crossover.py --seed 0 --n_total 400 --n_test 60 --n_cells 100 --n_post 100 --fracs 0.05 --out results_dryrun/crossover
  python run_placement.py --seed 0 --n_total 400 --n_test 60 --n_cells 100 --n_post 100 --out results_dryrun/placement
  python run_diag_cost.py --seed 0 --Ns 200 400 --n_cells 100 --surrogates nb --out results_dryrun/diag_cost
  echo "dry run OK (results_dryrun/ is mock output, delete it)";;
telegraph)
  for s in 0 1 2 3 4; do   # item 5: NB placement, five seeds from one implementation
    nohup python run_placement.py --system telegraph --surrogate nb --f 0.25 --seed $s --threads $THREADS > logs/placement_s$s.log 2>&1 &
  done
  for s in 0 1 2 3 4; do   # item 4: crossover sweep, 0.02 and 0.05 are overlap checks
    nohup python run_crossover.py --system telegraph --surrogate lna --fracs 0.0025 0.005 0.01 0.02 0.05 --seed $s --threads $THREADS > logs/crossover_s$s.log 2>&1 &
  done
  for s in 0 1 2; do       # item 3: diagnostic cost curve (CPU, sklearn)
    nohup python run_diag_cost.py --system telegraph --surrogates nb lna cle --seed $s > logs/diag_tel_s$s.log 2>&1 &
  done
  echo "launched; follow with: tail -f logs/*.log";;
threestate)
  for s in 0 1 2; do
    nohup python run_diag_cost.py --system threestate --surrogates hybrid nb cle --seed $s > logs/diag_3s_s$s.log 2>&1 &
    nohup python run_crossover.py --system threestate --surrogate nb --target k_syn --fracs 0.005 0.01 0.02 0.05 --seed $s --threads $THREADS > logs/crossover3s_s$s.log 2>&1 &
  done
  echo "launched; follow with: tail -f logs/*.log";;
esac
