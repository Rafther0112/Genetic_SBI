#!/usr/bin/env bash
# run_all.sh -- pipeline completo E1-E4. Ejecutar desde la raiz del repo del paper,
# con exp_common.py y los run_*.py copiados ahi.
#
#   bash run_all.sh e4        # indice de localizacion, telegrafo (barato, sin sbi)
#   bash run_all.sh lr        # barrido de tasa de afinado para E1
#   bash run_all.sh e1 1e-4   # E1 completo con el lr elegido
#   bash run_all.sh e2        # cruce sesgo-varianza
#   bash run_all.sh e4b       # indice de localizacion, tres estados (decide si va E3)
#   bash run_all.sh e3        # asignacion guiada en tres estados
#   bash run_all.sh sum       # resumenes de todo lo corrido
set -e
mkdir -p results logs
STAGE=${1:-e4}; LR=${2:-1e-4}

case $STAGE in
  e4)
    for s in cme lna nb cle; do
      python run_localization.py --system telegraph --exact cme --surrogate $s \
        --seeds 0 1 2 2>&1 | tee -a logs/e4.log
    done
    python summarize.py results/e4_localization.jsonl ;;

  lr)
    python run_twostage.py --f 0.25 --seeds 0 --lr 3e-4 1e-4 3e-5 2>&1 | tee -a logs/e1_lr.log
    python summarize.py results/e1_twostage.jsonl ;;

  e1)
    python run_twostage.py --f 0.25 0.50 0.75 --seeds 0 1 2 3 4 --lr $LR 2>&1 | tee -a logs/e1.log
    python summarize.py results/e1_twostage.jsonl ;;

  e2)
    python run_crossover.py --f 0.02 0.04 0.08 0.16 0.25 0.50 \
      --seeds 0 1 2 --with_twostage 2>&1 | tee -a logs/e2.log
    python summarize.py results/e2_crossover.jsonl ;;

  e4b)
    for s in nb cle; do
      python run_localization.py --system threestate --exact fsp --surrogate $s \
        --seeds 0 1 2 2>&1 | tee -a logs/e4b.log
    done
    python summarize.py results/e4_localization.jsonl ;;

  e3)
    python run_alloc3state.py --f 0.25 0.50 --seeds 0 1 2 --surrogate nb  2>&1 | tee -a logs/e3.log
    python run_alloc3state.py --f 0.25 0.50 --seeds 0 1 2 --surrogate cle 2>&1 | tee -a logs/e3.log
    python summarize.py results/e3_alloc3state.jsonl ;;

  sum)
    python summarize.py results/e4_localization.jsonl results/e1_twostage.jsonl \
      results/e2_crossover.jsonl results/e3_alloc3state.jsonl ;;

  *) echo "etapa desconocida: $STAGE"; exit 1 ;;
esac
