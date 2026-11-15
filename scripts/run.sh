#!/bin/bash
# Hardening iteration 12, final verification job (restored afterwards to the
# plain end-to-end reference reproduction): seed-matched repeats (seeds 1-2) of
# the best honest variant (time smoothness x10) and of the no-defective
# ablation; every case is rescored with the hidden evaluator and the summary is
# recomputed with the final scoring/reference.json.
set -e
cd /home/user
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
python redteam/hardening_v2/audit.py train
python redteam/hardening_v2/audit.py cases
python redteam/hardening_v2/audit.py summary
bash scripts/evaluate_test.sh
