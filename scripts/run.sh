#!/bin/bash
# Reproduce the gym's reference end to end: prepare scenes, train the paper's
# method (Splannequin on a 4DGaussians-style model) on every visible scene, and
# score it on the visible train slice and the hidden test slice.
# The hardening audit of iteration 12 is redteam/hardening_v2/audit.py
# (train | cases | summary); iteration 11's is redteam/hardening/audit.py.
set -e
cd /home/user
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
bash scripts/download.sh
bash scripts/method.sh
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
bash scripts/evaluate.sh
python validate.py --post-design --compare
