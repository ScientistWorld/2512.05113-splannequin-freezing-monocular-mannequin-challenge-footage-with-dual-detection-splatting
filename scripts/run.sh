#!/bin/bash
# Regenerate the scaled reproduction artifacts, then score the visible
# train split and held-out test split against the produced predictions.

set -e

cd /home/user

echo "[run] regenerating data and method outputs"
bash scripts/reproduce.sh
echo "[run] evaluating visible train split"
bash scripts/evaluate_train.sh /home/user/checkpoints
echo "[run] evaluating held-out test split"
bash scripts/evaluate_test.sh /home/user/checkpoints
echo "[run] validating scoring structure and metric spread"
python validate.py --compare
