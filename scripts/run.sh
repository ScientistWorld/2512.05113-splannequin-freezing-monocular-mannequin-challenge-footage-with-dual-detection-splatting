#!/bin/bash
# Re-score the latest generated reproduction artifacts after evaluator changes,
# then score the visible train split and held-out test split.

set -e

cd /home/user

echo "[run] evaluating aggregate synthetic validation scores"
bash scripts/evaluate.sh
echo "[run] evaluating visible train split"
bash scripts/evaluate_train.sh /home/user/checkpoints
echo "[run] evaluating held-out test split"
bash scripts/evaluate_test.sh /home/user/checkpoints
echo "[run] validating scoring structure and metric spread"
python validate.py --compare
