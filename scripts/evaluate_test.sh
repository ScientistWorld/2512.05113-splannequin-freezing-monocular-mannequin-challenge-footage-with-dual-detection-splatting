#!/bin/bash
# Held-out test-slice evaluator for final anti-shortcut scoring.
#
# It scores the same method entries that were written by the train slice,
# using eval/test/ and writing scoring/scores_test.json.

set -e

cd /home/user

CHECKPOINTS_DIR="${1:-/home/user/checkpoints}"

mkdir -p scoring

python eval/test/evaluate.py \
  --data data/synthetic_mannequin.json \
  --checkpoints "$CHECKPOINTS_DIR" \
  --train-scores scoring/scores_train.json \
  --out scoring/scores_test.json
