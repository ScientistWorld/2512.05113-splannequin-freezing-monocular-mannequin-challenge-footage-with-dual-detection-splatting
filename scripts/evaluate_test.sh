#!/bin/bash
# Held-out test-slice evaluator for final anti-shortcut scoring.
#
# It scores the same method entries that were written by the train slice,
# using eval/test/ and writing scoring/scores_test.json.

set -e

cd /home/user

mkdir -p scoring

python eval/test/evaluate.py \
  --data data/synthetic_mannequin.json \
  --checkpoints checkpoints \
  --train-scores scoring/scores_train.json \
  --out scoring/scores_test.json
