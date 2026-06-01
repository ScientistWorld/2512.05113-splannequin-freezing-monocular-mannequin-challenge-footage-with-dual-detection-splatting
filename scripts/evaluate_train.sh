#!/bin/bash
# Visible train-slice evaluator for future improvement agents.
#
# Mirrors scripts/evaluate.sh's standalone contract: read pre-generated
# checkpoint predictions, stay method-agnostic, and write a scoring JSON.

set -e

cd /home/user

mkdir -p scoring

python eval/train/evaluate.py \
  --data data/synthetic_mannequin.json \
  --checkpoints checkpoints \
  --out scoring/scores_train.json
