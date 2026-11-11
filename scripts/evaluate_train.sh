#!/bin/bash
# Visible train-slice evaluator. Scores every method directory on disk
# (checkpoints/<method>/predictions/) on the public training scenes in
# data/train/ and writes scoring/scores_train.json plus a per-scene
# breakdown in scoring/diagnostics_train.json.
set -e
cd /home/user
CHECKPOINTS_DIR="${1:-/home/user/checkpoints}"
mkdir -p scoring
rm -f scoring/scores_train.json
python eval/train/evaluate.py --checkpoints "$CHECKPOINTS_DIR" --out scoring/scores_train.json
