#!/bin/bash
# Standard evaluation entry point (visible scenes only; eval/ never imports method/).
#
# Reads frozen-state predictions from checkpoints/<method>/predictions/<experiment>/<scene_id>.json
# for the public training scenes and writes scoring/scores.json (same content
# and format as scoring/scores_train.json).
#
# Usage: bash scripts/evaluate.sh [checkpoints_dir]
set -e
cd /home/user
CHECKPOINTS_DIR="${1:-/home/user/checkpoints}"
mkdir -p scoring
python eval/train/evaluate.py --checkpoints "$CHECKPOINTS_DIR" --out scoring/scores.json --details scoring/diagnostics_train.json
