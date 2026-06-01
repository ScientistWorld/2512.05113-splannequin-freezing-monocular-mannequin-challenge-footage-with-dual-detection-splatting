#!/bin/bash
# Evaluation script — the standard way to evaluate work in this environment.
#
# This script must be SELF-CONTAINED using only code in eval/.
# Do NOT import from method/ — evaluate.sh must work even when the
# paper's method code is hidden from a scientist agent.
#
# Design it so that anyone following the same output format can run
# this script to evaluate their results.
#
# OUTPUT CONTRACT: This script MUST write scoring/scores.json containing
# a JSON dict nested by experiment, e.g.:
#   {"imagenet_classification": {"top1_accuracy": 82.1, "top5_accuracy": 94.8}}
# Experiment and metric names must match the keys in scoring/reference.json.
# This file is read by the system for automated comparison.
#
# Example usage:
#   ./evaluate.sh
#   ./evaluate.sh checkpoints/baseline/predictions.json checkpoints/splannequin/predictions.json
#
# Called from run.sh after training, or independently.

set -e

cd /home/user

mkdir -p scoring

if [ "$#" -gt 0 ]; then
  INPUTS=("$@")
else
  INPUTS=(
    checkpoints/baseline/predictions.json
    checkpoints/splannequin/predictions.json
  )
  [ ! -f checkpoints/no_hidden/predictions.json ] || INPUTS+=(checkpoints/no_hidden/predictions.json)
  [ ! -f checkpoints/no_defective/predictions.json ] || INPUTS+=(checkpoints/no_defective/predictions.json)
fi

NAMES=()
for input in "${INPUTS[@]}"; do
  parent="$(basename "$(dirname "$input")")"
  case "$parent" in
    baseline) NAMES+=(baseline_dynamic_gs) ;;
    no_hidden) NAMES+=(splannequin_no_hidden) ;;
    no_defective) NAMES+=(splannequin_no_defective) ;;
    *) NAMES+=("$parent") ;;
  esac
done

python eval/evaluate_freeze.py \
  --data data/synthetic_mannequin.json \
  --inputs "${INPUTS[@]}" \
  --names "${NAMES[@]}" \
  --out scoring/scores.json
