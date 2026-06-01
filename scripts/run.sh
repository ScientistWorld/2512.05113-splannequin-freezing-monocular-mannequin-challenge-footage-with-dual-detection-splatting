#!/bin/bash
# Regenerate the scaled reproduction artifacts, then score the visible
# train split and held-out test split against the produced predictions.

set -e

cd /home/user

bash scripts/reproduce.sh
bash scripts/evaluate_train.sh /home/user/checkpoints
bash scripts/evaluate_test.sh /home/user/checkpoints
