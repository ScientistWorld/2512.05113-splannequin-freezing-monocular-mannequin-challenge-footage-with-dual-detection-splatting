#!/bin/bash
# Score the visible train split and held-out test split against all
# checkpoint prediction artifacts.

set -e

cd /home/user

bash scripts/evaluate_train.sh /home/user/checkpoints
bash scripts/evaluate_test.sh /home/user/checkpoints
