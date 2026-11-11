#!/bin/bash
# Reproduce the reference end to end: build the public scenes, run the paper's
# method on all visible scenes, and score the visible train slice.
# (The hidden test slice is scored separately by scripts/evaluate_test.sh.)
set -e
cd /home/user
bash scripts/download.sh
bash scripts/method.sh
bash scripts/evaluate.sh
