#!/bin/bash
# Hidden test-slice evaluator. Runs with no arguments and scores every method
# directory on disk (checkpoints/<method>/predictions/) on the private test
# scenes held in eval/test/. Any stale score file is removed first; a method
# that fails (missing/invalid outputs, over budget) gets worst values.
set -e
cd /home/user
CHECKPOINTS_DIR="${1:-/home/user/checkpoints}"
mkdir -p scoring
rm -f scoring/scores_test.json
python eval/test/evaluate.py --checkpoints "$CHECKPOINTS_DIR" --out scoring/scores_test.json
