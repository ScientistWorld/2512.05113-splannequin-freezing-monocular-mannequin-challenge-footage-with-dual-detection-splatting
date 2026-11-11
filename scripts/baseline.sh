#!/bin/bash
# Run a comparison method from the paper on every visible scene.
#   bash scripts/baseline.sh            # 4DGaussians without Splannequin (dynamic baseline)
#   bash scripts/baseline.sh static     # static 3DGS-style reconstruction (no timestamp control)
# Writes checkpoints/<name>/predictions/. Such directories are scored like any
# other method on disk (under their directory name; `baseline` -> 4dgaussians).
set -e
cd /home/user
NAME="${1:-baseline}"
VARIANT="$NAME" OUT="checkpoints/$NAME" bash scripts/method.sh
