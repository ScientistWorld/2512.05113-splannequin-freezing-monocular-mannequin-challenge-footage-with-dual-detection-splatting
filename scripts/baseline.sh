#!/bin/bash
# Run a comparison method from the paper on every visible scene.
#   bash scripts/baseline.sh            # 4DGaussians without Splannequin (dynamic baseline)
#   bash scripts/baseline.sh static     # static 3DGS-style reconstruction (no timestamp control)
# Writes checkpoints/<name>/predictions/. These directories are scored like any
# other method on disk.
set -e
cd /home/user
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
VARIANT="${1:-baseline}"
mkdir -p logs
pids=()
for split in train test; do
  for exp in synthetic_validation_paper synthetic_large_motion; do
    python method/splannequin.py --variant "$VARIANT" --scenes data/$split/$exp/*.json \
      --out-dir "checkpoints/$VARIANT" > "logs/${VARIANT}_${split}_${exp}.log" 2>&1 &
    pids+=($!)
  done
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit $status
