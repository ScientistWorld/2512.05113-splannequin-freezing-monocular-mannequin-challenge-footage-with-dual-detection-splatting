#!/bin/bash
# Run the paper's method (Splannequin on a 4DGaussians-style dynamic Gaussian
# model) on every visible scene: public training scenes and the test-scene
# inputs. Writes checkpoints/splannequin/predictions/<experiment>/<scene_id>.json.
# Each batch of scenes trains in its own single-threaded process.
set -e
cd /home/user
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p logs
pids=()
for split in train test; do
  for exp in synthetic_validation_paper synthetic_large_motion; do
    python method/splannequin.py --variant splannequin --seed "${SEED:-0}" \
      --scenes data/$split/$exp/*.json --out-dir checkpoints/splannequin \
      > "logs/method_${split}_${exp}.log" 2>&1 &
    pids+=($!)
  done
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
tail -n 2 logs/method_*.log
exit $status
