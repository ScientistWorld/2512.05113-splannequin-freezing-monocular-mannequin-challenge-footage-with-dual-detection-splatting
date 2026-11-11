#!/bin/bash
# Run the paper's method (Splannequin on a 4DGaussians-style dynamic Gaussian
# model) on every visible scene: the public training scenes and the test-scene
# inputs. Writes checkpoints/splannequin/predictions/<experiment>/<scene_id>.json.
# Scenes are trained in batches of 8 (sorted order) in single-threaded
# processes; each scene's parameters and random schedules are independent.
set -e
cd /home/user
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
VARIANT="${VARIANT:-splannequin}"
OUT="${OUT:-checkpoints/splannequin}"
mkdir -p logs
pids=()
for split in train test; do
  for exp in synthetic_validation_paper synthetic_large_motion; do
    files=(data/$split/$exp/*.json)
    for ((i = 0; i < ${#files[@]}; i += 8)); do
      python method/splannequin.py --variant "$VARIANT" --seed "${SEED:-0}" \
        --scenes "${files[@]:i:8}" --out-dir "$OUT" \
        > "logs/${VARIANT}_${split}_${exp}_$((i / 8)).log" 2>&1 &
      pids+=($!)
    done
  done
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
tail -qn 1 logs/${VARIANT}_*.log
exit $status
