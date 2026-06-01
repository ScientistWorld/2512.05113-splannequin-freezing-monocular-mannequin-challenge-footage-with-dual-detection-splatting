#!/bin/bash
# Run the paper's proposed method.
#
# This script trains/runs the paper's method using code in method/.
#
# Usage:
#   bash scripts/method.sh

set -e

cd /home/user

python data/synthetic_mannequin.py --out data/synthetic_mannequin.json
python method/splannequin_1d.py \
  --variant splannequin \
  --data data/synthetic_mannequin.json \
  --out checkpoints/splannequin/predictions.json \
  --steps "${STEPS:-900}"
python method/splannequin_1d.py \
  --variant no_hidden \
  --data data/synthetic_mannequin.json \
  --out checkpoints/no_hidden/predictions.json \
  --steps "${STEPS:-900}"
python method/splannequin_1d.py \
  --variant no_defective \
  --data data/synthetic_mannequin.json \
  --out checkpoints/no_defective/predictions.json \
  --steps "${STEPS:-900}"

for seed in 11 29; do
  scene="extra_seed_${seed}"
  python method/splannequin_1d.py \
    --variant baseline \
    --data "data/${scene}.json" \
    --out "checkpoints/${scene}/baseline/predictions.json" \
    --steps "${EXTRA_STEPS:-${STEPS:-900}}"
  python method/splannequin_1d.py \
    --variant splannequin \
    --data "data/${scene}.json" \
    --out "checkpoints/${scene}/splannequin/predictions.json" \
    --steps "${EXTRA_STEPS:-${STEPS:-900}}"
  python method/splannequin_1d.py \
    --variant no_hidden \
    --data "data/${scene}.json" \
    --out "checkpoints/${scene}/no_hidden/predictions.json" \
    --steps "${EXTRA_STEPS:-${STEPS:-900}}"
  python method/splannequin_1d.py \
    --variant no_defective \
    --data "data/${scene}.json" \
    --out "checkpoints/${scene}/no_defective/predictions.json" \
    --steps "${EXTRA_STEPS:-${STEPS:-900}}"
done

bash scripts/evaluate.sh
