#!/bin/bash
# Iteration-11 hardening job: build scenes, smoke-test, train the reference and
# all audit variants, score every case through the hidden evaluator, then
# regenerate the reference score files with the no-argument scripts.
set -e
cd /home/user
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
python -c "import os, torch, numpy; print('cpus', len(os.sched_getaffinity(0)), 'torch', torch.__version__, 'numpy', numpy.__version__)"
# Remove previous-format artifacts so only new-format method directories remain.
rm -rf checkpoints/baseline checkpoints/no_hidden checkpoints/no_defective checkpoints/extra_seed_11 checkpoints/extra_seed_29 checkpoints/splannequin/predictions.json
echo "[run] scenes"
python data/scenes.py --out-root data/train
python eval/test/make_test_scenes.py
echo "[run] smoke"
bash redteam/hardening/smoke.sh
echo "[run] train"
python redteam/hardening/audit.py train || echo "[run] some training tasks failed"
echo "[run] cases"
python redteam/hardening/audit.py cases || echo "[run] case scoring failed"
echo "[run] summary"
python redteam/hardening/audit.py summary || true
echo "[run] reference scores via the no-argument scripts"
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
bash scripts/evaluate.sh
cat scoring/scores_test.json
python validate.py --post-design --compare || true
