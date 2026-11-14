#!/bin/bash
# Hardening iteration 12 audit job (restored afterwards to the plain
# end-to-end reference reproduction):
#   1. regenerate the public training scenes (crowded layout) and the hidden
#      test scenes (maintainer step, private seeds in eval/test/);
#   2. train the reference (seed 0) on every visible scene, plus seeds 1-2,
#      the 4DGaussians baseline, both ablations, better variants and static;
#   3. score every case with the unmodified hidden evaluator;
#   4. regenerate scoring/scores_{train,test}.json from the reference alone.
set -e
cd /home/user
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
python -c "import os, torch, numpy; print('cpus', len(os.sched_getaffinity(0)), 'torch', torch.__version__, 'numpy', numpy.__version__)"
python eval/test/make_test_scenes.py
bash scripts/download.sh
python redteam/hardening_v2/audit.py train
python redteam/hardening_v2/audit.py cases
python redteam/hardening_v2/audit.py summary
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
bash scripts/evaluate.sh
python validate.py --post-design --compare || echo "validate.py reported problems (reference.json is updated after this job)"
