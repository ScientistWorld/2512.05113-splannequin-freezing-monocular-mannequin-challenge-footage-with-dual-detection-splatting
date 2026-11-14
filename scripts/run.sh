#!/bin/bash
# Hardening iteration 12 audit job, run 2 (restored afterwards to the plain
# end-to-end reference reproduction): adaptive density control restored and
# the anchor held fixed. Scenes are already generated (idempotent steps).
set -e
cd /home/user
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
python -c "import os, torch, numpy; print('cpus', len(os.sched_getaffinity(0)), 'torch', torch.__version__, 'numpy', numpy.__version__)"
python eval/test/make_test_scenes.py
bash scripts/download.sh
# Smoke test: every variant through the densification / anchoring code paths in seconds.
SMOKE="--iterations 160 --coarse-iterations 20 --densify-from 5 --densify-until 120 --densify-interval 10 --opacity-reset-interval 50 --reg-start 40 --l1-start 100 --log-every 80"
for v in splannequin baseline static no_hidden no_defective; do
  python method/splannequin.py --variant $v $SMOKE --out-dir tmp/smoke_run2/$v --snap-out-dir tmp/smoke_run2/${v}_snap \
    --scenes data/train/synthetic_validation_paper/scene_00.json data/train/synthetic_validation_paper/scene_01.json
done
python method/splannequin.py --variant splannequin --anchor-canonical-grad $SMOKE --out-dir tmp/smoke_run2/canon \
  --scenes data/train/synthetic_validation_paper/scene_00.json
python redteam/hardening_v2/audit.py train
python redteam/hardening_v2/audit.py cases
python redteam/hardening_v2/audit.py summary
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
bash scripts/evaluate.sh
python validate.py --post-design --compare || echo "validate.py reported problems (reference.json is updated after this job)"
