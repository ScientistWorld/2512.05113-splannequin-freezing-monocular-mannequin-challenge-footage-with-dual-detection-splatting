#!/bin/bash
# TEMPORARY (hardening iteration 12 pilot): checks the revised scene layout and
# the defective-gradient threshold before the full audit. Restored to the
# end-to-end reference reproduction afterwards.
set -e
cd /home/user
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
python -c "import os, torch, numpy; print('cpus', len(os.sched_getaffinity(0)), 'torch', torch.__version__, 'numpy', numpy.__version__)"
python redteam/hardening_v2/pilot.py run
python redteam/hardening_v2/pilot.py summary
