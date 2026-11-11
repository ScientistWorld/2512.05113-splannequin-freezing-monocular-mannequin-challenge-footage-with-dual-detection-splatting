#!/bin/bash
set -euo pipefail
cd /home/user
python redteam/audit.py --case shortcut_failed_method_baseline_only
