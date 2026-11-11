#!/bin/bash
set -euo pipefail
cd /home/user
python redteam/audit.py --case shortcut_infinite_predictions
