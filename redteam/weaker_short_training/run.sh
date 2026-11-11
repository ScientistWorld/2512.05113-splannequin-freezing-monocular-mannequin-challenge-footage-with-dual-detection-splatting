#!/bin/bash
set -euo pipefail
cd /home/user
python redteam/audit.py --case weaker_short_training
