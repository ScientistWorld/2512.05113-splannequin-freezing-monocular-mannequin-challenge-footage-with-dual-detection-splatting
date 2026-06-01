#!/bin/bash
# Download all data and models needed for this environment.
#
# Run on a node with internet access BEFORE building the container
# or submitting compute jobs.
#
# This script should be idempotent — safe to run multiple times.
# It should skip files that already exist.
#
# Update the storage estimate below as you add downloads.
#
# Estimated total download size: 0 GB
# Estimated disk usage after extraction: <1 MB
#
# This benchmark uses a tiny procedural dataset, so no external fetches
# are required. The command below is deterministic and safe to rerun.

set -e

cd /home/user

python data/synthetic_mannequin.py --out data/synthetic_mannequin.json
