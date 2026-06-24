#!/usr/bin/env bash
set -euo pipefail

python train/run_experiment.py --config configs/base.yaml "$@"
