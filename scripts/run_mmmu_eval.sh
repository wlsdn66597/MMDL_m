#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
mkdir -p logs
# Use the already activated MMDL virtualenv; never install or change packages here.
python -u eval_mmmu.py "$@" 2>&1 | tee "logs/eval_$(date +%Y%m%d_%H%M%S)_$$.log"
