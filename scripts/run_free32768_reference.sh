#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
python -u scripts/run_free_reproduction.py "$@"
