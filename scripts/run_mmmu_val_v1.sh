#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Canonical 900-question evaluation. Conflicting overrides are rejected by eval_mmmu.py.
exec bash scripts/run_mmmu_eval.sh \
  --evaluation-profile configs/mmmu_val_v1.json \
  --limit-per-subject 0 \
  --batch-size 1 \
  --prompt-style direct \
  --max-tokens 256 \
  --max-model-len 8192 \
  --min-pixels 1003520 \
  --max-pixels 4014080 \
  --gpu-memory-utilization 0.85 \
  "$@"
