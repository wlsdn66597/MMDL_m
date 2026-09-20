#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 {standard-4|standard-10|vision} OUTPUT_DIR [extra eval arguments...]" >&2
  exit 2
fi

setting="$1"
output_dir="$2"
shift 2

case "$setting" in
  standard-4|standard-10|vision) ;;
  *) echo "Unknown setting: $setting" >&2; exit 2 ;;
esac

export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
mkdir -p logs
python -u eval_mmmu_pro.py \
  --evaluation-profile configs/mmmu_pro_v1.json \
  --setting "$setting" \
  --output-dir "$output_dir" \
  --limit 0 \
  --batch-size 1 \
  --max-tokens 2048 \
  --max-model-len 9048 \
  --min-pixels 1003520 \
  --max-pixels 4014080 \
  --gpu-memory-utilization 0.85 \
  "$@" 2>&1 | tee "logs/mmmu_pro_${setting}_$(date +%Y%m%d_%H%M%S)_$$.log"
