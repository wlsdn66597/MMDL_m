#!/usr/bin/env bash
# Same development questions, three sequential runs on one GPU. No environment installation.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
if [ "$#" -lt 1 ]; then
  echo "Usage: bash scripts/run_output_policy_ablation.sh NEW_OUTPUT_ROOT [eval_output_policy.py options]" >&2
  exit 2
fi
root="$1"
shift
for argument in "$@"; do
  case "$argument" in
    --mode|--mode=*|--output-dir|--output-dir=*|--check-only)
      echo "Wrapper controls mode/output-dir and requires inference." >&2
      exit 2 ;;
  esac
done
mkdir -p "$(dirname "$root")"
mkdir "$root"
mkdir "$root/logs"
printf '%s\n' "$$" > "$root/runner.pid"
stage="preflight"
printf 'running: %s\n' "$stage" > "$root/status.txt"
trap 'code=$?; printf "failed: %s (exit %s)\n" "$stage" "$code" > "$root/status.txt"; exit "$code"' ERR
trap 'printf "interrupted: %s\n" "$stage" > "$root/status.txt"; exit 130' INT TERM
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
python -c 'import sys, torch; from vllm.sampling_params import StructuredOutputsParams; print("Python:", sys.executable); assert torch.cuda.is_available(), "CUDA unavailable"; print(StructuredOutputsParams(choice=["A", "B"]))'
for mode in free constrained two-stage; do
  stage="$mode"
  printf 'running: %s\n' "$stage" > "$root/status.txt"
  python -u eval_output_policy.py --mode "$mode" --output-dir "$root/$mode" "$@" \
    2>&1 | tee "$root/logs/$mode.log"
done
stage="comparison"
printf 'running: %s\n' "$stage" > "$root/status.txt"
python scripts/summarize_output_policies.py "$root" | tee "$root/logs/comparison.log"
printf 'complete\n' > "$root/status.txt"
printf '[ALL DONE] %s/comparison.md\n' "$root"
