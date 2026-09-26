#!/usr/bin/env bash
# Uniform full-coverage MMMU validation candidate. Run on one GPU, sequentially.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$#" -ne 5 ] || [ "$2" != "--model-path" ] || [ "$4" != "--data-root" ]; then
  echo "Usage: $0 NEW_OUTPUT_ROOT --model-path Qwen/Qwen3-VL-4B-Instruct --data-root MMMU/MMMU" >&2
  exit 2
fi
root="$1"
model="$3"
data="$5"
if [ -e "$root" ]; then
  echo "Output root already exists: $root" >&2
  exit 2
fi
mkdir -p "$root/checks" "$root/logs"
exec 9>"$root/.lock"
flock -n 9 || { echo "Another process is using this output root" >&2; exit 2; }
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
printf 'running\n' > "$root/status.txt"
trap 'rc=$?; printf "failed exit=%s\n" "$rc" > "$root/status.txt"; exit "$rc"' ERR

common=(--mode two-stage --evaluation-profile configs/two_stage8192_val_v1.json
        --benchmark mmmu-val --setting standard
        --model-path "$model" --data-root "$data")
python -u eval_output_policy.py "${common[@]}" --check-only \
  --output-dir "$root/checks/mmmu_val" 2>&1 | tee "$root/logs/check_mmmu_val.log"
python -u eval_output_policy.py "${common[@]}" \
  --checked-inputs "$root/checks/mmmu_val" --output-dir "$root/mmmu_val" \
  2>&1 | tee "$root/logs/mmmu_val.log"
python - "$root/mmmu_val" <<'PY'
import sys
from baseline_contract import validate_run
_, summary, _ = validate_run(sys.argv[1], "mmmu-val", "standard", require_base=True)
print(f"[validated] {summary['n']} questions; accuracy={summary['accuracy']:.2%}; "
      f"draft_truncated={summary['reasoning_length_limited']}; unparsed={summary['unparsed']}")
PY
printf 'complete\n' > "$root/status.txt"
echo "[done] $root/mmmu_val/report_draft.md"
