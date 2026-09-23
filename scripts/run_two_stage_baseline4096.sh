#!/usr/bin/env bash
# Four complete sequential runs; never run multiple vLLM engines concurrently.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 NEW_OUTPUT_ROOT [--resume] [--model-path PATH] [--model-revision REV] [--mmmu-data-root PATH] [--pro-data-root PATH]" >&2
  exit 2
fi
root="$1"
shift
resume=0
model="Qwen/Qwen3-VL-4B-Instruct"
revision="ebb281ec70b05090aa6165b016eac8ec08e71b17"
mmmu_data="MMMU/MMMU"
pro_data="MMMU/MMMU_Pro"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --resume) resume=1; shift ;;
    --model-path|--model-revision|--mmmu-data-root|--pro-data-root)
      [ "$#" -ge 2 ] || { echo "Missing value for $1" >&2; exit 2; }
      case "$1" in
        --model-path) model="$2" ;;
        --model-revision) revision="$2" ;;
        --mmmu-data-root) mmmu_data="$2" ;;
        --pro-data-root) pro_data="$2" ;;
      esac
      shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
if [ -e "$root" ] && [ "$resume" -ne 1 ]; then
  echo "Output root exists. Use a new path, or --resume to validate and skip completed runs." >&2
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
python scripts/summarize_two_stage_baseline.py "$root" --initialize "$model" "$revision" "$mmmu_data" "$pro_data"
common=(--mode two-stage --evaluation-profile configs/two_stage4096_v1.json
        --model-path "$model" --model-revision "$revision")
names=(mmmu_val standard-4 standard-10 vision)
for name in "${names[@]}"; do
  if [ "$name" = mmmu_val ]; then
    dataset=(--benchmark mmmu-val --setting standard --data-root "$mmmu_data")
  else
    dataset=(--benchmark mmmu-pro --setting "$name" --data-root "$pro_data")
  fi
  if [ -e "$root/checks/$name" ]; then
    python scripts/summarize_two_stage_baseline.py "$root" --run "$name" --preflight
  else
    python -u eval_output_policy.py "${common[@]}" "${dataset[@]}" --check-only \
      --output-dir "$root/checks/$name" 2>&1 | tee "$root/logs/check_${name}.log"
  fi
done
# All 6090 input records have been checked before the first GPU run.
python scripts/summarize_two_stage_baseline.py "$root" --preflight
for name in "${names[@]}"; do
  if [ "$name" = mmmu_val ]; then
    dataset=(--benchmark mmmu-val --setting standard --data-root "$mmmu_data")
  else
    dataset=(--benchmark mmmu-pro --setting "$name" --data-root "$pro_data")
  fi
  if [ -e "$root/$name" ]; then
    python scripts/summarize_two_stage_baseline.py "$root" --run "$name"
    echo "[skip] Validated complete run: $name"
  else
    python -u eval_output_policy.py "${common[@]}" "${dataset[@]}" \
      --checked-inputs "$root/checks/$name" --output-dir "$root/$name" 2>&1 | tee "$root/logs/${name}.log"
  fi
done
python scripts/summarize_two_stage_baseline.py "$root"
printf 'complete\n' > "$root/status.txt"
echo "[done] $root/baseline_summary.md"
