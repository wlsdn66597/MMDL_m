#!/usr/bin/env bash
# Launch once under nohup; six full evaluations run sequentially on one GPU.
set -Eeuo pipefail
cd "$(dirname "$0")/.."

if [ "$#" -ne 1 ]; then
  echo "Usage: bash scripts/run_mmmu_pro_token_sweep.sh NEW_OUTPUT_ROOT" >&2
  exit 2
fi
root="$1"
# Atomic creation rejects existing runs, including partial ones.
mkdir -p "$(dirname "$root")"
mkdir "$root"
mkdir "$root/logs" "$root/reports"
printf '%s\n' "$$" > "$root/runner.pid"
printf 'running\n' > "$root/status.txt"
stage="preflight"
trap 'code=$?; printf "failed: %s (exit %s)\n" "$stage" "$code" > "$root/status.txt"; exit "$code"' ERR
trap 'printf "interrupted: %s\n" "$stage" > "$root/status.txt"; exit 130' INT TERM

export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1

python -c 'import sys, torch, vllm, datasets; print("Python:", sys.executable); assert torch.cuda.is_available(), "CUDA unavailable"; print("GPU:", torch.cuda.get_device_name(0))'

for tokens in 4096 8192; do
  for setting in standard-4 standard-10 vision; do
    stage="tokens${tokens}_${setting}"
    printf 'running: %s\n' "$stage" > "$root/status.txt"
    printf '[START] %s %s\n' "$(date -Is)" "$stage"
    python -u eval_mmmu_pro.py \
      --evaluation-profile "configs/mmmu_pro_tokens${tokens}_v1.json" \
      --setting "$setting" \
      --output-dir "$root/$stage" \
      --limit 0 --batch-size 1 \
      --max-tokens "$tokens" --max-model-len 16384 \
      --min-pixels 1003520 --max-pixels 4014080 \
      --gpu-memory-utilization 0.85 \
      2>&1 | tee "$root/logs/$stage.log"
    printf '[DONE] %s %s\n' "$(date -Is)" "$stage"
  done
  stage="report_tokens${tokens}"
  python scripts/summarize_mmmu_pro.py \
    --standard-4 "$root/tokens${tokens}_standard-4" \
    --standard-10 "$root/tokens${tokens}_standard-10" \
    --vision "$root/tokens${tokens}_vision" \
    --output-prefix "$root/reports/tokens${tokens}"
done

stage="budget_comparison"
for setting in standard-4 standard-10 vision; do
  python scripts/compare_runs.py \
    "$root/tokens4096_${setting}" "$root/tokens8192_${setting}" \
    --label-a tokens4096 --label-b tokens8192 \
    --allow-config-differences \
    --output "$root/reports/tokens4096_vs_8192_${setting}.md"
done

python - "$root" <<'PY' | tee "$root/reports/generation_summary.tsv"
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
print("tokens\tsetting\tn\taccuracy_pct\tunparsed\tlength_limited\tminutes")
for tokens in (4096, 8192):
    for setting in ("standard-4", "standard-10", "vision"):
        folder = root / f"tokens{tokens}_{setting}"
        s = json.loads((folder / "summary.json").read_text())
        m = json.loads((folder / "manifest.json").read_text())
        if m["status"] != "complete" or s["n"] != 1730:
            raise RuntimeError(f"Incomplete run: {folder}")
        print(f"{tokens}\t{setting}\t{s['n']}\t{s['accuracy']*100:.2f}\t"
              f"{s['unparsed']}\t{s['length_limited']}\t{s['total_seconds']/60:.1f}")
PY
printf 'complete\n' > "$root/status.txt"
printf '[ALL DONE] %s Reports: %s/reports\n' "$(date -Is)" "$root"
