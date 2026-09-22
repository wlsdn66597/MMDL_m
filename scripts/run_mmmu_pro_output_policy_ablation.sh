#!/usr/bin/env bash
# Reuse complete parser-v2 8192-token baselines; run only the two new policies.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
if [ "$#" -ne 2 ]; then
  echo "Usage: bash scripts/run_mmmu_pro_output_policy_ablation.sh NEW_OUTPUT_ROOT PARSER_V2_BASELINE_ROOT" >&2
  exit 2
fi
root="$1"
baseline_root="$2"
if [ ! -d "$baseline_root" ]; then
  echo "Baseline root not found: $baseline_root" >&2
  exit 2
fi
for setting in standard-4 standard-10 vision; do
  baseline="$baseline_root/tokens8192_${setting}"
  for required in manifest.json summary.json predictions.jsonl inputs.jsonl; do
    if [ ! -f "$baseline/$required" ]; then
      echo "Missing complete baseline file: $baseline/$required" >&2
      exit 2
    fi
  done
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
python -c 'import torch; from vllm.sampling_params import StructuredOutputsParams; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0)); print(StructuredOutputsParams(choice=["A", "B"]))'
for setting in standard-4 standard-10 vision; do
  mkdir "$root/$setting"
  for mode in constrained two-stage; do
    stage="${setting}_${mode}"
    printf 'running: %s\n' "$stage" > "$root/status.txt"
    python -u eval_output_policy.py \
      --benchmark mmmu-pro --setting "$setting" --per-subject 0 \
      --mode "$mode" --output-dir "$root/$setting/$mode" \
      --max-tokens 8192 --reasoning-tokens 1024 --final-tokens 16 \
      --max-model-len 16384 --min-pixels 1003520 --max-pixels 4014080 \
      --gpu-memory-utilization 0.85 \
      2>&1 | tee "$root/logs/$stage.log"
  done
  stage="${setting}_comparison"
  python scripts/summarize_output_policies.py "$root/$setting" \
    --baseline-free "$baseline_root/tokens8192_${setting}" \
    2>&1 | tee "$root/logs/$stage.log"
done
stage="combined_summary"
python - "$root" <<'PY' | tee "$root/summary.tsv"
import json
from pathlib import Path
import sys
root=Path(sys.argv[1])
print("setting\tmode\tn\taccuracy_pct\tunparsed\tfinal_length\tdraft_length\tmean_output_tokens\tinference_minutes")
for setting in ("standard-4","standard-10","vision"):
    data=json.loads((root/setting/"comparison.json").read_text())
    for mode in ("free","constrained","two-stage"):
        row=data["summaries"][mode]
        print(f"{setting}\t{mode}\t{row['n']}\t{100*row['accuracy']:.2f}\t{row['unparsed']}\t"
              f"{row['length_limited']}\t{row['reasoning_length_limited']}\t"
              f"{row['output_tokens']/row['n']:.1f}\t{row['inference_seconds']/60:.1f}")
PY
printf 'complete\n' > "$root/status.txt"
printf '[ALL DONE] %s/summary.tsv\n' "$root"
