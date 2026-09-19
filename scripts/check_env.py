#!/usr/bin/env python3
"""Read-only, offline environment checks. No installs, model loads or inference."""
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

MODEL_REV = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
DATA_REV = "98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68"


def main():
    failed = []

    def show(label, ok, detail):
        print(f"{'OK  ' if ok else 'FAIL'} | {label}: {detail}", flush=True)
        if not ok:
            failed.append(label)

    show("Python", sys.version_info >= (3, 10), f"{sys.version.split()[0]} ({sys.executable})")
    for name in ("torch", "transformers", "vllm", "datasets", "huggingface-hub", "Pillow", "numpy"):
        try:
            show(name, True, importlib.metadata.version(name))
        except importlib.metadata.PackageNotFoundError:
            show(name, False, "not installed in this Python environment")
    try:
        import torch
        available = torch.cuda.is_available()
        show("CUDA available", available, f"torch CUDA runtime={torch.version.cuda}")
        if available:
            gpu = torch.cuda.get_device_properties(0)
            show("Visible GPU 0", True, f"{gpu.name}, {gpu.total_memory / 2**30:.2f} GiB")
            show("BF16 support", torch.cuda.is_bf16_supported(), str(torch.cuda.is_bf16_supported()))
    except Exception as exc:
        show("torch/CUDA import", False, str(exc))
    try:
        from transformers import Qwen3VLForConditionalGeneration
        show("Qwen3-VL transformers class", True, Qwen3VLForConditionalGeneration.__name__)
    except Exception as exc:
        show("Qwen3-VL transformers class", False, str(exc))
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        hub = Path(HF_HUB_CACHE)
        print(f"INFO | HF Hub cache: {hub}")
        snapshot = hub / "models--Qwen--Qwen3-VL-4B-Instruct" / "snapshots" / MODEL_REV
        show("Model revision cached", snapshot.is_dir(), str(snapshot))
        config_file = snapshot / "config.json"
        if config_file.is_file():
            config = json.loads(config_file.read_text(encoding="utf-8"))
            show("Model architecture", config.get("model_type") == "qwen3_vl", str(config.get("model_type")))
            show("No quantization config", not config.get("quantization_config"), str(config.get("quantization_config")))
        else:
            show("Model config", False, "config.json missing")
        for name in ("tokenizer_config.json", "preprocessor_config.json"):
            show(name, (snapshot / name).is_file(), str(snapshot / name))
        index = snapshot / "model.safetensors.index.json"
        if index.is_file():
            shards = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
            missing = sorted(name for name in shards if not (snapshot / name).is_file())
            show("Model weight shards", not missing, f"{len(shards)-len(missing)}/{len(shards)} present; missing={missing}")
        else:
            show("Model weights", (snapshot / "model.safetensors").is_file(), "single-file checkpoint or index required")
        data = hub / "datasets--MMMU--MMMU" / "snapshots" / DATA_REV
        show("Dataset revision cached", data.is_dir(), str(data))
        print("INFO | Dataset snapshot presence does not verify all 900 rows; use eval_mmmu.py --check-only for that.")
    except Exception as exc:
        show("HF cache inspection", False, str(exc))
    value = os.environ.get("VLLM_USE_FLASHINFER_SAMPLER")
    if value is None:
        print("INFO | VLLM_USE_FLASHINFER_SAMPLER is unset; run_mmmu_eval.sh sets it to 0.")
    else:
        show("FlashInfer sampler workaround", value == "0", value)
    print(f"INFO | CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}")
    disk = shutil.disk_usage(Path.cwd())
    print(f"INFO | Disk free at repository: {disk.free / 2**30:.2f} GiB")
    print("INFO | nvidia-smi (read-only; includes other users' GPU processes):", flush=True)
    try:
        result = subprocess.run(["nvidia-smi"], timeout=15, check=False)
        show("nvidia-smi", result.returncode == 0, f"exit={result.returncode}")
    except Exception as exc:
        show("nvidia-smi", False, str(exc))
    print(f"\n{'CHECKS PASSED' if not failed else 'CHECKS FAILED: ' + ', '.join(failed)}")
    print("No model was loaded. No inference, downloads, package changes or driver changes were performed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
