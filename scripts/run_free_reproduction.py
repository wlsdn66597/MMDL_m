#!/usr/bin/env python3
"""Run the four complete reference-prompt evaluations sequentially on one GPU."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import eval_mmmu as val
import eval_mmmu_pro as pro
from baseline_contract import RUNS
from eval_free_reproduction import code_hashes, profile
from scripts.summarize_free_reproduction import summarize_suite


def execute(command, logfile):
    print("[command] " + " ".join(command), flush=True)
    with logfile.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", bufsize=1)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            if process.wait():
                raise RuntimeError(f"Evaluation failed; see {logfile}")
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--model-path", default=val.MODEL)
    parser.add_argument("--model-revision", default=val.MODEL_REV)
    parser.add_argument("--mmmu-data-root", default="MMMU/MMMU")
    parser.add_argument("--pro-data-root", default=pro.DATASET)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if root.exists() and not args.resume:
        parser.error("Output root exists; use --resume or a new output path")
    root.mkdir(parents=True, exist_ok=True)
    # Linux GPU runner: release the lock automatically on process exit.
    import fcntl
    with (root / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("This suite is already running")
        request = {k: getattr(args, k) for k in ("model_path", "model_revision", "mmmu_data_root", "pro_data_root")}
        request.update(profile=profile(), code_sha256=code_hashes())
        request_path = root / "suite_request.json"
        if request_path.exists() and json.loads(request_path.read_text(encoding="utf-8")) != request:
            parser.error("Cannot resume with different code/model/data/profile")
        val.write_json(request_path, request)
        (root / "logs").mkdir(exist_ok=True)
        status = root / "status.txt"
        phase = "initialization"
        try:
            for check_only in (True, False):
                if not check_only and args.preflight_only:
                    status.write_text("inputs_checked_no_inference\n", encoding="utf-8")
                    return
                for name, benchmark, setting in RUNS:
                    phase = ("check_" if check_only else "infer_") + name
                    status.write_text(f"running {phase}\n", encoding="utf-8")
                    out = root / "checks" / name if check_only else root / name
                    command = [sys.executable, "-u", str(ROOT / "eval_free_reproduction.py"),
                               "--benchmark", benchmark, "--setting", setting,
                               "--model-path", args.model_path, "--model-revision", args.model_revision,
                               "--data-root", args.mmmu_data_root if benchmark == "mmmu-val" else args.pro_data_root,
                               "--output-dir", str(out)]
                    if check_only:
                        command.append("--check-only")
                    if args.resume:
                        command.append("--resume")
                    execute(command, root / "logs" / f"{phase}.log")
            phase = "summary"
            summarize_suite(root)
            status.write_text("complete\n", encoding="utf-8")
            print(f"[done] {root / 'baseline_summary.md'}", flush=True)
        except BaseException:
            status.write_text(f"failed {phase}\n", encoding="utf-8")
            raise


if __name__ == "__main__":
    main()
