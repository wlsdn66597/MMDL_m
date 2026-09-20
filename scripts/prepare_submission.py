#!/usr/bin/env python3
"""Copy a complete 900-question run into the repository's submission layout."""
import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_mmmu import evaluation_signature, load_evaluation_profile, validate_evaluation_profile
from types import SimpleNamespace


ARTIFACTS = (
    "summary.json", "manifest.json", "predictions.jsonl", "inputs.jsonl",
    "sampling_params.txt", "chat_template.txt", "environment.txt", "requirements.freeze.txt",
    "evaluation_profile.json",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--name", default="mmmu_val_baseline")
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    repo = Path(__file__).resolve().parents[1]
    summary_path = result_dir / "summary.json"
    report_path = result_dir / "report_draft.md"
    manifest_path = result_dir / "manifest.json"
    if not summary_path.is_file() or not report_path.is_file() or not manifest_path.is_file():
        raise SystemExit("The result directory must contain summary.json, manifest.json and report_draft.md")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("n") != 900 or not summary.get("complete_900"):
        raise SystemExit("Refusing to prepare a submission from an incomplete run")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    profile, _, profile_hash = load_evaluation_profile(repo / "configs" / "mmmu_val_v1.json")
    try:
        validate_evaluation_profile(profile, SimpleNamespace(**manifest.get("arguments", {})))
    except ValueError as exc:
        raise SystemExit(f"Refusing a run outside the fixed evaluation profile: {exc}") from exc
    recorded = manifest.get("evaluation_profile") or {}
    signature = manifest.get("evaluation_signature") or {}
    expected_signature = evaluation_signature(SimpleNamespace(**manifest["arguments"]), profile_hash)
    if recorded.get("sha256") != profile_hash or signature != expected_signature:
        raise SystemExit("Refusing an unsigned/legacy run: rerun with scripts/run_mmmu_val_v1.sh")
    reports = repo / "reports"
    destination = repo / "artifacts" / args.name
    reports.mkdir(exist_ok=True)
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copy2(report_path, reports / "mmmu_baseline.md")
    copied = []
    for name in ARTIFACTS:
        source = result_dir / name
        if source.is_file():
            shutil.copy2(source, destination / name)
            copied.append(name)
    assignment = repo / "assignment"
    assignment.mkdir(exist_ok=True)
    (assignment / "assignment1.md").write_text(
        "# Assignment 1\n\nThe current submission is [reports/mmmu_baseline.md](../reports/mmmu_baseline.md).\n",
        encoding="utf-8",
    )
    print(f"Prepared reports/mmmu_baseline.md and artifacts/{args.name}/")
    print("Copied: " + ", ".join(copied))
    print("Complete the TODO fields and gap analysis before committing.")


if __name__ == "__main__":
    main()
