#!/usr/bin/env python3
"""Validate an existing run manifest against a fixed evaluation profile."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_mmmu import load_evaluation_profile, validate_evaluation_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--profile", type=Path, default=Path("configs/mmmu_val_v1.json"))
    args = parser.parse_args()
    manifest_path = args.run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    profile, profile_path, profile_hash = load_evaluation_profile(args.profile)
    try:
        validate_evaluation_profile(profile, SimpleNamespace(**manifest.get("arguments", {})))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    expected_manifest = {
        "mc_parser_policy": profile["mc_parser_policy"],
        "data_revision": profile["dataset_revision"],
        "parser_revision": profile["parser_revision"],
    }
    mismatches = [
        f"manifest.{key}: expected {expected!r}, got {manifest.get(key)!r}"
        for key, expected in expected_manifest.items() if manifest.get(key) != expected
    ]
    if manifest.get("recipe") != profile["sampling_recipe"]:
        mismatches.append("manifest.recipe differs from profile.sampling_recipe")
    if mismatches:
        raise SystemExit("Run manifest mismatch:\n  " + "\n  ".join(mismatches))
    recorded = manifest.get("evaluation_profile") or {}
    if recorded.get("sha256") == profile_hash:
        print(f"OK: {args.run_dir} has the signed {profile['profile_name']} profile ({profile_hash})")
    else:
        print(f"OK (legacy): recorded arguments/revisions/recipe match {profile['profile_name']} ({profile_hash})")
        print("The run predates v1 signatures; use scripts/run_mmmu_val_v1.sh for the submitted rerun.")
    print(f"Profile: {profile_path}")


if __name__ == "__main__":
    main()
