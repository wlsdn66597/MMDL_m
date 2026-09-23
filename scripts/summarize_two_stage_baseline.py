#!/usr/bin/env python3
"""Verify every row and summarize the four full two-stage baseline runs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline_contract import RUNS, check_coverage, read_jsonl, validate_run, validate_profile, load_profile
from eval_mmmu import canonical_sha256, write_json
from eval_output_policy import configuration


def check_preflight(folder, benchmark, setting):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    args = SimpleNamespace(**manifest["arguments"])
    profile = load_profile()
    validate_profile(profile, args)
    if manifest.get("status") != "inputs_checked_no_inference" or (args.benchmark, args.setting) != (benchmark, setting):
        raise ValueError("Preflight is incomplete or is for a different setting")
    rows = read_jsonl(folder / "inputs.jsonl")
    check_coverage(rows, benchmark)
    ids = [r["id"] for r in rows]
    if ids != json.loads((folder / "selected_ids.json").read_text(encoding="utf-8")):
        raise ValueError("Preflight IDs differ from the selection")
    config = configuration(args)
    config["selected_ids_sha256"] = canonical_sha256(ids)
    if (manifest.get("evaluation_signature") != {"config": config, "sha256": canonical_sha256(config)}
            or manifest.get("evaluation_profile", {}).get("sha256") != canonical_sha256(profile)):
        raise ValueError("Preflight profile or signature mismatch")
    return ids


def summarize_suite(root):
    root = Path(root)
    results, pro_ids, identity = {}, None, None
    for name, benchmark, setting in RUNS:
        manifest, summary, rows = validate_run(root / name, benchmark, setting)
        current_identity = {key: manifest["arguments"][key] for key in ("model_path", "model_revision")}
        current_identity["code_sha256"] = manifest["code_sha256"]
        if identity is not None and current_identity != identity:
            raise ValueError("Suite mixes checkpoints or source code versions")
        identity = current_identity
        if benchmark == "mmmu-pro":
            ids = {r["id"] for r in rows}
            if pro_ids is not None and ids != pro_ids:
                raise ValueError("MMMU-Pro settings have different ID sets")
            pro_ids = ids
        results[name] = summary
    columns = ["run", "n", "accuracy_pct", "macro_accuracy_pct", "unparsed", "final_length",
               "draft_length", "mean_output_tokens", "inference_minutes", "total_minutes"]
    table = []
    for name, s in results.items():
        table.append([name, str(s["n"]), f"{s['accuracy']*100:.2f}", f"{s['macro_accuracy']*100:.2f}",
                      str(s["unparsed"]), str(s["length_limited"]), str(s["reasoning_length_limited"]),
                      f"{s['output_tokens']/s['n']:.1f}", f"{s['inference_seconds']/60:.1f}", f"{s['total_seconds']/60:.1f}"])
    (root / "summary.tsv").write_text("\n".join("\t".join(row) for row in [columns, *table])+"\n", encoding="utf-8")
    lines = ["# Full two-stage 4096 baseline", "", "All four runs passed ID, coverage, scoring and fixed-profile checks.", "",
             "MMMU validation: 900 = 847 multiple-choice + 53 open. MMMU-Pro: 1730 in each setting (same IDs).", "",
             "| " + " | ".join(columns) + " |", "|" + "---|"*len(columns)]
    lines += ["| " + " | ".join(row) + " |" for row in table]
    lines += ["", "Final-answer failure and draft truncation are separate. All questions remain in the denominator.",
              "MMMU-Pro accuracy is micro accuracy. MMMU submission uses subject macro accuracy (equal to micro for 30×30).",
              "Earlier test ablations informed this protocol; disclose this history. Avoid further item-level test tuning."]
    (root / "baseline_summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    write_json(root / "baseline_summary.json", {"profile": "two_stage4096_v1", "identity": identity, "runs": results})
    print((root / "summary.tsv").read_text(encoding="utf-8"), end="")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--run", choices=[r[0] for r in RUNS], help="Only verify a completed run (for resume)")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--initialize", nargs=4, metavar=("MODEL", "REVISION", "MMMU_DATA", "PRO_DATA"))
    args = parser.parse_args()
    if args.initialize:
        repo = Path(__file__).resolve().parents[1]
        sources = ("eval_output_policy.py", "eval_mmmu.py", "eval_mmmu_pro.py", "mc_parser.py",
                   "baseline_contract.py", "vendor/mmmu_eval_utils.py", "configs/two_stage4096_v1.json",
                   "scripts/run_two_stage_baseline4096.sh", "scripts/summarize_two_stage_baseline.py")
        request = {"arguments": args.initialize, "sources": {
            name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in sources}}
        path = args.root / "suite_request.json"
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != request:
            raise ValueError("Resume request or source code differs; use a new output root")
        if not path.exists():
            write_json(path, request)
        return
    if args.run:
        _, benchmark, setting = next(r for r in RUNS if r[0] == args.run)
        if args.preflight:
            check_preflight(args.root / "checks" / args.run, benchmark, setting)
        else:
            validate_run(args.root / args.run, benchmark, setting)
        print(f"Verified {args.run}")
    else:
        if args.preflight:
            pro_ids = None
            for name, benchmark, setting in RUNS:
                ids = set(check_preflight(args.root / "checks" / name, benchmark, setting))
                if benchmark == "mmmu-pro":
                    if pro_ids is not None and ids != pro_ids:
                        raise ValueError("Preflight MMMU-Pro setting IDs differ")
                    pro_ids = ids
            print("Verified all 6090 input records and paired MMMU-Pro ID sets")
            return
        summarize_suite(args.root)


if __name__ == "__main__":
    main()
