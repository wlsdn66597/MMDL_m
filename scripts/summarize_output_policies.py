#!/usr/bin/env python3
"""Require identical MC inputs and report output-policy accuracy and full generation cost."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_output_policy import MODES, paired_input_fingerprint, summarize
from scripts.compare_runs import load_rows, load_evaluation_config, metric, config_differences
from eval_mmmu import write_json

POLICY_KEYS = {"mode", "max_tokens", "reasoning_tokens", "reasoning_instruction", "select_instruction", "constraint"}


def baseline_summary(rows, stored):
    values = list(rows.values())
    n = len(values)
    return {"n": n, "correct": sum(r["correct"] for r in values),
            "accuracy": sum(r["correct"] for r in values) / n,
            "unparsed": sum(r["parsed_answer"] is None for r in values),
            "length_limited": sum(r["finish_reason"] == "length" for r in values),
            "reasoning_length_limited": 0, "any_stage_length_limited":
                sum(r["finish_reason"] == "length" for r in values),
            "output_tokens": sum(r["output_tokens"] for r in values),
            "inference_seconds": sum(r["batch_seconds"] / r.get("batch_size", 1) for r in values),
            "calls": n, "total_seconds": stored["total_seconds"]}


def load_external_baseline(folder):
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    stored = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or not manifest.get("rescoring"):
        raise ValueError("External baseline must be a complete parser-v2 rescored run")
    rows = load_rows(folder)
    audits = {row["id"]: row for row in
              (json.loads(line) for line in (folder / "inputs.jsonl").read_text(encoding="utf-8").splitlines()
               if line.strip())}
    if set(rows) != set(audits) or len(rows) != 1730 or stored.get("n") != 1730:
        raise ValueError("External baseline must contain all 1,730 paired inputs and predictions")
    for identifier, row in rows.items():
        if row["correct"] != (row["parsed_answer"] == row["answer"]):
            raise ValueError(f"Inconsistent baseline score: {identifier}")
        row["input_sha256"] = paired_input_fingerprint(audits[identifier])
    config, verified = load_evaluation_config(folder)
    if not verified:
        raise ValueError("External baseline requires a verified evaluation signature")
    return rows, baseline_summary(rows, stored), config, manifest


def compatible_generation(baseline_config, baseline_manifest, method_config):
    expected = {
        "benchmark": "MMMU-Pro test", "setting": method_config["setting"],
        "dataset_revision": method_config["dataset_revision"],
        "sampling_recipe": method_config["sampling_recipe"],
        "prompt": method_config["prompt"], "max_tokens": 8192,
        "max_model_len": method_config["max_model_len"],
        "min_pixels": method_config["min_pixels"], "max_pixels": method_config["max_pixels"],
        "batch_size": 1, "gpu_memory_utilization": method_config["gpu_memory_utilization"],
        "dtype": "bfloat16", "quantization": None,
    }
    mismatches = [(key, baseline_config.get(key), value) for key, value in expected.items()
                  if baseline_config.get(key) != value]
    arguments = baseline_manifest.get("arguments", {})
    for key in ("model_path", "model_revision"):
        if arguments.get(key) != method_config.get(key):
            mismatches.append((key, arguments.get(key), method_config.get(key)))
    if mismatches:
        raise ValueError(f"External baseline generation settings differ: {mismatches}")


def report(root, baseline_free=None):
    rows_by_mode, summaries, configs = {}, {}, {}
    modes = MODES if baseline_free is None else MODES[1:]
    for mode in modes:
        folder = root / mode
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise ValueError(f"Incomplete run: {folder}")
        config, verified = load_evaluation_config(folder)
        if not verified or config.get("mode") != mode:
            raise ValueError(f"Invalid policy manifest: {folder}")
        rows = load_rows(folder)
        if not rows or len(rows) != manifest["selected_n"]:
            raise ValueError(f"Wrong prediction count: {folder}")
        for row in rows.values():
            if row["correct"] != (row["parsed_answer"] == row["answer"]):
                raise ValueError(f"Inconsistent score: {row['id']}")
        rows_by_mode[mode], configs[mode] = rows, config
        summary = summarize(list(rows.values()))
        stored = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        summary["total_seconds"] = stored["total_seconds"]
        summaries[mode] = summary
    baseline_manifest = None
    if baseline_free is not None:
        rows, summary, config, baseline_manifest = load_external_baseline(baseline_free)
        rows_by_mode["free"], summaries["free"], configs["free"] = rows, summary, config
        compatible_generation(config, baseline_manifest, configs["constrained"])
    baseline = rows_by_mode["free"]
    pairs, differences = {}, {}
    for mode in MODES[1:]:
        rows = rows_by_mode[mode]
        if set(rows) != set(baseline):
            raise ValueError("Question ID sets differ")
        for key in baseline:
            for field in ("answer", "subject", "question_type", "option_count", "input_sha256"):
                if rows[key][field] != baseline[key][field]:
                    raise ValueError(f"Mismatched paired input: {key}/{field}")
        diff = config_differences(configs["free"], configs[mode])
        if baseline_free is None and any(key not in POLICY_KEYS for key, _, _ in diff):
            raise ValueError(f"Non-policy settings changed: {diff}")
        differences[mode] = diff
        pairs[mode] = metric(baseline, rows, sorted(baseline))
    lines = ["# Output policy comparison", "",
             f"Dataset: {configs['free']['benchmark']} / {configs['free']['setting']}. "
             "Identical IDs, original prompts/images, labels and non-policy settings verified.", "",
             ("Full 1,730-question MMMU-Pro test ablation using the existing parser-v2 free-generation baseline. "
              "Disclose this test ablation and do not use item-level errors for further tuning."
              if baseline_free is not None else
              "MC-only development experiment; not the full 900-question assignment baseline."), "",
             "Budgets intentionally differ: free generation versus short constrained answers versus a "
             "bounded draft plus constrained selection. This is a method/cost comparison, not an equal-compute experiment.", "",
             "| Mode | N | Accuracy | Unparsed | Final length | Draft length | Any-stage length | Mean output tokens (all stages) | Inference min | Total min |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for mode, s in summaries.items():
        lines.append(f"| {mode} | {s['n']} | {100*s['accuracy']:.2f}% | {s['unparsed']} | "
                     f"{s['length_limited']} | {s['reasoning_length_limited']} | {s['any_stage_length_limited']} | "
                     f"{s['output_tokens']/s['n']:.1f} | {s['inference_seconds']/60:.2f} | {s['total_seconds']/60:.2f} |")
    lines += ["", "| Comparison | Delta pp | Free only correct | Method only correct | Exact McNemar p |",
              "|---|---:|---:|---:|---:|"]
    for mode, p in pairs.items():
        lines.append(f"| free -> {mode} | {100*p['delta']:+.2f} | {p['a_only']} | {p['b_only']} | {p['mcnemar_exact_p']:.6g} |")
    lines += ["", "A constrained valid letter is not necessarily correct. Zero final truncation in two-stage mode "
              "does not imply the working draft finished; inspect Draft length separately.", "",
              "Shared sampling: " + json.dumps(configs['free']['sampling_recipe']), "",
              "Policy differences: " + json.dumps(differences, ensure_ascii=False)]
    write_json(root / "comparison.json", {"summaries": summaries, "paired": pairs,
               "differences": differences,
               "external_free_baseline": str(baseline_free.resolve()) if baseline_free else None,
               "baseline_predictions_sha256":
                   baseline_manifest.get("rescoring", {}).get("source_predictions_sha256")
                   if baseline_manifest else None})
    text = "\n".join(lines) + "\n"
    (root / "comparison.md").write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--baseline-free", type=Path,
                        help="Complete parser-v2 MMMU-Pro free-generation run to reuse")
    args = parser.parse_args()
    print(report(args.root, args.baseline_free))
