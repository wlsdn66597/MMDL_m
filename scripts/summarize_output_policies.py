#!/usr/bin/env python3
"""Require identical MC inputs and report output-policy accuracy and full generation cost."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_output_policy import MODES, summarize
from scripts.compare_runs import load_rows, load_evaluation_config, metric, config_differences
from eval_mmmu import write_json

POLICY_KEYS = {"mode", "max_tokens", "reasoning_tokens", "reasoning_instruction", "select_instruction", "constraint"}


def report(root):
    rows_by_mode, summaries, configs = {}, {}, {}
    for mode in MODES:
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
    baseline = rows_by_mode["free"]
    pairs, differences = {}, {}
    for mode in MODES[1:]:
        rows = rows_by_mode[mode]
        if set(rows) != set(baseline):
            raise ValueError("Question ID sets differ")
        for key in baseline:
            for field in ("answer", "subject", "question_type", "choices", "input_sha256"):
                if rows[key][field] != baseline[key][field]:
                    raise ValueError(f"Mismatched paired input: {key}/{field}")
        diff = config_differences(configs["free"], configs[mode])
        if any(key not in POLICY_KEYS for key, _, _ in diff):
            raise ValueError(f"Non-policy settings changed: {diff}")
        differences[mode] = diff
        pairs[mode] = metric(baseline, rows, sorted(baseline))
    lines = ["# Output policy comparison", "",
             f"Dataset: {configs['free']['benchmark']} / {configs['free']['setting']}. "
             "Identical IDs, original prompts/images, labels and non-policy settings verified.", "",
             "MC-only development experiment; not the full 900-question assignment baseline. "
             "If run on MMMU-Pro, treat it as a test ablation and disclose it; do not use it for repeated tuning.", "",
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
    write_json(root / "comparison.json", {"summaries": summaries, "paired": pairs, "differences": differences})
    text = "\n".join(lines) + "\n"
    (root / "comparison.md").write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    print(report(parser.parse_args().root))
