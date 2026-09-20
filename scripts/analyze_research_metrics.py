#!/usr/bin/env python3
"""Compute reproducible MMMU research metrics from one run and an optional paired run."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from math import comb, sqrt
from pathlib import Path
import re
import statistics


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError(f"Duplicate IDs in {path}")
    return by_id


def wilson_interval(correct, n, z=1.959963984540054):
    if n == 0:
        return [None, None]
    p = correct / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    margin = z * sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denominator
    return [max(0.0, centre - margin), min(1.0, centre + margin)]


def accuracy_metric(rows):
    n = len(rows)
    correct = sum(bool(row["correct"]) for row in rows)
    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else None,
        "wilson_95_ci": wilson_interval(correct, n),
    }


def final_prompt_text(input_row):
    messages = input_row.get("messages_without_image_bytes", [])
    if not messages:
        return ""
    chunks = messages[0].get("content", [])
    text_chunks = [chunk.get("text", "") for chunk in chunks if chunk.get("type") == "text"]
    return text_chunks[-1] if text_chunks else ""


def input_features(input_row):
    image_count = len(input_row.get("images", []))
    references = [int(value) for value in re.findall(r"\[Image\s+(\d+)\]", final_prompt_text(input_row), re.I)]
    unique_references = list(dict.fromkeys(references))
    if not unique_references:
        reference_pattern = "implicit/no-explicit-reference"
    elif len(unique_references) == 1:
        reference_pattern = "single-explicit-reference"
    else:
        reference_pattern = "multi-explicit-reference"
    image_bucket = "1" if image_count == 1 else "2" if image_count == 2 else "3+"
    return {
        "image_count": image_count,
        "image_count_bucket": image_bucket,
        "referenced_image_count": len(unique_references),
        "reference_pattern": reference_pattern,
        "reference_order": (
            "out-of-order" if len(unique_references) > 1 and unique_references != sorted(unique_references)
            else "in-order" if len(unique_references) > 1 else "not-multi-reference"
        ),
    }


def group_metrics(rows, field):
    groups = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {name: accuracy_metric(group) for name, group in sorted(groups.items())}


def load_run(run_dir):
    run_dir = Path(run_dir)
    required = [run_dir / "predictions.jsonl", run_dir / "inputs.jsonl", run_dir / "manifest.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"Run is missing required files: {missing}")
    predictions = read_jsonl(required[0])
    inputs = read_jsonl(required[1])
    if not predictions:
        raise ValueError("No prediction rows found")
    if set(predictions) != set(inputs):
        raise ValueError("predictions.jsonl and inputs.jsonl ID sets differ")
    manifest = read_json(required[2])
    summary_path = run_dir / "summary.json"
    summary = read_json(summary_path) if summary_path.is_file() else None
    rows = []
    for identifier, prediction in predictions.items():
        row = dict(prediction)
        row.update(input_features(inputs[identifier]))
        rows.append(row)
    return rows, manifest, summary


def safe_mean(values):
    values = [value for value in values if value is not None]
    return statistics.mean(values) if values else None


def summarize_run(run_dir):
    rows, manifest, summary = load_run(run_dir)
    subjects = defaultdict(list)
    for row in rows:
        subjects[row["subject"]].append(row)
    subject_metrics = {name: accuracy_metric(group) for name, group in sorted(subjects.items())}
    macro_subject = statistics.mean(metric["accuracy"] for metric in subject_metrics.values())
    parse_failures = [row for row in rows if "unparsed" in row.get("parsing", {}).get("mode", "")]
    ambiguous = [row for row in rows if len(set(row.get("parsing", {}).get("candidates", []))) > 1]
    length_limited = [row for row in rows if row.get("finish_reason") == "length"]
    prompt_style = manifest.get("arguments", {}).get("prompt_style", "direct")
    expected_mode = "exact_letter" if prompt_style == "direct" else "explicit_final"
    mc_rows = [row for row in rows if row["question_type"] == "multiple-choice"]
    expected_format = [row for row in mc_rows if row.get("parsing", {}).get("mode") == expected_mode]
    batches = {}
    for row in rows:
        batch_id = row.get("batch_id")
        seconds = row.get("batch_seconds")
        if batch_id is None or seconds is None:
            continue
        if batch_id in batches and abs(batches[batch_id] - seconds) > 1e-9:
            raise ValueError(f"Inconsistent batch_seconds for {batch_id}")
        batches[batch_id] = seconds
    inference_seconds = sum(batches.values()) if batches else None
    image_groups = group_metrics(rows, "image_count_bucket")
    reference_groups = group_metrics(rows, "reference_pattern")
    single = image_groups.get("1", {}).get("accuracy")
    multi_rows = [row for row in rows if row["image_count"] >= 2]
    multi = accuracy_metric(multi_rows)
    return {
        "schema_version": "research-metrics-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(Path(run_dir).resolve()),
        "evaluation_signature": manifest.get("evaluation_signature"),
        "overall": accuracy_metric(rows),
        "macro_subject_accuracy": macro_subject,
        "groups": {
            "question_type": group_metrics(rows, "question_type"),
            "image_count": image_groups,
            "reference_pattern": reference_groups,
            "reference_order": group_metrics(rows, "reference_order"),
            "subject": subject_metrics,
        },
        "visual_structure": {
            "multi_image": multi,
            "multi_minus_single_accuracy_pp": (
                100 * (multi["accuracy"] - single) if multi["accuracy"] is not None and single is not None else None
            ),
            "mean_images_per_question": safe_mean([row["image_count"] for row in rows]),
            "mean_explicit_references_per_question": safe_mean([row["referenced_image_count"] for row in rows]),
        },
        "output_quality": {
            "unparsed": len(parse_failures),
            "unparsed_rate": len(parse_failures) / len(rows),
            "ambiguous_mc": len(ambiguous),
            "ambiguous_mc_rate": len(ambiguous) / len(mc_rows) if mc_rows else None,
            "length_limited": len(length_limited),
            "length_limited_rate": len(length_limited) / len(rows),
            "expected_mc_format": expected_mode,
            "expected_mc_format_count": len(expected_format),
            "expected_mc_format_rate": len(expected_format) / len(mc_rows) if mc_rows else None,
            "mean_output_tokens": safe_mean([row.get("output_tokens") for row in rows]),
            "median_output_tokens": statistics.median(
                [row["output_tokens"] for row in rows if row.get("output_tokens") is not None]
            ),
            "mean_input_tokens": safe_mean([row.get("input_tokens") for row in rows]),
        },
        "efficiency": {
            "inference_seconds_unique_batches": inference_seconds,
            "unique_batches": len(batches),
            "questions_per_inference_second": len(rows) / inference_seconds if inference_seconds else None,
            "total_wall_seconds": summary.get("total_seconds") if summary else None,
            "questions_per_total_wall_second": (
                len(rows) / summary["total_seconds"] if summary and summary.get("total_seconds") else None
            ),
            "peak_device_used_mib_sampled": (
                summary.get("gpu", {}).get("peak_device_used_mib_sampled") if summary else None
            ),
        },
    }


def exact_mcnemar_p(a_only, b_only):
    discordant = a_only + b_only
    if discordant == 0:
        return 1.0
    tail = sum(comb(discordant, i) for i in range(min(a_only, b_only) + 1)) / (2 ** discordant)
    return min(1.0, 2 * tail)


def canonical_answer(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def paired_metric(rows_a, rows_b):
    by_b = {row["id"]: row for row in rows_b}
    a_only = sum(row["correct"] and not by_b[row["id"]]["correct"] for row in rows_a)
    b_only = sum(not row["correct"] and by_b[row["id"]]["correct"] for row in rows_a)
    changed = sum(canonical_answer(row.get("parsed_answer")) != canonical_answer(by_b[row["id"]].get("parsed_answer"))
                  for row in rows_a)
    n = len(rows_a)
    return {
        "n": n,
        "accuracy_a": sum(row["correct"] for row in rows_a) / n,
        "accuracy_b": sum(by_b[row["id"]]["correct"] for row in rows_a) / n,
        "delta_b_minus_a_pp": 100 * (b_only - a_only) / n,
        "answer_change_count": changed,
        "answer_change_rate": changed / n,
        "correct_to_wrong": a_only,
        "wrong_to_correct": b_only,
        "mcnemar_exact_p": exact_mcnemar_p(a_only, b_only),
    }


def config_differences(manifest_a, manifest_b):
    config_a = (manifest_a.get("evaluation_signature") or {}).get("config")
    config_b = (manifest_b.get("evaluation_signature") or {}).get("config")
    if config_a is None or config_b is None:
        return None
    return {key: {"a": config_a.get(key), "b": config_b.get(key)}
            for key in sorted(set(config_a) | set(config_b)) if config_a.get(key) != config_b.get(key)}


def summarize_pair(run_a, run_b, allow_config_differences=False):
    rows_a, manifest_a, _ = load_run(run_a)
    rows_b, manifest_b, _ = load_run(run_b)
    by_a, by_b = {row["id"]: row for row in rows_a}, {row["id"]: row for row in rows_b}
    if set(by_a) != set(by_b):
        raise ValueError("Paired runs have different question ID sets")
    differences = config_differences(manifest_a, manifest_b)
    if differences is None and not allow_config_differences:
        raise ValueError("Both paired runs need v1 evaluation signatures")
    if differences and not allow_config_differences:
        raise ValueError(f"Evaluation configs differ: {list(differences)[:5]}")
    for identifier in by_a:
        keys = ("subject", "question_type", "answer")
        if any(by_a[identifier][key] != by_b[identifier][key] for key in keys):
            raise ValueError(f"Question metadata differs for {identifier}")
    groups = {"overall": sorted(by_a)}
    for label, predicate in (
        ("single_image", lambda row: row["image_count"] == 1),
        ("multi_image", lambda row: row["image_count"] >= 2),
        ("multi_explicit_reference", lambda row: row["reference_pattern"] == "multi-explicit-reference"),
    ):
        groups[label] = sorted(identifier for identifier, row in by_a.items() if predicate(row))
    return {
        "run_a": str(Path(run_a).resolve()),
        "run_b": str(Path(run_b).resolve()),
        "config_differences": differences,
        "groups": {
            name: paired_metric([by_a[i] for i in identifiers], [by_b[i] for i in identifiers])
            for name, identifiers in groups.items() if identifiers
        },
    }


def pct(value):
    return "N/A" if value is None else f"{100 * value:.2f}%"


def pp(value):
    return "N/A" if value is None else f"{value:+.2f} pp"


def render_group_table(title, groups):
    lines = [f"### {title}", "", "| Group | N | Correct | Accuracy | Wilson 95% CI |",
             "|---|---:|---:|---:|---:|"]
    for name, metric in groups.items():
        low, high = metric["wilson_95_ci"]
        ci = "N/A" if low is None else f"{100*low:.2f}%–{100*high:.2f}%"
        lines.append(f"| {name} | {metric['n']} | {metric['correct']} | {pct(metric['accuracy'])} | {ci} |")
    return lines


def render_markdown(report):
    overall = report["overall"]
    low, high = overall["wilson_95_ci"]
    output = report["output_quality"]
    efficiency = report["efficiency"]
    lines = ["# MMMU Research Metrics", "",
             f"Run: `{report['run_dir']}`", "",
             "## Primary metrics", "", "| Metric | Value |", "|---|---:|",
             f"| Accuracy | {pct(overall['accuracy'])} ({overall['correct']}/{overall['n']}) |",
             f"| Accuracy Wilson 95% CI | {100*low:.2f}%–{100*high:.2f}% |",
             f"| Macro subject accuracy | {pct(report['macro_subject_accuracy'])} |",
             f"| Multi-image minus single-image accuracy | {pp(report['visual_structure']['multi_minus_single_accuracy_pp'])} |",
             "", "## Output quality", "", "| Metric | Value |", "|---|---:|",
             f"| Unparsed | {output['unparsed']} ({pct(output['unparsed_rate'])}) |",
             f"| Ambiguous MC | {output['ambiguous_mc']} ({pct(output['ambiguous_mc_rate'])}) |",
             f"| Length limited | {output['length_limited']} ({pct(output['length_limited_rate'])}) |",
             f"| Expected MC format `{output['expected_mc_format']}` | {pct(output['expected_mc_format_rate'])} |",
             f"| Mean / median output tokens | {output['mean_output_tokens']:.2f} / {output['median_output_tokens']:.2f} |",
             "", "## Efficiency", "", "| Metric | Value |", "|---|---:|",
             f"| Unique-batch inference seconds | {efficiency['inference_seconds_unique_batches']} |",
             f"| Questions / inference second | {efficiency['questions_per_inference_second']} |",
             f"| Total wall seconds | {efficiency['total_wall_seconds']} |",
             f"| Sampled peak device memory (MiB) | {efficiency['peak_device_used_mib_sampled']} |", ""]
    lines += render_group_table("Question type", report["groups"]["question_type"])
    lines += [""] + render_group_table("Number of images", report["groups"]["image_count"])
    lines += [""] + render_group_table("Explicit image references", report["groups"]["reference_pattern"])
    lines += ["", "The image/reference groups are observational diagnostics. They do not establish that an answer used visual evidence."]
    if report.get("paired"):
        lines += ["", "## Paired run comparison", "",
                  "| Group | N | A acc | B acc | B−A | Answer changed | Correct→wrong | Wrong→correct | McNemar p |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for name, metric in report["paired"]["groups"].items():
            lines.append(f"| {name} | {metric['n']} | {pct(metric['accuracy_a'])} | {pct(metric['accuracy_b'])} | "
                         f"{metric['delta_b_minus_a_pp']:+.2f} pp | {pct(metric['answer_change_rate'])} | "
                         f"{metric['correct_to_wrong']} | {metric['wrong_to_correct']} | {metric['mcnemar_exact_p']:.4f} |")
        lines += ["", "Answer-change rate measures sensitivity, not improvement by itself. Interpret it with accuracy changes and a controlled intervention."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Result directory containing predictions, inputs and manifest")
    parser.add_argument("--paired-run", type=Path, help="Optional result directory with the same question IDs")
    parser.add_argument("--allow-config-differences", action="store_true",
                        help="Allow a paired comparison when settings intentionally differ")
    parser.add_argument("--output-prefix", type=Path,
                        help="Output path without extension; default reports/<run>_research_metrics")
    args = parser.parse_args()
    try:
        report = summarize_run(args.run)
        if args.paired_run:
            report["paired"] = summarize_pair(args.run, args.paired_run, args.allow_config_differences)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    prefix = args.output_prefix or Path("reports") / f"{args.run.name}_research_metrics"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path, markdown_path = prefix.with_suffix(".json"), prefix.with_suffix(".md")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"Wrote {json_path} and {markdown_path}")


if __name__ == "__main__":
    main()
