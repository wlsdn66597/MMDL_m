#!/usr/bin/env python3
"""Combine three complete MMMU-Pro settings into one paired diagnostic report."""
import argparse
from collections import defaultdict
import json
from math import comb
from pathlib import Path


EXPECTED = 1730


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError(f"Duplicate IDs in {path}")
    return by_id


def load_run(path, expected_setting):
    path = Path(path)
    summary = read_json(path / "summary.json")
    manifest = read_json(path / "manifest.json")
    rows = read_jsonl(path / "predictions.jsonl")
    if summary.get("setting") != expected_setting or manifest.get("arguments", {}).get("setting") != expected_setting:
        raise ValueError(f"{path} is not a {expected_setting} run")
    if len(rows) != EXPECTED or not summary.get("complete_1730"):
        raise ValueError(f"{path} is not a complete {EXPECTED}-question run")
    profile = manifest.get("evaluation_profile") or {}
    if not profile.get("sha256"):
        raise ValueError(f"{path} has no fixed-profile signature")
    return summary, manifest, rows


def exact_mcnemar_p(a_only, b_only):
    discordant = a_only + b_only
    if discordant == 0:
        return 1.0
    tail = sum(comb(discordant, index) for index in range(min(a_only, b_only) + 1)) / (2 ** discordant)
    return min(1.0, 2 * tail)


def paired_correctness(rows_a, rows_b, identifiers):
    a_only = sum(rows_a[i]["correct"] and not rows_b[i]["correct"] for i in identifiers)
    b_only = sum(not rows_a[i]["correct"] and rows_b[i]["correct"] for i in identifiers)
    n = len(identifiers)
    accuracy_a = sum(rows_a[i]["correct"] for i in identifiers) / n
    accuracy_b = sum(rows_b[i]["correct"] for i in identifiers) / n
    return {"n": n, "accuracy_a": accuracy_a, "accuracy_b": accuracy_b,
            "delta_b_minus_a_pp": 100 * (accuracy_b - accuracy_a),
            "a_only": a_only, "b_only": b_only,
            "mcnemar_exact_p": exact_mcnemar_p(a_only, b_only)}


def subject_pairs(rows_a, rows_b):
    groups = defaultdict(list)
    for identifier, row in rows_a.items():
        groups[row["subject"]].append(identifier)
    return {subject: paired_correctness(rows_a, rows_b, sorted(ids))
            for subject, ids in sorted(groups.items())}


def same_base_config(manifests):
    ignored = {"setting", "prompt"}
    configs = []
    for manifest in manifests:
        config = dict((manifest.get("evaluation_signature") or {}).get("config") or {})
        if not config:
            raise ValueError("Missing evaluation signature")
        configs.append({key: value for key, value in config.items() if key not in ignored})
    if configs[0] != configs[1] or configs[0] != configs[2]:
        raise ValueError("MMMU-Pro runs differ in settings other than the intended setting/prompt")


def pct(value):
    return f"{100*value:.2f}%"


def render(report):
    scores = report["scores"]
    lines = ["# MMMU-Pro Three-Setting Baseline", "",
             "> Pre-intervention test baseline. Do not use item-level test errors for training or hyperparameter selection.", "",
             "## Overall", "", "| Setting | N | Correct | Accuracy |", "|---|---:|---:|---:|"]
    for name in ("standard-4", "standard-10", "vision"):
        row = scores[name]
        lines.append(f"| {name} | {row['n']} | {row['correct']} | {pct(row['accuracy'])} |")
    lines += ["", "## Controlled gaps", "",
              "| Comparison | B−A | A only | B only | Exact McNemar p |", "|---|---:|---:|---:|---:|"]
    for label, row in report["paired"].items():
        lines.append(f"| {label} | {row['delta_b_minus_a_pp']:+.2f} pp | {row['a_only']} | "
                     f"{row['b_only']} | {row['mcnemar_exact_p']:.4g} |")
    lines += ["", "- `standard-4 → standard-10`: sensitivity to six additional distractors.",
              "- `standard-10 → vision`: added screenshot/OCR and visual question-reading burden.",
              "- These are paired correctness transitions on the same IDs; option letters can differ between settings.",
              "", "## Subject deltas", "",
              "| Subject | 4→10 (pp) | 10→vision (pp) |", "|---|---:|---:|"]
    for subject in report["subjects"]["standard-4_to_standard-10"]:
        first = report["subjects"]["standard-4_to_standard-10"][subject]
        second = report["subjects"]["standard-10_to_vision"][subject]
        lines.append(f"| {subject} | {first['delta_b_minus_a_pp']:+.2f} | {second['delta_b_minus_a_pp']:+.2f} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standard-4", type=Path, required=True)
    parser.add_argument("--standard-10", type=Path, required=True)
    parser.add_argument("--vision", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, default=Path("reports/mmmu_pro_baseline"))
    args = parser.parse_args()
    try:
        s4, m4, r4 = load_run(args.standard_4, "standard-4")
        s10, m10, r10 = load_run(args.standard_10, "standard-10")
        sv, mv, rv = load_run(args.vision, "vision")
        if set(r4) != set(r10) or set(r4) != set(rv):
            raise ValueError("The three MMMU-Pro settings have different ID sets")
        for identifier in r4:
            if r4[identifier]["subject"] != r10[identifier]["subject"] or r4[identifier]["subject"] != rv[identifier]["subject"]:
                raise ValueError(f"Subject differs for {identifier}")
        same_base_config((m4, m10, mv))
        identifiers = sorted(r4)
        report = {
            "schema_version": "mmmu-pro-three-setting-v1",
            "scores": {name: {key: summary[key] for key in ("n", "correct", "accuracy")}
                       for name, summary in (("standard-4", s4), ("standard-10", s10), ("vision", sv))},
            "paired": {
                "standard-4_to_standard-10": paired_correctness(r4, r10, identifiers),
                "standard-10_to_vision": paired_correctness(r10, rv, identifiers),
            },
            "subjects": {
                "standard-4_to_standard-10": subject_pairs(r4, r10),
                "standard-10_to_vision": subject_pairs(r10, rv),
            },
        }
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output_prefix.with_suffix(".json")
    markdown_path = args.output_prefix.with_suffix(".md")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render(report), encoding="utf-8")
    print(f"Wrote {json_path} and {markdown_path}")


if __name__ == "__main__":
    main()
