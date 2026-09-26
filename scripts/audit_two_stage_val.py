#!/usr/bin/env python3
"""Aggregate MMMU-val two-stage errors without inspecting test items or running inference."""
import argparse
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline_contract import validate_run


def group(rows):
    n = len(rows)
    correct = sum(bool(row["correct"]) for row in rows)
    reasoning_tokens = sum(row["stages"][0]["output_tokens"] for row in rows)
    final_tokens = sum(row["stages"][1]["output_tokens"] for row in rows)
    seconds = sum(row["batch_seconds"] for row in rows)
    return n, correct, correct / n if n else 0, reasoning_tokens / n if n else 0, final_tokens / n if n else 0, seconds / 60


def format_group(name, rows):
    n, correct, accuracy, draft_tokens, final_tokens, minutes = group(rows)
    return (f"{name:24} n={n:3} correct={correct:3} accuracy={accuracy:.2%} "
            f"draft_tokens={draft_tokens:.1f} final_tokens={final_tokens:.1f} inference_min={minutes:.1f}")


def report(rows):
    lines = [format_group("all", rows)]
    for kind in ("multiple-choice", "open"):
        subset = [row for row in rows if row["question_type"] == kind]
        lines.append(format_group(kind, subset))
        for limited in (False, True):
            part = [row for row in subset if bool(row["reasoning_length_limited"]) == limited]
            lines.append(format_group("  draft_truncated=" + str(limited), part))
    lines.append("draft_finish_reasons=" + str(dict(Counter(row["stages"][0]["finish_reason"] for row in rows))))
    lines.append("final_finish_reasons=" + str(dict(Counter(row["stages"][1]["finish_reason"] for row in rows))))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_dir", type=Path, help="Existing full two-stage MMMU-val run directory")
    args = parser.parse_args()
    _, _, rows = validate_run(args.baseline_dir, "mmmu-val", "standard")
    print(report(rows))


if __name__ == "__main__":
    main()
