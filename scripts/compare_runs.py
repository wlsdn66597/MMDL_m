#!/usr/bin/env python3
"""Compare two MMMU prediction files on the same question IDs."""
import argparse
from collections import defaultdict
from math import comb
import json
from pathlib import Path


def predictions_path(path):
    return path / "predictions.jsonl" if path.is_dir() else path


def load_rows(path):
    path = predictions_path(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError(f"Duplicate IDs in {path}")
    return by_id


def exact_mcnemar_p(a_only, b_only):
    discordant = a_only + b_only
    if discordant == 0:
        return 1.0
    tail = sum(comb(discordant, i) for i in range(min(a_only, b_only) + 1)) / (2 ** discordant)
    return min(1.0, 2 * tail)


def metric(rows_a, rows_b, ids):
    n = len(ids)
    correct_a = sum(bool(rows_a[i]["correct"]) for i in ids)
    correct_b = sum(bool(rows_b[i]["correct"]) for i in ids)
    a_only = sum(bool(rows_a[i]["correct"]) and not bool(rows_b[i]["correct"]) for i in ids)
    b_only = sum(bool(rows_b[i]["correct"]) and not bool(rows_a[i]["correct"]) for i in ids)
    return dict(n=n, correct_a=correct_a, correct_b=correct_b,
                accuracy_a=correct_a / n, accuracy_b=correct_b / n,
                delta=(correct_b - correct_a) / n, a_only=a_only, b_only=b_only,
                both_correct=sum(bool(rows_a[i]["correct"]) and bool(rows_b[i]["correct"]) for i in ids),
                both_wrong=sum(not bool(rows_a[i]["correct"]) and not bool(rows_b[i]["correct"]) for i in ids),
                mcnemar_exact_p=exact_mcnemar_p(a_only, b_only))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--output", type=Path, help="Optional Markdown output path")
    args = parser.parse_args()
    rows_a, rows_b = load_rows(args.run_a), load_rows(args.run_b)
    if set(rows_a) != set(rows_b):
        only_a = sorted(set(rows_a) - set(rows_b))[:5]
        only_b = sorted(set(rows_b) - set(rows_a))[:5]
        raise SystemExit(f"Question ID sets differ: only A={only_a}, only B={only_b}")
    ids = sorted(rows_a)
    groups = {"Overall": ids}
    by_type = defaultdict(list)
    for identifier in ids:
        metadata_a = (rows_a[identifier]["subject"], rows_a[identifier]["question_type"], rows_a[identifier]["answer"])
        metadata_b = (rows_b[identifier]["subject"], rows_b[identifier]["question_type"], rows_b[identifier]["answer"])
        if metadata_a != metadata_b:
            raise SystemExit(f"Question metadata differs for {identifier}")
        by_type[rows_a[identifier]["question_type"]].append(identifier)
    groups.update(by_type)
    results = {name: metric(rows_a, rows_b, group_ids) for name, group_ids in groups.items()}
    lines = [f"# Paired comparison: {args.label_a} vs {args.label_b}", "",
             "| Group | N | A acc | B acc | B-A | A only | B only | Exact McNemar p |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in results.items():
        lines.append(f"| {name} | {row['n']} | {100*row['accuracy_a']:.2f}% | "
                     f"{100*row['accuracy_b']:.2f}% | {100*row['delta']:+.2f} pp | "
                     f"{row['a_only']} | {row['b_only']} | {row['mcnemar_exact_p']:.4f} |")
    if len(ids) == 900:
        lines += ["", "A/B are paired across the complete 900-question MMMU validation split."]
    else:
        lines += ["", "A/B are paired by question ID. A development subset is diagnostic only; "
                        "use the full 900-question run for the submitted baseline."]
    output = "\n".join(lines) + "\n"
    print(output, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
