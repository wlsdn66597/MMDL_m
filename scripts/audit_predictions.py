#!/usr/bin/env python3
"""Summarize parsing and generation failures in an evaluation predictions JSONL."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--tails", type=int, default=0,
                        help="Print this many trailing characters for problematic responses")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit("No prediction rows found")
    unique = len({row["id"] for row in rows})
    modes = Counter(row["parsing"]["mode"] for row in rows)
    finishes = Counter(row["finish_reason"] for row in rows)
    tokens = [row["output_tokens"] for row in rows]
    ambiguous = [row for row in rows if len(set(row["parsing"].get("candidates", []))) > 1]
    unparsed = [row for row in rows if "unparsed" in row["parsing"]["mode"]]
    limited = [row for row in rows if row["finish_reason"] == "length"]
    print(f"rows={len(rows)} unique_ids={unique} correct={sum(row['correct'] for row in rows)} accuracy={sum(row['correct'] for row in rows)/len(rows):.6f}")
    print(f"question_types={dict(Counter(row['question_type'] for row in rows))}")
    print(f"finish_reasons={dict(finishes)}")
    print(f"parse_modes={dict(modes)}")
    print(f"output_tokens=min:{min(tokens)} median:{statistics.median(tokens)} max:{max(tokens)}")
    print(f"unparsed={len(unparsed)} ambiguous={len(ambiguous)} length_limited={len(limited)}")
    if unique != len(rows):
        print("WARNING: duplicate IDs are present")
    if args.tails:
        problem_ids = {row["id"] for row in unparsed + limited + ambiguous}
        for row in rows:
            if row["id"] not in problem_ids:
                continue
            print("\n" + "=" * 80)
            print({key: row[key] for key in (
                "id", "subject", "answer", "parsed_answer", "correct",
                "finish_reason", "output_tokens")})
            print("parsing:", row["parsing"])
            print("tail:", repr(row["raw_response"][-args.tails:]))


if __name__ == "__main__":
    main()
