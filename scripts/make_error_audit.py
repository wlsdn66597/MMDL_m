#!/usr/bin/env python3
"""Export all open-ended errors and a subject-stratified MC error sample to CSV."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prompt_text(input_row):
    if not input_row:
        return ""
    messages = input_row.get("messages_without_image_bytes", [])
    if not messages:
        return ""
    chunks = messages[0].get("content", [])
    return "\n".join(chunk.get("text", "") for chunk in chunks if chunk.get("type") == "text")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--inputs", type=Path, help="Optional inputs.jsonl for prompt and image metadata")
    parser.add_argument("--mc-per-subject", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("reports/error_audit.csv"))
    args = parser.parse_args()
    if args.mc_per_subject < 0:
        parser.error("--mc-per-subject must be non-negative")
    rows = read_jsonl(args.predictions)
    inputs = {row["id"]: row for row in read_jsonl(args.inputs)} if args.inputs else {}
    open_errors = [row for row in rows if not row["correct"] and row["question_type"] == "open"]
    mc_errors = defaultdict(list)
    for row in rows:
        if not row["correct"] and row["question_type"] == "multiple-choice":
            mc_errors[row["subject"]].append(row)
    selected = open_errors[:]
    for subject in sorted(mc_errors):
        selected.extend(mc_errors[subject][:args.mc_per_subject])
    fieldnames = [
        "id", "subject", "question_type", "gold_answer", "parsed_answer", "raw_response",
        "prompt", "image_count", "finish_reason", "parse_mode", "error_category", "notes",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected:
            input_row = inputs.get(row["id"])
            writer.writerow({
                "id": row["id"],
                "subject": row["subject"],
                "question_type": row["question_type"],
                "gold_answer": row["answer"],
                "parsed_answer": json.dumps(row["parsed_answer"], ensure_ascii=False),
                "raw_response": row["raw_response"],
                "prompt": prompt_text(input_row),
                "image_count": len(input_row.get("images", [])) if input_row else "",
                "finish_reason": row["finish_reason"],
                "parse_mode": row["parsing"]["mode"],
                "error_category": "",
                "notes": "",
            })
    print(f"Wrote {len(selected)} errors to {args.output}: "
          f"open={len(open_errors)}, mc={len(selected)-len(open_errors)}")
    print("Fill error_category with one of: perception/OCR, knowledge, reasoning/calculation, "
          "prompt/format, parser/scoring, dataset/ambiguous.")


if __name__ == "__main__":
    main()
