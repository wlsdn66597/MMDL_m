#!/usr/bin/env python3
"""Separate Qwen-style MMMU answer extraction; never reruns the evaluated VLM.

Default is offline rule extraction. An explicit judge model AND API base enable
paid/remote judge requests; keep credentials in an environment variable.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_mmmu as val
from baseline_contract import read_jsonl
from eval_free_reproduction import QWEN_COMMIT, validate_run
from vendor.qwen_mmmu_extract import can_infer, build_prompt


def extraction_input(row):
    # Open-reference reformulation is GRADER ONLY, after blind VLM generation.
    choices = (deepcopy(row["choices"]) if row["question_type"] == "multiple-choice"
               else {"A": str(row["answer"]), "B": "Other Answers"})
    prediction = str(row["raw_response"]).split("</think>")[-1].strip()
    options = "There are several options: \n" + "".join(f"{k}. {v}\n" for k, v in choices.items())
    prompt = build_prompt(row["input"]["question"], options, prediction)
    return choices, prediction, prompt


def judge_request(base, model, key, prompt):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": 4096}
    request = Request(base.rstrip("/") + "/chat/completions",
                      data=json.dumps(body).encode("utf-8"),
                      headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    with urlopen(request, timeout=180) as response:
        result = json.load(response)
    return result


def extract(row, judge=None):
    choices, prediction, prompt = extraction_input(row)
    answer = can_infer(prediction, deepcopy(choices)) or None
    record = {"id": row["id"], "subject": row["subject"], "question_type": row["question_type"],
              "source_sha256": val.canonical_sha256(row), "method": "rule", "judge_response": None}
    if answer is None:
        record.update(method="pending_judge", judge_prompt=prompt)
        if judge is not None:
            response = judge(prompt)
            record.update(method="judge", judge_response=response)
            completion = response["choices"][0]
            # Never turn an incomplete judge response into an option.
            if completion.get("finish_reason") == "stop":
                text = completion["message"].get("content") or ""
                answer = can_infer(text, deepcopy(choices)) or None
    if answer not in {*choices, "Z"}:
        answer = None
    gold = row["answer"] if row["question_type"] == "multiple-choice" else "A"
    record.update(extracted_answer=answer, resolved=answer is not None, correct=answer == gold)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Completed free32768 MMMU validation run")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--judge-model")
    parser.add_argument("--api-base", help="OpenAI-compatible base URL ending in /v1")
    parser.add_argument("--api-key-env", default="QWEN_JUDGE_API_KEY")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if bool(args.judge_model) != bool(args.api_base):
        parser.error("Judge calls require both --judge-model and --api-base")
    if args.api_base and not args.api_base.startswith("https://"):
        parser.error("Remote judge credentials require HTTPS")
    manifest, _, rows = validate_run(args.run)
    if manifest["identity"]["benchmark"] != "mmmu-val":
        parser.error("The published Qwen MMMU scorer is for MMMU, not MMMU-Pro")
    key = os.environ.get(args.api_key_env, "")
    if args.judge_model and not key:
        parser.error(f"Set {args.api_key_env} before enabling judge calls")
    out = args.output_dir.expanduser().resolve()
    if out == args.run.resolve() or args.run.resolve() in out.parents:
        parser.error("Keep rescoring in a separate directory outside the original run")
    if out.exists() and not args.resume:
        parser.error("Output exists; use a new folder or --resume")
    out.mkdir(parents=True, exist_ok=True)
    request = {"prediction_sha256": hashlib.sha256((args.run / "predictions.jsonl").read_bytes()).hexdigest(),
               "qwen_source_commit": QWEN_COMMIT, "judge_model": args.judge_model, "api_base": args.api_base,
               "policy": "qwen-rules-and-prompt; one-judge-call; no-random-fallback-v1"}
    request_path = out / "request.json"
    if request_path.exists() and json.loads(request_path.read_text(encoding="utf-8")) != request:
        parser.error("Resume source/model/endpoint/policy mismatch")
    val.write_json(request_path, request)
    path = out / "extractions.jsonl"
    records = read_jsonl(path) if path.exists() else []
    if len(records) > len(rows) or any(r["source_sha256"] != val.canonical_sha256(row)
                                      for r, row in zip(records, rows)):
        parser.error("Saved judge records do not match the ordered source predictions")
    judge = (lambda prompt: judge_request(args.api_base, args.judge_model, key, prompt)) if args.judge_model else None
    with path.open("a", encoding="utf-8") as stream:
        for row in rows[len(records):]:
            record = extract(row, judge)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            records.append(record)
            print(f"[score] {len(records)}/{len(rows)} {record['method']}", flush=True)
    unresolved = sum(not r["resolved"] for r in records)
    summary = {"n": len(records), "correct": sum(r["correct"] for r in records), "unresolved": unresolved,
               "accuracy_lower_bound": sum(r["correct"] for r in records)/len(records),
               "accuracy": None if unresolved else sum(r["correct"] for r in records)/len(records),
               "judge_calls": sum(r["method"] == "judge" for r in records),
               "note": "Separate Qwen-style extraction, not an exact official score. Unresolved remain in denominator. "
                       "HF validation data; no official 25-attempt retries or random fallback."}
    val.write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
