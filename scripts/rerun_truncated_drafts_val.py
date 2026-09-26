#!/usr/bin/env python3
"""Rerun only 4096-truncated MMMU-val drafts with a larger budget and original final policy."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_mmmu as val
from baseline_contract import check_coverage, read_jsonl, validate_run
from eval_output_policy import generate_policy, load_selection, score_answer
from scripts.replay_two_stage_final_val import source_messages


def target_indices(rows):
    indices = [index for index, row in enumerate(rows) if row["reasoning_length_limited"]]
    if not indices or any(row["stages"][0]["finish_reason"] != "length" for row in
                          (rows[index] for index in indices)):
        raise ValueError("No valid baseline draft-truncation cohort")
    return indices


def combined_summary(baseline_rows, rerun_by_id, baseline_seconds, rerun_seconds):
    selected = set(rerun_by_id)
    if selected != {row["id"] for row in baseline_rows if row["reasoning_length_limited"]}:
        raise ValueError("Rerun IDs do not equal all and only the truncated baseline IDs")
    combined = [rerun_by_id.get(row["id"], row) for row in baseline_rows]
    count = lambda rows: sum(bool(row["correct"]) for row in rows)
    old_selected = [row for row in baseline_rows if row["id"] in selected]
    new_selected = [rerun_by_id[row["id"]] for row in old_selected]
    return {"n": len(combined), "rerun_n": len(selected),
            "baseline_correct": count(baseline_rows), "combined_correct": count(combined),
            "baseline_accuracy": count(baseline_rows) / len(baseline_rows),
            "combined_accuracy": count(combined) / len(combined),
            "truncated_baseline_correct": count(old_selected),
            "truncated_rerun_correct": count(new_selected),
            "baseline_only_correct": sum(old["correct"] and not new["correct"] for old, new in
                                         zip(old_selected, new_selected)),
            "rerun_only_correct": sum(new["correct"] and not old["correct"] for old, new in
                                       zip(old_selected, new_selected)),
            "rerun_draft_length_limited": sum(row["reasoning_length_limited"] for row in new_selected),
            "rerun_final_length_limited": sum(row["finish_reason"] == "length" for row in new_selected),
            "rerun_unparsed": sum(row["parsed_answer"] is None for row in new_selected),
            "rerun_inference_seconds": rerun_seconds,
            "baseline_inference_seconds": baseline_seconds,
            "adaptive_inference_seconds": baseline_seconds + rerun_seconds,
            "question_types": {kind: {"n": len(group), "correct": count(group),
                                       "accuracy": count(group)/len(group)}
                               for kind in ("multiple-choice", "open")
                               for group in [[row for row in combined if row["question_type"] == kind]]}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reasoning-tokens", type=int, default=8192)
    parser.add_argument("--data-root", help="Optional local pinned MMMU snapshot")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    baseline_dir = args.baseline_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir == baseline_dir or baseline_dir in output_dir.parents or output_dir.exists():
        parser.error("Use a new output directory separate from the baseline")
    baseline_manifest, baseline_summary, baseline_rows = validate_run(
        baseline_dir, "mmmu-val", "standard", require_base=True)
    baseline_args = SimpleNamespace(**baseline_manifest["arguments"])
    if not baseline_args.mode == "two-stage" or args.reasoning_tokens <= baseline_args.reasoning_tokens:
        parser.error("New draft budget must exceed the original two-stage budget")
    indices = target_indices(baseline_rows)
    baseline_inputs = read_jsonl(baseline_dir / "inputs.jsonl")
    if args.data_root:
        baseline_args.data_root = args.data_root
    selection, fingerprints, source_n = load_selection(baseline_args)
    coverage = [{"id": ds[index]["id"], "subject": subject,
                 "question_type": ds[index]["question_type"]}
                for ds, index, subject in selection]
    check_coverage(coverage, "mmmu-val")
    if [row["id"] for row in coverage] != [row["id"] for row in baseline_rows]:
        raise ValueError("Dataset selection differs from the baseline")
    # Conservative context check before loading the GPU model. The final prompt will
    # contain at most (new budget - old draft length) additional draft tokens.
    for i in indices:
        old = baseline_rows[i]
        first, second = old["stages"]
        final_budget = baseline_args.open_final_tokens if old["question_type"] == "open" else baseline_args.final_tokens
        if (first["input_tokens"] is None or second["input_tokens"] is None or
                first["input_tokens"] + args.reasoning_tokens + 128 > baseline_args.max_model_len or
                second["input_tokens"] + args.reasoning_tokens - first["output_tokens"] +
                final_budget + 128 > baseline_args.max_model_len):
            raise ValueError(f"8192 draft may exceed the 16384 context: {old['id']}")
    print(f"[preflight] Checking {len(indices)} originally truncated inputs...", flush=True)
    for i in indices:
        dataset, index, subject = selection[i]
        if subject != baseline_rows[i]["subject"] or subject != baseline_inputs[i]["subject"]:
            raise ValueError(f"Subject changed for {baseline_rows[i]['id']}")
        source_messages(dataset[index], baseline_inputs[i], baseline_rows[i])
    print(f"[preflight] {len(indices)}/{len(indices)} verified; "
          f"{len(selection) - len(indices)} completed drafts will be reused", flush=True)
    if args.check_only:
        return

    from huggingface_hub import snapshot_download
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    model_arg = Path(baseline_args.model_path).expanduser()
    model_path = (str(model_arg.resolve()) if model_arg.is_dir() else
                  snapshot_download(baseline_args.model_path, revision=baseline_args.model_revision))
    model_config = json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8"))
    if model_config.get("quantization_config") or model_config.get("model_type") != "qwen3_vl":
        raise ValueError("Expected the same unquantized Qwen3-VL model type")
    run_args = SimpleNamespace(reasoning_tokens=args.reasoning_tokens,
                               final_tokens=baseline_args.final_tokens,
                               open_final_tokens=baseline_args.open_final_tokens,
                               max_model_len=baseline_args.max_model_len)
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                "purpose": "diagnostic_only_not_uniform_submission_baseline",
                "baseline_dir": str(baseline_dir),
                "baseline_manifest_sha256": hashlib.sha256((baseline_dir / "manifest.json").read_bytes()).hexdigest(),
                "baseline_profile_sha256": baseline_manifest["evaluation_profile"]["sha256"],
                "dataset_fingerprints": fingerprints, "source_n": source_n,
                "model_path": baseline_args.model_path, "model_revision": baseline_args.model_revision,
                "target_rule": "baseline reasoning finish_reason == length",
                "target_n": len(indices), "reasoning_tokens": args.reasoning_tokens,
                "final_sampling_recipe": val.RECIPE, "final_tokens": baseline_args.final_tokens,
                "open_final_tokens": baseline_args.open_final_tokens,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    val.write_json(output_dir / "manifest.json", manifest)
    rerun_by_id = {}
    started = time.perf_counter()
    try:
        llm = LLM(model=model_path, tokenizer=model_path, dtype="bfloat16", seed=val.RECIPE["seed"],
                  max_model_len=baseline_args.max_model_len,
                  gpu_memory_utilization=baseline_args.gpu_memory_utilization,
                  limit_mm_per_prompt={"image": 7}, max_num_seqs=1,
                  mm_processor_kwargs={"min_pixels": baseline_args.min_pixels,
                                       "max_pixels": baseline_args.max_pixels},
                  generation_config="vllm")
        if llm.get_tokenizer().get_chat_template() != (baseline_dir / "chat_template.txt").read_text(encoding="utf-8"):
            raise ValueError("Model chat template differs from the baseline")
        with (output_dir / "rerun_predictions.jsonl").open("w", encoding="utf-8") as stream:
            for position, i in enumerate(indices, start=1):
                dataset, index, _ = selection[i]
                example = dataset[index]
                messages, choices, _ = val.build_message(example, "direct")
                is_open = example["question_type"] == "open"
                suffix = (val.OPEN_TEMPLATE if is_open else val.MC_TEMPLATE).split("\n\n")[-1]
                generated = generate_policy(llm, messages, choices, "two-stage", suffix,
                                            run_args, SamplingParams, StructuredOutputsParams,
                                            example["question_type"])
                row = dict(baseline_rows[i])
                row.update(generated)
                row["correct"] = score_answer(row["answer"], row["parsed_answer"], row["question_type"])
                row["baseline_correct"] = baseline_rows[i]["correct"]
                row["baseline_parsed_answer"] = baseline_rows[i]["parsed_answer"]
                rerun_by_id[row["id"]] = row
                stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                stream.flush()
                print(f"[progress] {position}/{len(indices)} id={row['id']} "
                      f"draft_finish={row['stages'][0]['finish_reason']} correct={row['correct']}", flush=True)
        summary = combined_summary(baseline_rows, rerun_by_id, baseline_summary["inference_seconds"],
                                   sum(row["batch_seconds"] for row in rerun_by_id.values()))
        summary["rerun_total_seconds"] = time.perf_counter() - started
        val.write_json(output_dir / "summary.json", summary)
        with (output_dir / "combined_predictions.jsonl").open("w", encoding="utf-8") as stream:
            for old in baseline_rows:
                row = rerun_by_id.get(old["id"], old)
                compact = {"id": old["id"], "subject": old["subject"],
                           "question_type": old["question_type"],
                           "source": (f"rerun_{args.reasoning_tokens}" if old["id"] in rerun_by_id
                                      else f"baseline_{baseline_args.reasoning_tokens}"),
                           "input_sha256": old["input_sha256"],
                           "baseline_correct": bool(old["correct"]),
                           "correct": bool(row["correct"]),
                           "baseline_parsed_answer": old["parsed_answer"],
                           "parsed_answer": row["parsed_answer"]}
                stream.write(json.dumps(compact, ensure_ascii=False, default=str) + "\n")
        manifest["status"] = "complete"
        print(f"[done] baseline={summary['baseline_accuracy']:.2%} "
              f"combined={summary['combined_accuracy']:.2%}", flush=True)
    except BaseException as exc:
        manifest.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        val.write_json(output_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
