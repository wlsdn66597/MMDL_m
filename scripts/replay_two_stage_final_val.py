#!/usr/bin/env python3
"""Replay only the final answer call of a frozen full MMMU-val two-stage run."""
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
from eval_output_policy import (OPEN_REASON_INSTRUCTION, OPEN_SELECT_INSTRUCTION,
                                REASON_INSTRUCTION, SELECT_INSTRUCTION,
                                load_selection, paired_input_fingerprint,
                                score_answer, stage_messages)
from vendor.mmmu_eval_utils import parse_open_response


def source_messages(example, input_row, baseline_row):
    """Rebuild the original image-bearing prompt and verify it before generation."""
    messages, choices, details = val.build_message(example, "direct")
    is_open = example["question_type"] == "open"
    suffix = (val.OPEN_TEMPLATE if is_open else val.MC_TEMPLATE).split("\n\n")[-1]
    instruction = OPEN_REASON_INSTRUCTION if is_open else REASON_INSTRUCTION
    draft_messages = stage_messages(messages, suffix, instruction)
    redacted = stage_messages(details["messages_without_image_bytes"], suffix, instruction)
    if (example["id"] != input_row["id"] or example["id"] != baseline_row["id"]
            or example["question_type"] != input_row["question_type"]
            or example["question_type"] != baseline_row["question_type"]
            or example["answer"] != baseline_row["answer"]
            or choices != input_row["choices"] or choices != baseline_row["choices"]
            or paired_input_fingerprint(details) != input_row["input_sha256"]
            or input_row["input_sha256"] != baseline_row["input_sha256"]
            or redacted != input_row["reasoning_messages_without_image_bytes"]
            or input_row["final_user_instruction"] !=
               (OPEN_SELECT_INSTRUCTION if is_open else SELECT_INSTRUCTION)):
        raise ValueError(f"Dataset/prompt/input changed for {example['id']}")
    draft = baseline_row["stages"][0]["raw_response"]
    return (draft_messages + [{"role": "assistant", "content": draft},
                              {"role": "user", "content": input_row["final_user_instruction"]}], choices)


def grouped_comparison(rows):
    result = {}
    for group in ("overall", "multiple-choice", "open", "draft_truncated", "draft_completed"):
        selected = [row for row in rows if group == "overall" or row["question_type"] == group or
                    (group == "draft_truncated" and row["draft_truncated"]) or
                    (group == "draft_completed" and not row["draft_truncated"])]
        result[group] = {"n": len(selected),
                         "baseline_correct": sum(row["baseline_correct"] for row in selected),
                         "greedy_correct": sum(row["correct"] for row in selected),
                         "baseline_only": sum(row["baseline_correct"] and not row["correct"] for row in selected),
                         "greedy_only": sum(row["correct"] and not row["baseline_correct"] for row in selected)}
        result[group]["baseline_accuracy"] = (result[group]["baseline_correct"] / len(selected)
                                              if selected else None)
        result[group]["greedy_accuracy"] = (result[group]["greedy_correct"] / len(selected)
                                            if selected else None)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True,
                        help="Existing results/two_stage4096_v1/mmmu_val")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New directory; baseline files are never modified")
    parser.add_argument("--data-root", help="Optional local copy of the pinned MMMU revision")
    parser.add_argument("--check-only", action="store_true", help="Verify all inputs without loading the model")
    args = parser.parse_args(argv)
    baseline_dir = args.baseline_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir == baseline_dir or baseline_dir in output_dir.parents:
        parser.error("Output directory must be separate from the baseline directory")
    if output_dir.exists():
        parser.error(f"Output directory already exists: {output_dir}")
    baseline_manifest, baseline_summary, baseline_rows = validate_run(
        baseline_dir, "mmmu-val", "standard", require_base=True)
    baseline_inputs = read_jsonl(baseline_dir / "inputs.jsonl")
    baseline_args = SimpleNamespace(**baseline_manifest["arguments"])
    if args.data_root:
        baseline_args.data_root = args.data_root
    selection, fingerprints, source_n = load_selection(baseline_args)
    coverage = [{"id": dataset[index]["id"], "subject": subject,
                 "question_type": dataset[index]["question_type"]}
                for dataset, index, subject in selection]
    check_coverage(coverage, "mmmu-val")
    if [item["id"] for item in coverage] != [row["id"] for row in baseline_rows]:
        raise ValueError("Dataset selection/order differs from the baseline")
    print("[preflight] Verifying all 900 original image prompts and saved drafts...", flush=True)
    for (dataset, index, subject), input_row, baseline_row in zip(selection, baseline_inputs, baseline_rows):
        if subject != input_row["subject"] or subject != baseline_row["subject"]:
            raise ValueError(f"Subject mismatch for {baseline_row['id']}")
        source_messages(dataset[index], input_row, baseline_row)
    print("[preflight] 900/900 verified", flush=True)
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
    recipe = dict(val.RECIPE, temperature=0.0, top_p=1.0, top_k=0)
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                "baseline_dir": str(baseline_dir),
                "baseline_manifest_sha256": hashlib.sha256((baseline_dir / "manifest.json").read_bytes()).hexdigest(),
                "baseline_profile_sha256": baseline_manifest["evaluation_profile"]["sha256"],
                "dataset_fingerprints": fingerprints, "source_n": source_n,
                "model_path": baseline_args.model_path, "model_revision": baseline_args.model_revision,
                "final_sampling_recipe": recipe, "only_final_call_replayed": True,
                "final_tokens": baseline_args.final_tokens,
                "open_final_tokens": baseline_args.open_final_tokens,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    val.write_json(output_dir / "manifest.json", manifest)
    started = time.perf_counter()
    rows = []
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
        with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            for (dataset, index, _), input_row, baseline_row in zip(selection, baseline_inputs, baseline_rows):
                example = dataset[index]
                messages, choices = source_messages(example, input_row, baseline_row)
                is_open = example["question_type"] == "open"
                budget = baseline_args.open_final_tokens if is_open else baseline_args.final_tokens
                kwargs = dict(recipe, n=1, max_tokens=budget)
                if not is_open:
                    kwargs["structured_outputs"] = StructuredOutputsParams(choice=list(choices))
                began = time.perf_counter()
                outputs = llm.chat(messages, sampling_params=SamplingParams(**kwargs), use_tqdm=False)
                seconds = time.perf_counter() - began
                if len(outputs) != 1 or len(outputs[0].outputs) != 1:
                    raise RuntimeError("Expected one request and one completion")
                output = outputs[0]
                generated = output.outputs[0]
                if output.prompt_token_ids is not None and len(output.prompt_token_ids) + budget > baseline_args.max_model_len:
                    raise RuntimeError(f"Final call exceeds context: {example['id']}")
                if is_open:
                    parsed = (sorted(parse_open_response(generated.text), key=lambda x: (type(x).__name__, str(x)))
                              if generated.text.strip() and generated.finish_reason != "length" else None)
                    parsed = parsed or None
                else:
                    parsed = generated.text.strip()
                    if parsed not in choices:
                        raise RuntimeError(f"Structured output contract violated for {example['id']}: {parsed!r}")
                row = {"id": example["id"], "subject": baseline_row["subject"],
                       "question_type": example["question_type"], "answer": baseline_row["answer"],
                       "input_sha256": baseline_row["input_sha256"],
                       "draft_truncated": bool(baseline_row["reasoning_length_limited"]),
                       "baseline_parsed_answer": baseline_row["parsed_answer"],
                       "baseline_correct": bool(baseline_row["correct"]),
                       "raw_response": generated.text, "parsed_answer": parsed,
                       "correct": score_answer(baseline_row["answer"], parsed, example["question_type"]),
                       "finish_reason": generated.finish_reason, "output_tokens": len(generated.token_ids),
                       "input_tokens": len(output.prompt_token_ids) if output.prompt_token_ids is not None else None,
                       "seconds": seconds}
                stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                stream.flush()
                rows.append(row)
                print(f"[progress] {len(rows)}/900 id={row['id']} correct={row['correct']}", flush=True)
        if len(rows) != 900:
            raise RuntimeError("Incomplete replay")
        summary = {"n": 900, "baseline_accuracy": baseline_summary["accuracy"],
                   "greedy_accuracy": sum(row["correct"] for row in rows) / 900,
                   "groups": grouped_comparison(rows),
                   "finish_reasons": dict(Counter(row["finish_reason"] for row in rows)),
                   "replayed_final_inference_seconds": sum(row["seconds"] for row in rows),
                   "replay_total_seconds": time.perf_counter() - started}
        val.write_json(output_dir / "summary.json", summary)
        manifest["status"] = "complete"
        print(f"[done] baseline={summary['baseline_accuracy']:.2%} greedy={summary['greedy_accuracy']:.2%}", flush=True)
    except BaseException as exc:
        manifest.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        val.write_json(output_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
