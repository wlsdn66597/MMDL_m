#!/usr/bin/env python3
"""Controlled MC output-policy experiments; development data by default."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import traceback

import eval_mmmu as val
import eval_mmmu_pro as pro
from mc_parser import PARSER_POLICY, parse_pro

MODES = ("free", "constrained", "two-stage")
REASON_INSTRUCTION = (
    "Work out the problem briefly, using the supplied images. Identify the relevant visual "
    "evidence and perform only the necessary calculations. Avoid repeating alternatives. "
    "This is a working draft; a separate step will select the final option."
)
SELECT_INSTRUCTION = (
    "Now select the single best option for the ORIGINAL question using the original images "
    "and the working draft above. The draft may be incomplete or incorrect; do not blindly "
    "copy it. Output only one valid option letter."
)


def paired_input_fingerprint(details):
    """Fields recorded by both legacy MMMU-Pro runs and new policy runs."""
    core = {key: details[key] for key in
            ("messages_without_image_bytes", "images", "option_count") if key in details}
    if "messages_without_image_bytes" not in core or "images" not in core:
        raise ValueError("Input audit is missing messages or images")
    return val.canonical_sha256(core)


def select_indices(ids, types, count, seed):
    """Outcome-independent, stable sample within each subject, MC only."""
    indices = [i for i, kind in enumerate(types) if kind == "multiple-choice"]
    indices.sort(key=lambda i: hashlib.sha256(f"{seed}:{ids[i]}".encode()).hexdigest())
    return indices[:count] if count else indices


def stage_messages(messages, suffix):
    result = deepcopy(messages)
    last = result[-1]["content"][-1]
    if last.get("type") != "text" or not last["text"].endswith(suffix):
        raise ValueError("Cannot locate the direct instruction; refusing to alter input silently")
    last["text"] = last["text"][:-len(suffix)] + REASON_INSTRUCTION
    return result


def generate_policy(llm, messages, choices, mode, suffix, args, params_class, structured_class):
    """Never pass labels to generation. Keep both stages, including truncated drafts."""
    letters = list(choices)
    if not 2 <= len(letters) <= 26 or letters != [chr(65+i) for i in range(len(letters))]:
        raise ValueError("Expected contiguous A-Z option letters")
    stages = []

    def call(conversation, budget, constrained, name):
        kwargs = dict(val.RECIPE, n=1, max_tokens=budget)
        if constrained:
            kwargs["structured_outputs"] = structured_class(choice=letters)
        params = params_class(**kwargs)
        start = time.perf_counter()
        outputs = llm.chat(conversation, sampling_params=params, use_tqdm=False)
        seconds = time.perf_counter() - start
        if len(outputs) != 1 or len(outputs[0].outputs) != 1:
            raise RuntimeError("Expected one request and one completion")
        output = outputs[0]
        generated = output.outputs[0]
        stages.append({"stage": name, "raw_response": generated.text,
                       "output_tokens": len(generated.token_ids),
                       "input_tokens": len(output.prompt_token_ids) if output.prompt_token_ids is not None else None,
                       "finish_reason": generated.finish_reason, "stop_reason": generated.stop_reason,
                       "seconds": seconds, "sampling_params": str(params),
                       "constrained_choices": letters if constrained else None})
        return generated

    if mode == "two-stage":
        draft_messages = stage_messages(messages, suffix)
        draft = call(draft_messages, args.reasoning_tokens, False, "reasoning")
        # Retain the original images. There is no second copy of the image payload.
        final_messages = draft_messages + [
            {"role": "assistant", "content": draft.text},
            {"role": "user", "content": SELECT_INSTRUCTION}]
        final = call(final_messages, args.final_tokens, True, "final")
    else:
        final = call(messages, args.max_tokens if mode == "free" else args.final_tokens,
                     mode == "constrained", "final")
    if mode != "free":
        # Do not let the permissive MC parser hide a broken constraint backend.
        parsed = final.text.strip()
        if parsed not in choices:
            raise RuntimeError(f"Structured output contract violated: {final.text!r}")
        info = {"mode": "exact_letter", "candidates": [parsed]}
    else:
        parsed, info = parse_pro(final.text, choices, final.finish_reason)
    return {"raw_response": final.text, "parsed_answer": parsed, "parsing": info,
            "finish_reason": final.finish_reason, "stop_reason": final.stop_reason,
            "output_tokens": sum(stage["output_tokens"] for stage in stages),
            "input_tokens": sum(stage["input_tokens"] for stage in stages)
                if all(stage["input_tokens"] is not None for stage in stages) else None,
            "final_output_tokens": stages[-1]["output_tokens"],
            "batch_seconds": sum(stage["seconds"] for stage in stages),
            "reasoning_length_limited": any(stage["stage"] == "reasoning" and
                                            stage["finish_reason"] == "length" for stage in stages),
            "any_stage_length_limited": any(stage["finish_reason"] == "length" for stage in stages),
            "stages": stages}


def load_selection(args):
    from datasets import load_dataset
    selection, fingerprints, seen = [], {}, set()
    if args.benchmark == "mmmu-val":
        for subject in val.SUBJECTS:
            dataset = load_dataset("MMMU/MMMU", subject, split="validation", revision=val.DATA_REV)
            ids = list(dataset["id"])
            if len(ids) != 30 or len(set(ids)) != 30 or seen.intersection(ids):
                raise ValueError(f"Invalid MMMU subject: {subject}")
            seen.update(ids)
            fingerprints[subject] = dataset._fingerprint
            indices = select_indices(ids, dataset["question_type"], args.per_subject, args.selection_seed)
            selection.extend((dataset, i, subject) for i in indices)
    else:
        dataset = load_dataset(pro.DATASET, pro.SETTINGS[args.setting], split="test", revision=pro.DATA_REV)
        ids = list(dataset["id"])
        if len(ids) != 1730 or len(set(ids)) != 1730:
            raise ValueError("Expected 1730 unique MMMU-Pro IDs")
        seen.update(ids)
        fingerprints[args.setting] = dataset._fingerprint
        subjects = list(dataset["subject"])
        for subject in sorted(set(subjects)):
            types = ["multiple-choice" if s == subject else "excluded" for s in subjects]
            indices = select_indices(ids, types, args.per_subject, args.selection_seed)
            selection.extend((dataset, i, subject) for i in indices)
    if not selection:
        raise ValueError("Empty selection")
    return selection, fingerprints, len(seen)


def configuration(args):
    return {"benchmark": "MMMU validation MC" if args.benchmark == "mmmu-val" else "MMMU-Pro test",
            "pipeline_version": "output-policy-v1", "parser_policy": PARSER_POLICY,
            "dataset_revision": val.DATA_REV if args.benchmark == "mmmu-val" else pro.DATA_REV,
            "setting": args.setting, "mode": args.mode, "sampling_recipe": val.RECIPE,
            "max_tokens": args.max_tokens if args.mode == "free" else args.final_tokens,
            "reasoning_tokens": args.reasoning_tokens if args.mode == "two-stage" else 0,
            "reasoning_instruction": REASON_INSTRUCTION if args.mode == "two-stage" else None,
            "select_instruction": SELECT_INSTRUCTION if args.mode == "two-stage" else None,
            "prompt": val.MC_TEMPLATE if args.benchmark == "mmmu-val" else pro.prompt_for(args.setting),
            "constraint": "per-row choice letters" if args.mode != "free" else None,
            "min_pixels": args.min_pixels, "max_pixels": args.max_pixels,
            "max_model_len": args.max_model_len, "batch_size": 1,
            "dtype": "bfloat16", "gpu_memory_utilization": args.gpu_memory_utilization,
            "selection_seed": args.selection_seed, "per_subject": args.per_subject,
            "model_path": args.model_path, "model_revision": args.model_revision}


def summarize(rows):
    n = len(rows)
    return {"n": n, "correct": sum(r["correct"] for r in rows),
            "accuracy": sum(r["correct"] for r in rows) / n,
            "unparsed": sum(r["parsed_answer"] is None for r in rows),
            "length_limited": sum(r["finish_reason"] == "length" for r in rows),
            "reasoning_length_limited": sum(r["reasoning_length_limited"] for r in rows),
            "any_stage_length_limited": sum(r["any_stage_length_limited"] for r in rows),
            "output_tokens": sum(r["output_tokens"] for r in rows),
            "inference_seconds": sum(r["batch_seconds"] for r in rows),
            "calls": sum(len(r["stages"]) for r in rows)}


def run(args, outdir, manifest):
    selection, fingerprints, source_n = load_selection(args)
    selected_ids = [ds[i]["id"] for ds, i, _ in selection]
    config = configuration(args)
    config["selected_ids_sha256"] = val.canonical_sha256(selected_ids)
    manifest.update(dataset_fingerprints=fingerprints, source_n=source_n, selected_n=len(selection),
                    evaluation_signature={"config": config, "sha256": val.canonical_sha256(config)})
    val.write_json(outdir / "selected_ids.json", selected_ids)
    val.write_json(outdir / "manifest.json", manifest)
    print(f"[data] {len(selection)} MC questions; {args.benchmark}/{args.setting}; {args.mode}", flush=True)
    llm = None
    model_seconds = 0
    if not args.check_only:
        from huggingface_hub import snapshot_download
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        model_path = (str(Path(args.model_path).expanduser().resolve()) if Path(args.model_path).expanduser().is_dir()
                      else snapshot_download(args.model_path, revision=args.model_revision))
        model_config = json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8"))
        if model_config.get("quantization_config"):
            raise ValueError("This comparison requires an unquantized checkpoint")
        start = time.perf_counter()
        llm = LLM(model=model_path, tokenizer=model_path, dtype="bfloat16", seed=val.RECIPE["seed"],
                  max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization,
                  limit_mm_per_prompt={"image": 7}, max_num_seqs=1,
                  mm_processor_kwargs={"min_pixels": args.min_pixels, "max_pixels": args.max_pixels},
                  generation_config="vllm")
        model_seconds = time.perf_counter() - start
        manifest["resolved_model_path"] = model_path
        (outdir / "chat_template.txt").write_text(llm.get_tokenizer().get_chat_template(), encoding="utf-8")
    rows, image_counts = [], Counter()
    with (outdir / "inputs.jsonl").open("w", encoding="utf-8") as audit, \
         (outdir / "predictions.jsonl").open("w", encoding="utf-8") as predictions:
        for ds, index, subject in selection:
            ex = ds[index]
            if args.benchmark == "mmmu-val":
                messages, choices, details = val.build_message(ex, "direct")
                suffix = val.MC_TEMPLATE.split("\n\n")[-1]
            else:
                messages, choices, details = pro.build_message(ex, args.setting)
                suffix = pro.prompt_for(args.setting)
            if ex["answer"] not in choices:
                raise ValueError(f"Gold answer not in option set: {ex['id']}")
            image_counts[len(details["images"])] += 1
            # Common input hash excludes labels and output policy; validates paired inputs.
            input_hash = paired_input_fingerprint(details)
            entry = {"id": ex["id"], "subject": subject, "choices": choices,
                     **details, "input_sha256": input_hash}
            if args.mode == "two-stage":
                entry["reasoning_messages_without_image_bytes"] = stage_messages(
                    details["messages_without_image_bytes"], suffix)
                entry["final_user_instruction"] = SELECT_INSTRUCTION
            audit.write(json.dumps(entry, ensure_ascii=False) + "\n")
            audit.flush()
            if args.check_only:
                continue
            generated = generate_policy(llm, messages, choices, args.mode, suffix, args,
                                        SamplingParams, StructuredOutputsParams)
            row = {"id": ex["id"], "subject": subject, "question_type": "multiple-choice",
                   "answer": ex["answer"], "choices": choices, "option_count": len(choices),
                   "mode": args.mode, "setting": args.setting, "input_sha256": input_hash,
                   "batch_id": ex["id"], "batch_size": 1, **generated}
            row["correct"] = row["parsed_answer"] == row["answer"]
            predictions.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            predictions.flush()
            rows.append(row)
            print(f"[progress] {len(rows)}/{len(selection)} calls={len(row['stages'])} "
                  f"tokens={row['output_tokens']} seconds={row['batch_seconds']:.1f}", flush=True)
    manifest["image_count_distribution"] = dict(image_counts)
    if args.check_only:
        manifest["status"] = "inputs_checked_no_inference"
        return None
    if len(rows) != len(selection) or len({r["id"] for r in rows}) != len(rows):
        raise RuntimeError("Incomplete or duplicate predictions")
    manifest["status"] = "complete"
    return {**summarize(rows), "model_load_seconds": model_seconds, "mode": args.mode,
            "image_count_distribution": dict(image_counts)}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--benchmark", choices=("mmmu-val", "mmmu-pro"), default="mmmu-val")
    parser.add_argument("--setting", choices=("standard", *pro.SETTINGS), default="standard")
    parser.add_argument("--per-subject", type=int, default=4, help="Stable MC sample per subject; 0 = all MC")
    parser.add_argument("--selection-seed", type=int, default=3407)
    parser.add_argument("--max-tokens", type=int, default=8192, help="Free generation ceiling")
    parser.add_argument("--reasoning-tokens", type=int, default=1024)
    parser.add_argument("--final-tokens", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--min-pixels", type=int, default=1003520)
    parser.add_argument("--max-pixels", type=int, default=4014080)
    parser.add_argument("--gpu-memory-utilization", type=float, default=.85)
    parser.add_argument("--monitor-gpu", default="0")
    parser.add_argument("--model-path", default=val.MODEL)
    parser.add_argument("--model-revision", default=val.MODEL_REV)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if (args.benchmark == "mmmu-val") != (args.setting == "standard"):
        parser.error("mmmu-val uses --setting standard; mmmu-pro requires standard-4, standard-10 or vision")
    if args.per_subject < 0 or not 0 < args.min_pixels <= args.max_pixels or not 0 < args.gpu_memory_utilization < 1:
        parser.error("Invalid sample size, pixels or GPU utilization")
    if min(args.max_tokens, args.reasoning_tokens, args.final_tokens) < 1:
        parser.error("All output budgets must be positive")
    if max(args.max_tokens, args.reasoning_tokens + args.final_tokens) >= args.max_model_len:
        parser.error("Context must exceed output budgets; leave room for images, question and stage-2 instruction")
    return args


def main():
    args = arguments()
    outdir = args.output_dir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    packages = {}
    for name in ("torch", "vllm", "transformers", "datasets", "Pillow", "huggingface-hub"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    manifest = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                "arguments": vars(args), "packages": packages, "python": sys.version,
                "code_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                for name in ("eval_output_policy.py", "eval_mmmu.py", "eval_mmmu_pro.py", "mc_parser.py")}}
    val.write_json(outdir / "manifest.json", manifest)
    monitor = val.GpuMonitor(args.monitor_gpu)
    monitor.thread.start()
    try:
        summary = run(args, outdir, manifest)
        if summary is not None:
            summary.update(gpu=monitor.stop(), total_seconds=time.perf_counter()-start)
            val.write_json(outdir / "summary.json", summary)
            print(f"[done] accuracy={summary['accuracy']:.4f} unparsed={summary['unparsed']}", flush=True)
    except BaseException as exc:
        manifest.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        manifest.update(gpu=monitor.stop(), total_seconds=time.perf_counter()-start)
        val.write_json(outdir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
