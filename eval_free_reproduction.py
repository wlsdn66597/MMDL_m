#!/usr/bin/env python3
"""One free response per question, using Qwen reference prompts and a 32K ceiling."""
import argparse
import base64
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

from PIL import Image
import eval_mmmu as val
import eval_mmmu_pro as pro
from baseline_contract import RUNS, aggregate_rows, check_coverage, read_jsonl
from eval_output_policy import load_selection, score_answer, summarize
from mc_parser import parse_pro
from vendor.mmmu_eval_utils import parse_open_response

PROFILE_PATH = Path(__file__).parent / "configs/free32768_reference_v1.json"
QWEN_COMMIT = "96588727e44c78b25ba03ea03b8e12f7e64fd0da"
MMMU_SUFFIX = "Please select the correct answer from the options above."
PRO_SUFFIX = "Please select the correct answer from the options."
VISION_PROMPT = "Identify the problem and solve it. Think step by step before answering."
SOURCES = ("eval_free_reproduction.py", "eval_mmmu.py", "eval_mmmu_pro.py",
           "eval_output_policy.py", "baseline_contract.py", "mc_parser.py",
           "vendor/mmmu_eval_utils.py", "configs/free32768_reference_v1.json")


def profile():
    value = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    if (value["max_tokens"] != 32768 or value["max_model_len"] != 65536
            or value["sampling_recipe"] != val.RECIPE or value["qwen_source_commit"] != QWEN_COMMIT
            or value["mmmu_revision"] != val.DATA_REV or value["mmmu_pro_revision"] != pro.DATA_REV):
        raise ValueError("The frozen free32768 reference profile was changed")
    return value


def code_hashes():
    root = Path(__file__).parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCES}


def build_reference_message(ex, benchmark, setting):
    # Reuse image validation/serialization, then apply the reference text/layout.
    if benchmark == "mmmu-val":
        messages, choices, details = val.build_message(ex, "direct")
    else:
        messages, choices, details = pro.build_message(ex, setting)
    is_open = benchmark == "mmmu-val" and ex["question_type"] == "open"
    if is_open:
        choices = {}
    if setting == "vision":
        text = VISION_PROMPT
    else:
        question = ex["question"]
        options = "\n".join(f"{letter}. {option}" for letter, option in choices.items())
        if benchmark == "mmmu-val":
            text = f"Question: {question}"
            if choices:
                text += f"\nOptions:\n{options}\n{MMMU_SUFFIX}"
        else:
            text = f"{question}\n{options}\n{PRO_SUFFIX}"
    # Qwen's published builder puts the images first, with no added Image N labels.
    for content in (messages[0]["content"], details["messages_without_image_bytes"][0]["content"]):
        content[:] = [item for item in content if item["type"] == "image_url"]
        content.append({"type": "text", "text": text})
    details.update(choices=choices, option_count=len(choices),
                   question=ex["question"] if setting != "vision" else "[Question is in the image]")
    return messages, choices, details


def input_hash(entry):
    return val.canonical_sha256({k: entry[k] for k in
                                ("messages_without_image_bytes", "images", "choices", "question")})


def selection_metadata(selection, benchmark):
    # Column reads avoid decoding every image just to read an ID or question type.
    cache, result = {}, []
    for ds, index, subject in selection:
        if id(ds) not in cache:
            cache[id(ds)] = (list(ds["id"]), list(ds["question_type"]) if benchmark == "mmmu-val"
                             else ["multiple-choice"] * len(ds))
        ids, kinds = cache[id(ds)]
        result.append({"id": ids[index], "subject": subject, "question_type": kinds[index]})
    check_coverage(result, benchmark)
    return result


def parse_response(text, choices, kind, finish_reason):
    if kind == "multiple-choice":
        return parse_pro(text, choices, finish_reason)
    parsed = (sorted(parse_open_response(text), key=lambda x: (type(x).__name__, str(x)))
              if text.strip() and finish_reason != "length" else None)
    return parsed or None, {"mode": "official_open" if parsed else "unparsed_open"}


def ensure_budget(prompt_tokens, p):
    if prompt_tokens + p["max_tokens"] > p["max_model_len"]:
        raise ValueError(f"Input {prompt_tokens} + output {p['max_tokens']} exceeds context "
                         f"{p['max_model_len']}; no input truncation or silent output reduction allowed")


def count_input_tokens(processor, messages, p):
    # The same HF processor and pixel limits are used by vLLM. Check before decoding.
    images = []
    for item in messages[0]["content"]:
        if item["type"] == "image_url":
            payload = item["image_url"]["url"].split(",", 1)[1]
            images.append(Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB"))
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = processor(text=[prompt_text], images=images, return_tensors=None,
                        min_pixels=p["min_pixels"], max_pixels=p["max_pixels"])
    count = len(encoded["input_ids"][0])
    ensure_budget(count, p)
    return count


def generate_one(llm, messages, choices, kind, p, params_class, checked_tokens):
    ensure_budget(checked_tokens, p)
    params = params_class(**p["sampling_recipe"], max_tokens=p["max_tokens"], n=1)
    start = time.perf_counter()
    outputs = llm.chat(messages, sampling_params=params, use_tqdm=False)
    seconds = time.perf_counter() - start
    if len(outputs) != 1 or len(outputs[0].outputs) != 1:
        raise ValueError("Expected one request and one completion")
    output, generated = outputs[0], outputs[0].outputs[0]
    if output.prompt_token_ids is None:
        raise ValueError("Cannot verify the actual multimodal context length")
    actual_tokens = len(output.prompt_token_ids)
    ensure_budget(actual_tokens, p)
    if generated.finish_reason not in ("stop", "length"):
        raise ValueError(f"Unexpected generation termination: {generated.finish_reason}")
    if generated.finish_reason == "length" and len(generated.token_ids) != p["max_tokens"]:
        raise ValueError("Generation stopped before the requested output ceiling; refusing a shortened budget")
    parsed, parsing = parse_response(generated.text, choices, kind, generated.finish_reason)
    stage = {"stage": "final", "raw_response": generated.text, "output_tokens": len(generated.token_ids),
             "input_tokens": actual_tokens, "finish_reason": generated.finish_reason,
             "stop_reason": generated.stop_reason, "seconds": seconds,
             "requested_max_tokens": p["max_tokens"], "constrained_choices": None}
    return {"raw_response": generated.text, "parsed_answer": parsed, "parsing": parsing,
            "finish_reason": generated.finish_reason, "stop_reason": generated.stop_reason,
            "output_tokens": len(generated.token_ids), "input_tokens": actual_tokens,
            "processor_input_tokens": checked_tokens, "batch_seconds": seconds, "stages": [stage],
            "reasoning_length_limited": False, "any_stage_length_limited": generated.finish_reason == "length"}


def validate_rows(rows, selected_ids, p):
    if len(rows) > len(selected_ids) or [r["id"] for r in rows] != selected_ids[:len(rows)]:
        raise ValueError("Predictions are not a unique, ordered prefix of the selected questions")
    for row in rows:
        if row["input_sha256"] != input_hash(row["input"]):
            raise ValueError(f"Input hash mismatch: {row['id']}")
        if any(row[k] != row["input"][k] for k in ("id", "subject", "question_type")):
            raise ValueError(f"Input metadata mismatch: {row['id']}")
        stages = row["stages"]
        if len(stages) != 1 or stages[0]["constrained_choices"] is not None:
            raise ValueError("This profile requires a single unconstrained generation")
        stage = stages[0]
        if (stage["requested_max_tokens"] != p["max_tokens"] or stage["raw_response"] != row["raw_response"]
                or stage["output_tokens"] != row["output_tokens"] or stage["finish_reason"] != row["finish_reason"]
                or stage["input_tokens"] != row["input_tokens"] or stage["seconds"] != row["batch_seconds"]
                or row["choices"] != row["input"]["choices"]):
            raise ValueError("Recorded output/budget does not match the generation")
        if row["finish_reason"] not in ("stop", "length") or not 0 <= row["output_tokens"] <= p["max_tokens"]:
            raise ValueError("Invalid termination or token count")
        ensure_budget(row["input_tokens"], p)
        if row["finish_reason"] == "length" and row["output_tokens"] != p["max_tokens"]:
            raise ValueError("A run silently reduced its output ceiling")
        parsed, parsing = parse_response(row["raw_response"], row["input"]["choices"],
                                         row["question_type"], row["finish_reason"])
        if (row["parsed_answer"] != parsed or row["parsing"] != parsing
                or row["correct"] != score_answer(row["answer"], parsed, row["question_type"])):
            raise ValueError(f"Stored scoring differs from raw-response scoring: {row['id']}")


def make_summary(rows):
    result = {**summarize(rows), **aggregate_rows(rows)}
    result["seconds_by_subject"] = {s: sum(r["batch_seconds"] for r in rows if r["subject"] == s)
                                    for s in sorted({r["subject"] for r in rows})}
    return result


def validate_run(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete" or manifest["profile"] != profile():
        raise ValueError(f"Incomplete or incompatible run: {folder}")
    rows = read_jsonl(folder / "predictions.jsonl")
    ids = json.loads((folder / "selected_ids.json").read_text(encoding="utf-8"))
    if (manifest["identity"]["profile"] != manifest["profile"]
            or manifest["selected_ids_sha256"] != val.canonical_sha256(ids)
            or manifest["selected_n"] != len(ids) or manifest["source_n"] != len(ids)):
        raise ValueError("Manifest profile or coverage fingerprint mismatch")
    validate_rows(rows, ids, manifest["profile"])
    check_coverage(rows, manifest["identity"]["benchmark"])
    if len(rows) != len(ids):
        raise ValueError("Missing predictions")
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    if any(summary.get(k) != v for k, v in make_summary(rows).items()):
        raise ValueError("Summary differs from the actual predictions")
    return manifest, summary, rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("mmmu-val", "mmmu-pro"), default="mmmu-val")
    parser.add_argument("--setting", choices=("standard", *pro.SETTINGS), default="standard")
    parser.add_argument("--model-path", default=val.MODEL)
    parser.add_argument("--model-revision", default=val.MODEL_REV)
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--monitor-gpu", default="0")
    args = parser.parse_args(argv)
    if (args.benchmark == "mmmu-val") != (args.setting == "standard"):
        parser.error("MMMU validation uses standard; MMMU-Pro uses standard-4, standard-10 or vision")
    p = profile()
    out = args.output_dir.expanduser().resolve()
    if out.exists() and not args.resume:
        raise FileExistsError(f"Output exists: {out}. Use --resume with the same configuration.")
    out.mkdir(parents=True, exist_ok=True)
    identity = {k: getattr(args, k) for k in ("benchmark", "setting", "model_path", "model_revision", "data_root")}
    identity.update(profile=p, code_sha256=code_hashes(), check_only=args.check_only)
    manifest_path = out / "manifest.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    if previous and previous["identity"] != identity:
        raise ValueError("Resume requires identical data/model/code/profile/check-only setting")
    if previous and previous["status"] == "complete":
        validate_run(out)
        print(f"[skip] Validated complete run: {out}")
        return
    if previous and previous["status"] == "inputs_checked_no_inference":
        check_coverage(read_jsonl(out / "inputs.jsonl"), args.benchmark)
        print(f"[skip] Validated input check: {out}")
        return
    manifest = previous or {"identity": identity, "profile": p, "sessions": []}
    manifest.update(status="running", python=sys.version)
    manifest.pop("error", None)
    session = {"started_utc": datetime.now(timezone.utc).isoformat()}
    manifest["sessions"].append(session)
    start, monitor, rows, success = time.perf_counter(), None, [], False
    val.write_json(manifest_path, manifest)
    try:
        loader_args = SimpleNamespace(**vars(args), require_full=True, include_open=True,
                                      per_subject=0, selection_seed=p["sampling_recipe"]["seed"])
        selection, fingerprints, source_n = load_selection(loader_args)
        metadata = selection_metadata(selection, args.benchmark)
        ids = [r["id"] for r in metadata]
        if previous and "selected_ids_sha256" in manifest:
            if (manifest["selected_ids_sha256"] != val.canonical_sha256(ids)
                    or manifest["dataset_fingerprints"] != fingerprints):
                raise ValueError("Dataset changed since the previous attempt")
        manifest.update(source_n=source_n, selected_n=len(ids), dataset_fingerprints=fingerprints,
                        selected_ids_sha256=val.canonical_sha256(ids))
        val.write_json(out / "selected_ids.json", ids)
        print(f"[data] {len(ids)} unique IDs; {dict(Counter(r['question_type'] for r in metadata))}", flush=True)
        pred_path = out / "predictions.jsonl"
        if pred_path.exists():
            rows = read_jsonl(pred_path)  # A partial/corrupt JSON line fails loudly; never guess/drop it.
            validate_rows(rows, ids, p)
        manifest["packages"] = {}
        for name in ("torch", "vllm", "transformers", "datasets", "Pillow", "huggingface-hub"):
            try:
                manifest["packages"][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                manifest["packages"][name] = None
        if previous and previous.get("environment_packages") not in (None, manifest["packages"]):
            raise ValueError("Package versions changed during a partial run")
        manifest["environment_packages"] = deepcopy(manifest["packages"])
        llm, processor, params_class = None, None, None
        if not args.check_only and len(rows) < len(ids):
            from huggingface_hub import snapshot_download
            from transformers import AutoProcessor
            from vllm import LLM, SamplingParams
            params_class = SamplingParams
            model_path = (str(Path(args.model_path).expanduser().resolve()) if Path(args.model_path).expanduser().is_dir()
                          else snapshot_download(args.model_path, revision=args.model_revision))
            model_config = json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8"))
            if model_config.get("quantization_config") or model_config.get("model_type") != "qwen3_vl":
                raise ValueError("Requires an unquantized Qwen3-VL checkpoint")
            manifest.update(resolved_model_path=model_path, model_config=model_config)
            (out / "requirements.freeze.txt").write_text(val.command_output([sys.executable, "-m", "pip", "freeze"]), encoding="utf-8")
            (out / "environment.txt").write_text(val.command_output(["nvidia-smi"]), encoding="utf-8")
            monitor = val.GpuMonitor(args.monitor_gpu)
            monitor.thread.start()
            load_start = time.perf_counter()
            engine = {k: p[k] for k in ("max_model_len", "gpu_memory_utilization", "max_num_seqs",
                      "max_num_batched_tokens", "enable_chunked_prefill", "enforce_eager", "dtype", "kv_cache_dtype")}
            llm = LLM(model=model_path, tokenizer=model_path, seed=p["sampling_recipe"]["seed"],
                      generation_config="vllm", limit_mm_per_prompt={"image": 7},
                      mm_processor_kwargs={"min_pixels": p["min_pixels"], "max_pixels": p["max_pixels"]}, **engine)
            processor = AutoProcessor.from_pretrained(model_path)
            template = processor.chat_template
            (out / "chat_template.txt").write_text(template if isinstance(template, str) else
                                                   json.dumps(template, ensure_ascii=False), encoding="utf-8")
            session["model_load_seconds"] = time.perf_counter() - load_start
        val.write_json(manifest_path, manifest)
        filename = out / "inputs.jsonl" if args.check_only else pred_path
        with filename.open("w" if args.check_only else "a", encoding="utf-8") as stream:
            for index, ((dataset, pos, subject), meta) in enumerate(zip(selection, metadata)):
                if not args.check_only and index < len(rows):
                    continue
                ex = dataset[pos]
                messages, choices, details = build_reference_message(ex, args.benchmark, args.setting)
                entry = {**meta, **details}
                if meta["question_type"] == "multiple-choice" and ex["answer"] not in choices:
                    raise ValueError(f"Invalid gold label: {meta['id']}")
                if args.check_only:
                    stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    if (index + 1) % 100 == 0:
                        print(f"[check] {index+1}/{len(ids)}", flush=True)
                    continue
                token_count = count_input_tokens(processor, messages, p)
                generated = generate_one(llm, messages, choices, meta["question_type"], p, params_class, token_count)
                row = {**meta, **generated, "answer": ex["answer"], "mode": "free", "setting": args.setting,
                       "choices": choices, "input": entry, "input_sha256": input_hash(entry)}
                row["correct"] = score_answer(row["answer"], row["parsed_answer"], row["question_type"])
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                rows.append(row)
                print(f"[progress] {len(rows)}/{len(ids)} tokens={row['output_tokens']} "
                      f"finish={row['finish_reason']} seconds={row['batch_seconds']:.1f}", flush=True)
        if not args.check_only:
            check_coverage(rows, args.benchmark)
            validate_rows(rows, ids, p)
        success = True
        manifest["status"] = "inputs_checked_no_inference" if args.check_only else "complete"
    except BaseException as exc:
        manifest.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        session.update(seconds=time.perf_counter()-start, success=success)
        if monitor is not None:
            session["gpu"] = monitor.stop()
        if success and not args.check_only:
            summary = make_summary(rows)
            summary.update(total_seconds=sum(s["seconds"] for s in manifest["sessions"]),
                           complete_900=args.benchmark == "mmmu-val", sessions=manifest["sessions"])
            val.write_json(out / "summary.json", summary)
            print(f"[done] {summary['n']} questions, accuracy={summary['accuracy']:.2%}, "
                  f"unparsed={summary['unparsed']}, length={summary['length_limited']}", flush=True)
        val.write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
