from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from contextlib import redirect_stdout
import io
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_mmmu as val
from eval_output_policy import (arguments, configuration, generate_policy, score_answer, select_indices,
                                summarize, paired_input_fingerprint, run)
from baseline_contract import (check_coverage, aggregate_rows, load_profile, validate_profile, validate_run,
                               write_report, RUNS, PROFILE_PATH, PROFILE_8192_PATH)
from scripts.summarize_two_stage_baseline import summarize_suite, check_preflight


class Params:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def chat(self, messages, sampling_params, use_tqdm):
        self.calls.append((deepcopy(messages), sampling_params))
        text, reason = next(self.responses)
        return [NS(prompt_token_ids=[1]*20, outputs=[NS(text=text, token_ids=[1, 2],
                                                       finish_reason=reason, stop_reason=None)])]

    def get_tokenizer(self):
        return NS(get_chat_template=lambda: "test template")


def full_metadata(benchmark):
    if benchmark == "mmmu-val":
        rows = [{"id": f"validation_{subject}_{i}", "subject": subject, "question_type": "multiple-choice"}
                for subject in val.SUBJECTS for i in range(30)]
        for row in rows[:53]:
            row["question_type"] = "open"
        return rows
    return [{"id": f"test_Math_{i}", "subject": "Math", "question_type": "multiple-choice"} for i in range(1730)]


def args_for(benchmark="mmmu-val", setting="standard", profile_path=PROFILE_PATH):
    return NS(**load_profile(profile_path)["locked_arguments"], benchmark=benchmark, setting=setting,
              model_path=val.MODEL, model_revision=val.MODEL_REV, open_unused=None,
              check_only=False, data_root=None, checked_inputs=None)


def write_fixture(folder, benchmark, setting, profile_path=PROFILE_PATH):
    folder.mkdir()
    args = args_for(benchmark, setting, profile_path)
    rows, inputs = [], []
    for meta in full_metadata(benchmark):
        is_open = meta["question_type"] == "open"
        entry = dict(meta, messages_without_image_bytes=[], images=[], choices={} if is_open else {"A": "one", "B": "two"})
        entry["input_sha256"] = paired_input_fingerprint(entry)
        inputs.append(entry)
        stages = [{"stage": "reasoning", "raw_response": "draft", "output_tokens": 2, "finish_reason": "stop"},
                  {"stage": "final", "raw_response": "2" if is_open else "B", "output_tokens": 1, "finish_reason": "stop"}]
        rows.append(dict(meta, answer="2" if is_open else "B", parsed_answer=[2.0] if is_open else "B", correct=True,
                         input_sha256=entry["input_sha256"], raw_response=stages[-1]["raw_response"], finish_reason="stop",
                         output_tokens=3, batch_seconds=.01, reasoning_length_limited=False,
                         any_stage_length_limited=False, stages=stages))
    config = configuration(args)
    config["selected_ids_sha256"] = val.canonical_sha256([r["id"] for r in rows])
    profile = load_profile(profile_path)
    manifest = dict(status="complete", arguments=vars(args), source_n=len(rows), selected_n=len(rows),
                    evaluation_profile=dict(name=profile["profile_name"], sha256=val.canonical_sha256(profile)),
                    evaluation_signature=dict(config=config, sha256=val.canonical_sha256(config)),
                    model_config={}, code_sha256={"test": "same"}, python="test", packages={})
    summary = dict(summarize(rows), **aggregate_rows(rows), complete_900=benchmark == "mmmu-val",
                   gpu={}, total_seconds=30, model_load_seconds=1)
    val.write_json(folder / "manifest.json", manifest)
    val.write_json(folder / "summary.json", summary)
    val.write_json(folder / "evaluation_profile.json", profile)
    val.write_json(folder / "selected_ids.json", [r["id"] for r in rows])
    for name, data in (("predictions.jsonl", rows), ("inputs.jsonl", inputs)):
        (folder / name).write_text("\n".join(json.dumps(r) for r in data)+"\n", encoding="utf-8")
    return manifest, summary, rows


class FullBaselineTests(unittest.TestCase):
    def test_full_coverage_rejects_drop_duplicate_wrong_type_and_subject(self):
        rows = full_metadata("mmmu-val")
        check_coverage(rows, "mmmu-val")
        for bad in (rows[:-1], rows[:-1] + [rows[0]]):
            with self.assertRaises(ValueError):
                check_coverage(bad, "mmmu-val")
        for field, value in (("question_type", "multiple-choice"), ("subject", "Unknown")):
            bad = deepcopy(rows)
            bad[0][field] = value
            with self.assertRaises(ValueError):
                check_coverage(bad, "mmmu-val")
        check_coverage(full_metadata("mmmu-pro"), "mmmu-pro")

    def test_open_is_selected_only_when_requested(self):
        ids, kinds = ["a", "b"], ["multiple-choice", "open"]
        self.assertEqual(len(select_indices(ids, kinds, 0, 1)), 1)
        self.assertEqual(len(select_indices(ids, kinds, 0, 1, include_open=True)), 2)

    def test_profile_applies_4096_and_rejects_override(self):
        argv = ["--mode", "two-stage", "--evaluation-profile", str(PROFILE_PATH), "--output-dir", "unused"]
        args = arguments(argv)
        self.assertEqual(args.reasoning_tokens, 4096)
        self.assertEqual(args.per_subject, 0)
        self.assertTrue(args.include_open and args.require_full)
        with self.assertRaises(ValueError):
            arguments(argv + ["--reasoning-tokens", "1024"])
        other = deepcopy(args)
        other.model_path = "trained-model"
        self.assertEqual(configuration(args), configuration(other))

    def test_uniform_8192_profile_and_submission_validation(self):
        argv = ["--mode", "two-stage", "--evaluation-profile", str(PROFILE_8192_PATH),
                "--output-dir", "unused"]
        args = arguments(argv)
        self.assertEqual(args.reasoning_tokens, 8192)
        self.assertEqual(args.per_subject, 0)
        self.assertTrue(args.include_open and args.require_full)
        with self.assertRaises(ValueError):
            arguments(argv + ["--reasoning-tokens", "4096"])
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            folder = repo / "run"
            manifest, summary, _ = write_fixture(folder, "mmmu-val", "standard", PROFILE_8192_PATH)
            write_report(folder, manifest, summary)
            self.assertIn("8192-token", (folder / "report_draft.md").read_text(encoding="utf-8"))
            validate_run(folder, "mmmu-val", "standard", require_base=True)
            import scripts.prepare_submission as prepare
            with patch.object(prepare, "__file__", str(repo / "scripts" / "prepare_submission.py")), \
                 patch.object(sys, "argv", ["prepare_submission.py", str(folder), "--name", "candidate8192"]), \
                 redirect_stdout(io.StringIO()):
                prepare.main()
            self.assertTrue((repo / "artifacts" / "candidate8192" / "selected_ids.json").is_file())

    def test_open_generation_scores_final_only_and_keeps_images(self):
        messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "fake"}},
                                                   {"type": "text", "text": "Question\nDIRECT"}]}]
        llm = FakeLLM([("Wrong draft answer 99", "length"), ("2", "stop")])
        row = generate_policy(llm, messages, {}, "two-stage", "DIRECT", args_for(), Params, Params, "open")
        self.assertTrue(score_answer("2", row["parsed_answer"], "open"))
        self.assertFalse(score_answer("99", row["parsed_answer"], "open"))
        self.assertEqual(llm.calls[0][1].max_tokens, 4096)
        self.assertEqual(llm.calls[1][1].max_tokens, 128)
        self.assertFalse(hasattr(llm.calls[1][1], "structured_outputs"))
        self.assertEqual(llm.calls[1][0][0]["content"][0], messages[0]["content"][0])
        self.assertNotIn("option letter", llm.calls[1][0][-1]["content"])
        self.assertTrue(row["reasoning_length_limited"])
        self.assertEqual(row["parsing"]["mode"], "official_open")

    def test_open_truncated_or_empty_final_is_incorrect(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": "DIRECT"}]}]
        for text, reason in (("2", "length"), ("", "stop")):
            row = generate_policy(FakeLLM([("draft", "stop"), (text, reason)]), messages, {},
                                  "two-stage", "DIRECT", args_for(), Params, Params, "open")
            self.assertIsNone(row["parsed_answer"])
            self.assertFalse(score_answer("2", row["parsed_answer"], "open"))

    def test_all_four_runs_and_reports_include_open_and_refuse_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, benchmark, setting in RUNS:
                manifest, summary, rows = write_fixture(root / name, benchmark, setting)
                write_report(root / name, manifest, summary)
            result = summarize_suite(root)
            self.assertEqual(result["mmmu_val"]["question_types"]["open"]["n"], 53)
            self.assertAlmostEqual(result["mmmu_val"]["macro_accuracy"], 1)
            validate_run(root / "mmmu_val", "mmmu-val", "standard", require_base=True)
            folder = root / "vision"
            data = json.loads((folder / "summary.json").read_text())
            data["correct"] -= 1
            val.write_json(folder / "summary.json", data)
            with self.assertRaisesRegex(ValueError, "Summary"):
                summarize_suite(root)

    def test_runner_open_end_to_end_without_gpu(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            model.mkdir()
            val.write_json(model / "config.json", {})
            args = args_for()
            args.require_full = False
            args.model_path = str(model)
            out = root / "run"
            out.mkdir()
            ex = dict(id="validation_Math_0", question="How many?", question_type="open", options="[]",
                      answer="2", image_1=Image.new("RGB", (4, 4)))
            llm = FakeLLM([("Count objects", "stop"), ("2", "stop")])
            modules = {"huggingface_hub": NS(snapshot_download=lambda *a, **k: str(model)),
                       "vllm": NS(LLM=lambda **kwargs: llm, SamplingParams=Params),
                       "vllm.sampling_params": NS(StructuredOutputsParams=Params)}
            manifest = {}
            with patch.dict(sys.modules, modules), patch("eval_output_policy.load_selection",
                    return_value=([([ex], 0, "Math")], {"Math": "test"}, 900)):
                summary = run(args, out, manifest)
            self.assertEqual(summary["correct"], 1)
            self.assertEqual(summary["question_types"]["open"]["n"], 1)
            self.assertFalse(summary["complete_900"])
            entry = json.loads((out / "inputs.jsonl").read_text())
            self.assertNotIn("answer", entry)
            self.assertNotIn("option letter", entry["final_user_instruction"])

    def test_preflight_manifest_lifecycle_includes_all_900_without_loading_model(self):
        import eval_output_policy as policy
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "checked"
            args = args_for()
            args.output_dir = out
            args.evaluation_profile = PROFILE_PATH
            args.check_only = True
            args.monitor_gpu = "0"
            examples = [dict(r, question="How many?", answer="2" if r["question_type"] == "open" else "B",
                             options="[]" if r["question_type"] == "open" else ["one", "two"],
                             image_1=Image.new("RGB", (2, 2))) for r in full_metadata("mmmu-val")]
            selection = [(examples, i, ex["subject"]) for i, ex in enumerate(examples)]
            monitor = NS(thread=NS(start=lambda: None), stop=lambda: {})
            with patch.object(policy, "arguments", return_value=args), \
                 patch.object(policy, "load_selection", return_value=(selection, {"test": "same"}, 900)), \
                 patch.object(val, "GpuMonitor", return_value=monitor), \
                 patch.object(val, "command_output", return_value="test environment"), redirect_stdout(io.StringIO()):
                policy.main()
            ids = check_preflight(out, "mmmu-val", "standard")
            self.assertEqual(len(ids), 900)
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "inputs_checked_no_inference")
            self.assertEqual(manifest["selected_question_types"], {"multiple-choice": 847, "open": 53})
            self.assertIn("vendor/mmmu_eval_utils.py", manifest["code_sha256"])
            self.assertEqual((out / "predictions.jsonl").read_text(), "")

    def test_submission_accepts_full_two_stage_and_copies_selected_ids(self):
        import scripts.prepare_submission as prepare
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            folder = repo / "run"
            manifest, summary, _ = write_fixture(folder, "mmmu-val", "standard")
            write_report(folder, manifest, summary)
            with patch.object(prepare, "__file__", str(repo / "scripts" / "prepare_submission.py")), \
                 patch.object(sys, "argv", ["prepare_submission.py", str(folder), "--name", "baseline"]), \
                 redirect_stdout(io.StringIO()):
                prepare.main()
            self.assertTrue((repo / "reports" / "mmmu_baseline.md").is_file())
            self.assertTrue((repo / "artifacts" / "baseline" / "selected_ids.json").is_file())
            self.assertTrue((repo / "assignment" / "assignment1.md").is_file())

    def test_preflight_mismatch_fails_before_gpu_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            check = root / "check"
            check.mkdir()
            val.write_json(check / "manifest.json", {"status": "inputs_checked_no_inference", "evaluation_signature": {}})
            (check / "inputs.jsonl").write_text('{}\n')
            out = root / "out"
            out.mkdir()
            args = args_for()
            args.require_full = False
            args.checked_inputs = check
            selection = [([{"id": "q", "question_type": "open"}], 0, "Math")]
            with patch("eval_output_policy.load_selection", return_value=(selection, {}, 900)):
                with self.assertRaisesRegex(ValueError, "Preflight"):
                    run(args, out, {})


if __name__ == "__main__":
    unittest.main()
