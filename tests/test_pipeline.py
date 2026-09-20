"""CPU-only checks: data leakage, image mapping, scoring and output completeness."""
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_mmmu as runner
import eval_mmmu_pro as pro_runner
from scripts.compare_runs import config_differences
from scripts.analyze_research_metrics import (
    input_features, paired_metric, render_markdown, summarize_run, wilson_interval,
)
from scripts.summarize_mmmu_pro import paired_correctness
from vendor.mmmu_eval_utils import eval_open, parse_open_response


class TinyImage:
    width, height = 32, 32
    mode, info = "RGB", {}
    def convert(self, _):
        return self
    def save(self, buffer, format):
        buffer.write(b"unit-test-image")


def example(identifier="test", question_type="multiple-choice"):
    return dict(id=identifier, question="Read <image 2> and <image 1>.",
                question_type=question_type, options="['first', 'second']",
                image_1=TinyImage(), image_2=TinyImage(),
                answer="A" if question_type == "multiple-choice" else "2",
                explanation="SECRET_GOLD_EXPLANATION")


class PipelineTests(unittest.TestCase):
    def test_mmmu_pro_standard_and_vision_messages_do_not_leak_gold(self):
        standard = {
            "id": "pro-1", "question": "Compare <image 1>.",
            "options": "['one', 'two', 'three', 'four']", "answer": "B",
            "explanation": "SECRET_GOLD_EXPLANATION", "image_1": TinyImage(),
        }
        messages, choices, audit = pro_runner.build_message(standard, "standard-4")
        serialized = json.dumps(messages)
        self.assertNotIn("SECRET_GOLD", serialized)
        self.assertIn("[Image 1]", serialized)
        self.assertEqual(len(choices), 4)
        self.assertEqual(len(audit["images"]), 1)
        vision = {"id": "pro-2", "options": [str(i) for i in range(10)],
                  "answer": "J", "image": TinyImage()}
        messages, choices, _ = pro_runner.build_message(vision, "vision")
        text = messages[0]["content"][-1]["text"]
        self.assertIn("Answer: $LETTER", text)
        self.assertNotIn("(A) 0", text)
        self.assertEqual(len(choices), 10)

        standard["options"] = "['one', 'two', 'three', 'four', 'five']"
        _, choices, audit = pro_runner.build_message(standard, "standard-4")
        self.assertEqual(len(choices), 5)
        self.assertEqual(audit["option_count"], 5)
        vision["options"] = [str(i) for i in range(12)]
        _, choices, audit = pro_runner.build_message(vision, "vision")
        self.assertEqual(len(choices), 12)
        self.assertEqual(audit["option_count"], 12)
        self.assertIn("L", choices)

    def test_mmmu_pro_parser_and_paired_correctness(self):
        choices = {chr(65 + index): str(index) for index in range(10)}
        parsed, info = pro_runner.pro_parse("Reasoning. Answer: J", choices)
        self.assertEqual(parsed, "J")
        self.assertEqual(info["mode"], "explicit_answer")
        rows_a = {"1": {"correct": True}, "2": {"correct": False}}
        rows_b = {"1": {"correct": False}, "2": {"correct": True}}
        metric = paired_correctness(rows_a, rows_b, ["1", "2"])
        self.assertEqual(metric["delta_b_minus_a_pp"], 0.0)
        self.assertEqual(metric["a_only"], 1)
        self.assertEqual(metric["b_only"], 1)
        self.assertEqual(metric["mcnemar_exact_p"], 1.0)

    def test_research_metrics_image_references_and_paired_transitions(self):
        input_row = {
            "images": [{"number": 1}, {"number": 2}],
            "messages_without_image_bytes": [{"content": [
                {"type": "text", "text": "Image 1:"},
                {"type": "image_url"},
                {"type": "text", "text": "Image 2:"},
                {"type": "image_url"},
                {"type": "text", "text": "Compare [Image 2] with [Image 1]."},
            ]}],
        }
        features = input_features(input_row)
        self.assertEqual(features["image_count_bucket"], "2")
        self.assertEqual(features["reference_pattern"], "multi-explicit-reference")
        self.assertEqual(features["reference_order"], "out-of-order")
        rows_a = [
            {"id": "1", "correct": True, "parsed_answer": "A"},
            {"id": "2", "correct": False, "parsed_answer": "B"},
        ]
        rows_b = [
            {"id": "1", "correct": False, "parsed_answer": "B"},
            {"id": "2", "correct": True, "parsed_answer": "A"},
        ]
        paired = paired_metric(rows_a, rows_b)
        self.assertEqual(paired["correct_to_wrong"], 1)
        self.assertEqual(paired["wrong_to_correct"], 1)
        self.assertEqual(paired["answer_change_rate"], 1.0)
        self.assertEqual(paired["mcnemar_exact_p"], 1.0)
        low, high = wilson_interval(5, 10)
        self.assertLess(low, .5)
        self.assertGreater(high, .5)

    def test_research_metrics_end_to_end_and_unique_batch_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            predictions = [
                {"id": "1", "subject": "Math", "question_type": "multiple-choice", "answer": "A",
                 "parsed_answer": "A", "correct": True, "parsing": {"mode": "exact_letter", "candidates": ["A"]},
                 "finish_reason": "stop", "output_tokens": 1, "input_tokens": 100,
                 "batch_id": "b1", "batch_seconds": 2.0},
                {"id": "2", "subject": "Math", "question_type": "multiple-choice", "answer": "B",
                 "parsed_answer": "A", "correct": False, "parsing": {"mode": "exact_letter", "candidates": ["A"]},
                 "finish_reason": "stop", "output_tokens": 1, "input_tokens": 120,
                 "batch_id": "b1", "batch_seconds": 2.0},
            ]
            inputs = [
                {"id": "1", "images": [{"number": 1}], "messages_without_image_bytes": [
                    {"content": [{"type": "text", "text": "Use [Image 1]."}]}]},
                {"id": "2", "images": [{"number": 1}, {"number": 2}], "messages_without_image_bytes": [
                    {"content": [{"type": "text", "text": "Compare [Image 1] and [Image 2]."}]}]},
            ]
            (run_dir / "predictions.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8")
            (run_dir / "inputs.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in inputs), encoding="utf-8")
            runner.write_json(run_dir / "manifest.json", {"arguments": {"prompt_style": "direct"}})
            runner.write_json(run_dir / "summary.json", {
                "total_seconds": 5.0, "gpu": {"peak_device_used_mib_sampled": 1000.0}})
            report = summarize_run(run_dir)
            self.assertEqual(report["overall"]["accuracy"], .5)
            self.assertEqual(report["efficiency"]["inference_seconds_unique_batches"], 2.0)
            self.assertEqual(report["efficiency"]["questions_per_inference_second"], 1.0)
            self.assertEqual(report["visual_structure"]["multi_minus_single_accuracy_pp"], -100.0)
            self.assertIn("Multi-image minus single-image accuracy", render_markdown(report))

    def test_fixed_profile_accepts_canonical_settings_and_rejects_override(self):
        profile_path = Path(__file__).resolve().parents[1] / "configs" / "mmmu_val_v1.json"
        profile, _, digest = runner.load_evaluation_profile(profile_path)
        args = SimpleNamespace(**profile["locked_arguments"])
        runner.validate_evaluation_profile(profile, args)
        signature = runner.evaluation_signature(args, digest)
        self.assertEqual(signature["config"]["profile_sha256"], digest)
        args.max_pixels -= 1
        with self.assertRaisesRegex(ValueError, "max-pixels"):
            runner.validate_evaluation_profile(profile, args)

    def test_comparison_config_ignores_checkpoint_but_detects_pipeline_change(self):
        profile_path = Path(__file__).resolve().parents[1] / "configs" / "mmmu_val_v1.json"
        profile, _, digest = runner.load_evaluation_profile(profile_path)
        args = SimpleNamespace(**profile["locked_arguments"])
        base = runner.evaluation_signature(args, digest)["config"]
        fine_tuned = dict(base)
        self.assertEqual(config_differences(base, fine_tuned), [])
        fine_tuned["max_tokens"] = 512
        self.assertEqual(config_differences(base, fine_tuned), [("max_tokens", 256, 512)])

    def test_palette_transparency_is_composited_on_white(self):
        img = Image.new("P", (1, 1))
        img.putpalette([255, 0, 0] + [0, 0, 0] * 255)
        img.info["transparency"] = 0
        converted = runner.image_as_rgb(img)
        self.assertEqual(converted.mode, "RGB")
        self.assertEqual(converted.getpixel((0, 0)), (255, 255, 255))

    def test_images_and_no_gold_leak(self):
        ex = example()
        ex["answer"] = "SECRET_GOLD_ANSWER"
        messages, _, audit = runner.build_message(ex)
        serialized = json.dumps(messages)
        self.assertNotIn("SECRET_GOLD", serialized)
        self.assertIn("[Image 2] and [Image 1]", serialized)
        self.assertEqual([im["number"] for im in audit["images"]], [1, 2])
        self.assertEqual(sum(c["type"] == "image_url" for c in messages[0]["content"]), 2)

    def test_cot_prompt_has_strict_final_answer_marker(self):
        messages, _, _ = runner.build_message(example(), prompt_style="cot")
        text = messages[0]["content"][-1]["text"]
        self.assertIn("Solve the problem step by step", text)
        self.assertIn("Final answer: (X)", text)
        self.assertNotIn("SECRET_GOLD", json.dumps(messages))

    def test_brief_cot_prompt_limits_reasoning_and_stops(self):
        messages, _, _ = runner.build_message(example(), prompt_style="cot-brief")
        text = messages[0]["content"][-1]["text"]
        self.assertIn("at most three short steps", text)
        self.assertIn("Stop immediately", text)
        self.assertIn("Final answer: (X)", text)

    def test_missing_image_in_option_raises(self):
        ex = example()
        ex["options"] = "['<image 3>', 'second']"
        with self.assertRaisesRegex(ValueError, "Missing image"):
            runner.build_message(ex)

    def test_mc_last_candidate_and_no_random_fallback(self):
        choices = {"A": "first", "B": "second"}
        answer, info = runner.mc_parse("Consider (A), but final answer: (B)", choices)
        self.assertEqual(answer, "B")
        self.assertEqual(info["mode"], "explicit_final")
        self.assertEqual(runner.mc_parse("**A**", choices)[0], "A")
        self.assertIsNone(runner.mc_parse("I cannot determine the answer.", choices)[0])
        self.assertIsNone(runner.mc_parse("", choices)[0])

    def test_official_open_numeric(self):
        self.assertTrue(eval_open("1,234.56", parse_open_response("Final answer: 1234.56")))
        self.assertFalse(eval_open("2", parse_open_response("Final answer: 3")))

    def test_complete_900_and_partial_return_rejected(self):
        class Dataset:
            _fingerprint = "fake-fingerprint"
            def __init__(self, subject):
                self.rows = [example(f"{subject}_{i}", "open" if i % 2 else "multiple-choice") for i in range(30)]
            def __len__(self):
                return len(self.rows)
            def __getitem__(self, key):
                return [r[key] for r in self.rows] if isinstance(key, str) else self.rows[key]

        incomplete = [False]
        truncate_mc = [False]
        class FakeLLM:
            def __init__(self, **kwargs):
                pass
            def get_tokenizer(self):
                return SimpleNamespace(get_chat_template=lambda: "fake chat template")
            def chat(self, messages, **kwargs):
                results = []
                for message in messages:
                    text = message[0]["content"][-1]["text"]
                    is_mc = "Choices:" in text
                    response = ("Reasoning mentions (A) and (B)" if is_mc and truncate_mc[0]
                                else "Final answer: (A)" if is_mc else "Final answer: 2")
                    results.append(SimpleNamespace(prompt_token_ids=[1], outputs=[SimpleNamespace(
                        text=response, token_ids=[1, 2],
                        finish_reason="length" if is_mc and truncate_mc[0] else "stop",
                        stop_reason=None)]))
                return results[:-1] if incomplete[0] else results

        dataset_module = ModuleType("datasets")
        dataset_module.load_dataset = lambda root, subject, **kw: Dataset(subject)
        hf_module = ModuleType("huggingface_hub")
        hf_module.snapshot_download = lambda *a, **k: None
        vllm_module = ModuleType("vllm")
        vllm_module.LLM = FakeLLM
        vllm_module.SamplingParams = lambda **kw: kw
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / runner.MODEL_REV
            model.mkdir()
            (model / "config.json").write_text("{}")
            args = SimpleNamespace(model_path=str(model), model_revision=runner.MODEL_REV,
                data_root="MMMU/MMMU", cache_dir=None, limit_per_subject=0, check_only=False,
                max_model_len=8192, gpu_memory_utilization=.85, batch_size=2,
                min_pixels=65536, max_pixels=589824, max_tokens=256, monitor_gpu="0",
                prompt_style="direct")
            modules = {"datasets": dataset_module, "huggingface_hub": hf_module, "vllm": vllm_module}
            with patch.dict(sys.modules, modules), patch("builtins.print"):
                manifest = {"arguments": vars(args), "packages": {"vllm": "mock"},
                            "started_utc": "test", "command": "test"}
                summary = runner.run(args, Path(tmp), manifest)
                self.assertEqual(summary["n"], 900)
                self.assertEqual(summary["correct"], 900)
                self.assertEqual(summary["macro_accuracy"], 1.0)
                self.assertTrue(summary["complete_900"])
                self.assertEqual(sum(row["n"] for row in summary["domains"]), 900)
                self.assertEqual(summary["question_types"][0]["n"], 450)
                summary.update(gpu={"peak_device_used_mib_sampled": None}, total_seconds=1)
                runner.make_report(Path(tmp), manifest, summary)
                report = (Path(tmp) / "report_draft.md").read_text(encoding="utf-8")
                self.assertIn("**900**", report)
                self.assertIn("Core discipline", report)
                self.assertEqual(len((Path(tmp) / "predictions.jsonl").read_text().splitlines()), 900)
                truncate_mc[0] = True
                truncated_summary = runner.run(args, Path(tmp), manifest)
                self.assertEqual(truncated_summary["unparsed"], 450)
                self.assertEqual(truncated_summary["length_limited"], 450)
                truncate_mc[0] = False
                incomplete[0] = True
                with self.assertRaisesRegex(RuntimeError, "output count mismatch"):
                    runner.run(args, Path(tmp), manifest)


if __name__ == "__main__":
    unittest.main()
