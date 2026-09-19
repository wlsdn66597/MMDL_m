"""CPU-only checks: data leakage, image mapping, scoring and output completeness."""
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_mmmu as runner
from vendor.mmmu_eval_utils import eval_open, parse_open_response


class TinyImage:
    width, height = 32, 32
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
    def test_images_and_no_gold_leak(self):
        ex = example()
        ex["answer"] = "SECRET_GOLD_ANSWER"
        messages, _, audit = runner.build_message(ex)
        serialized = json.dumps(messages)
        self.assertNotIn("SECRET_GOLD", serialized)
        self.assertIn("[Image 2] and [Image 1]", serialized)
        self.assertEqual([im["number"] for im in audit["images"]], [1, 2])
        self.assertEqual(sum(c["type"] == "image_url" for c in messages[0]["content"]), 2)

    def test_missing_image_in_option_raises(self):
        ex = example()
        ex["options"] = "['<image 3>', 'second']"
        with self.assertRaisesRegex(ValueError, "Missing image"):
            runner.build_message(ex)

    def test_mc_last_candidate_and_no_random_fallback(self):
        choices = {"A": "first", "B": "second"}
        answer, info = runner.mc_parse("Consider (A), but final answer: (B)", choices)
        self.assertEqual(answer, "B")
        self.assertEqual(len(info["candidates"]), 2)
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
        class FakeLLM:
            def __init__(self, **kwargs):
                pass
            def get_tokenizer(self):
                return SimpleNamespace(get_chat_template=lambda: "fake chat template")
            def chat(self, messages, **kwargs):
                results = []
                for message in messages:
                    text = message[0]["content"][-1]["text"]
                    response = "Final answer: (A)" if "Choices:" in text else "Final answer: 2"
                    results.append(SimpleNamespace(prompt_token_ids=[1], outputs=[SimpleNamespace(
                        text=response, token_ids=[1, 2], finish_reason="stop", stop_reason=None)]))
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
                min_pixels=65536, max_pixels=589824, max_tokens=2048, monitor_gpu="0")
            modules = {"datasets": dataset_module, "huggingface_hub": hf_module, "vllm": vllm_module}
            with patch.dict(sys.modules, modules), patch("builtins.print"):
                manifest = {"arguments": vars(args), "packages": {"vllm": "mock"},
                            "started_utc": "test", "command": "test"}
                summary = runner.run(args, Path(tmp), manifest)
                self.assertEqual(summary["n"], 900)
                self.assertEqual(summary["macro_accuracy"], 1.0)
                self.assertTrue(summary["complete_900"])
                summary.update(gpu={"peak_device_used_mib_sampled": None}, total_seconds=1)
                runner.make_report(Path(tmp), manifest, summary)
                self.assertIn("**900**", (Path(tmp) / "report_draft.md").read_text(encoding="utf-8"))
                self.assertEqual(len((Path(tmp) / "predictions.jsonl").read_text().splitlines()), 900)
                incomplete[0] = True
                with self.assertRaisesRegex(RuntimeError, "output count mismatch"):
                    runner.run(args, Path(tmp), manifest)


if __name__ == "__main__":
    unittest.main()
