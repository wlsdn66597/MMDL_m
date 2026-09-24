from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_mmmu as val
import eval_free_reproduction as free
from scripts.rescore_free_qwen import extract, extraction_input


class Params:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeLLM:
    def __init__(self, text="B", reason="stop", count=2, fail_at=None):
        self.text, self.reason, self.count, self.fail_at = text, reason, count, fail_at
        self.calls = []

    def chat(self, messages, sampling_params, use_tqdm):
        self.calls.append((messages, sampling_params))
        if len(self.calls) == self.fail_at:
            raise RuntimeError("simulated interruption")
        return [NS(prompt_token_ids=[1]*20, outputs=[NS(text=self.text, token_ids=[1]*self.count,
                   finish_reason=self.reason, stop_reason=None)])]


def example(kind="multiple-choice", options=None):
    return dict(id="validation_Math_1", question="Read <image 1>.", question_type=kind,
                options=options if options is not None else ["one", "two"], answer="B",
                image_1=Image.new("RGB", (16, 16), "white"))


class FreeTests(unittest.TestCase):
    def test_reference_prompts_and_no_gold(self):
        ex = example()
        messages, choices, audit = free.build_reference_message(ex, "mmmu-val", "standard")
        self.assertEqual([c["type"] for c in messages[0]["content"]], ["image_url", "text"])
        self.assertEqual(messages[0]["content"][-1]["text"],
                         "Question: Read <image 1>.\nOptions:\nA. one\nB. two\n" + free.MMMU_SUFFIX)
        ex["answer"] = "PRIVATE_GOLD"
        self.assertEqual(messages, free.build_reference_message(ex, "mmmu-val", "standard")[0])
        ex["question_type"] = "open"
        messages, choices, _ = free.build_reference_message(ex, "mmmu-val", "standard")
        self.assertEqual(choices, {})
        self.assertEqual(messages[0]["content"][-1]["text"], "Question: Read <image 1>.")

    def test_pro_vision_hides_question_and_options(self):
        ex = example(options=[str(i) for i in range(12)])
        ex["image"] = ex["image_1"]
        messages, choices, _ = free.build_reference_message(ex, "mmmu-pro", "vision")
        self.assertEqual(len(choices), 12)
        self.assertEqual(messages[0]["content"][-1]["text"], free.VISION_PROMPT)
        standard = free.build_reference_message(ex, "mmmu-pro", "standard-10")[0]
        self.assertTrue(standard[0]["content"][-1]["text"].endswith(free.PRO_SUFFIX))

    def test_single_unconstrained_call_and_early_eos(self):
        llm = FakeLLM()
        row = free.generate_one(llm, [], {"A": "one", "B": "two"}, "multiple-choice", free.profile(), Params, 20)
        self.assertEqual(row["parsed_answer"], "B")
        self.assertEqual(len(llm.calls), 1)
        params = llm.calls[0][1]
        self.assertEqual(params.max_tokens, 32768)
        self.assertEqual(params.n, 1)
        self.assertFalse(hasattr(params, "structured_outputs"))
        self.assertFalse(hasattr(params, "ignore_eos"))
        self.assertEqual(row["output_tokens"], 2)

    def test_context_overflow_does_not_call_model(self):
        llm = FakeLLM()
        with self.assertRaisesRegex(ValueError, "exceeds context"):
            free.generate_one(llm, [], {}, "open", free.profile(), Params, 32769)
        self.assertFalse(llm.calls)

    def test_premature_length_stop_rejected(self):
        with self.assertRaisesRegex(ValueError, "shortened budget"):
            free.generate_one(FakeLLM(reason="length", count=10), [], {}, "open", free.profile(), Params, 20)
        row = free.generate_one(FakeLLM(text="unfinished", reason="length", count=32768), [], {},
                                "open", free.profile(), Params, 20)
        self.assertIsNone(row["parsed_answer"])

    def test_offline_and_judge_extraction_are_separate(self):
        row = {"id": "x", "subject": "Math", "question_type": "multiple-choice", "answer": "B",
               "choices": {"A": "cat", "B": "dog"}, "raw_response": "Not enough information.",
               "input": {"question": "What is shown?"}}
        prompt = extraction_input(row)[2]
        changed_gold = dict(row, answer="A")
        self.assertEqual(prompt, extraction_input(changed_gold)[2])
        self.assertFalse(extract(row)["resolved"])
        response = {"choices": [{"finish_reason": "stop", "message": {"content": "B"}}]}
        scored = extract(row, lambda _: response)
        self.assertTrue(scored["correct"])
        self.assertEqual(scored["judge_response"], response)
        response["choices"][0]["finish_reason"] = "length"
        self.assertFalse(extract(row, lambda _: response)["resolved"])

    def test_complete_900_resume_rechecks_scores_and_never_reruns_saved_rows(self):
        # End-to-end orchestration with full coverage, mocked model/processor and no downloads/GPU.
        image = Image.new("RGB", (8, 8), "white")
        class Dataset:
            def __init__(self):
                self.rows = []
                for subject in val.SUBJECTS:
                    for i in range(30):
                        kind = "open" if len(self.rows) < 53 else "multiple-choice"
                        self.rows.append(dict(example(kind), id=f"validation_{subject}_{i}",
                                              subject=subject, image_1=image))
            def __getitem__(self, key):
                return [r[key] for r in self.rows] if isinstance(key, str) else self.rows[key]
            def __len__(self):
                return len(self.rows)
        class Processor:
            chat_template = "fixture"
            def apply_chat_template(self, *args, **kwargs):
                return "fixture"
            def __call__(self, *args, **kwargs):
                return {"input_ids": [[1]*20]}
        ds = Dataset()
        selection = [(ds, i, row["subject"]) for i, row in enumerate(ds.rows)]
        llm1, llm2 = FakeLLM(fail_at=3), FakeLLM()
        modules = {"huggingface_hub": NS(snapshot_download=lambda *a, **k: "unused"),
                   "transformers": NS(AutoProcessor=NS(from_pretrained=lambda _: Processor())),
                   "vllm": NS(LLM=lambda **kw: current[0], SamplingParams=Params)}
        monitor = NS(thread=NS(start=lambda: None), stop=lambda: {})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, out = root / "model", root / "run"
            model.mkdir()
            (model / "config.json").write_text('{"model_type":"qwen3_vl"}', encoding="utf-8")
            argv = ["--model-path", str(model), "--output-dir", str(out)]
            current = [llm1]
            with patch.dict(sys.modules, modules), patch.object(free, "load_selection", return_value=(selection, {"test":"same"},900)), \
                    patch.object(val, "GpuMonitor", return_value=monitor), patch.object(val, "command_output", return_value="fixture"), \
                    redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "simulated"):
                    free.main(argv)
                self.assertEqual(len(free.read_jsonl(out / "predictions.jsonl")), 2)
                current[0] = llm2
                free.main(argv + ["--resume"])
                self.assertEqual(len(llm2.calls), 898)
                free.main(argv + ["--resume"])
                self.assertEqual(len(llm2.calls), 898)
            manifest, summary, rows = free.validate_run(out)
            self.assertEqual(summary["n"], 900)
            self.assertEqual(summary["calls"], 900)
            self.assertEqual(summary["question_types"]["open"]["n"], 53)
            self.assertEqual(len(manifest["sessions"]), 2)
            broken = deepcopy(rows)
            broken[0]["correct"] = not broken[0]["correct"]
            with self.assertRaisesRegex(ValueError, "scoring"):
                free.validate_rows(broken, [r["id"] for r in rows], free.profile())
            with self.assertRaisesRegex(ValueError, "ordered prefix"):
                free.validate_rows(rows[::-1], [r["id"] for r in rows], free.profile())


if __name__ == "__main__":
    unittest.main()
