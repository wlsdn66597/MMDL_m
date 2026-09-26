"""Checks for the MMMU-val diagnostic and saved-draft replay helpers."""
from pathlib import Path
import sys
import unittest

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_mmmu as val
from eval_output_policy import (REASON_INSTRUCTION, SELECT_INSTRUCTION,
                                paired_input_fingerprint, stage_messages)
from scripts.audit_two_stage_val import report
from scripts.replay_two_stage_final_val import grouped_comparison, source_messages


class TwoStageDiagnosticTests(unittest.TestCase):
    def test_replay_reconstructs_original_images_and_saved_draft(self):
        example = {"id": "validation_Math_0", "question_type": "multiple-choice",
                   "question": "What is shown in <image 1>?", "options": ["one", "two"],
                   "answer": "B", "image_1": Image.new("RGB", (3, 3), "white")}
        messages, choices, details = val.build_message(example, "direct")
        suffix = val.MC_TEMPLATE.split("\n\n")[-1]
        input_row = {"id": example["id"], "question_type": example["question_type"],
                     "choices": choices, "input_sha256": paired_input_fingerprint(details),
                     "reasoning_messages_without_image_bytes": stage_messages(
                         details["messages_without_image_bytes"], suffix, REASON_INSTRUCTION),
                     "final_user_instruction": SELECT_INSTRUCTION}
        baseline_row = {"id": example["id"], "question_type": example["question_type"],
                        "answer": "B", "choices": choices,
                        "input_sha256": input_row["input_sha256"],
                        "stages": [{"raw_response": "The image shows two objects. Tentative answer B."}]}
        replay_messages, replay_choices = source_messages(example, input_row, baseline_row)
        self.assertEqual(replay_choices, choices)
        self.assertEqual(replay_messages[0]["content"][1], messages[0]["content"][1])
        self.assertEqual(replay_messages[1]["content"], baseline_row["stages"][0]["raw_response"])
        self.assertEqual(replay_messages[2]["content"], SELECT_INSTRUCTION)
        example["question"] = "A different question"
        with self.assertRaisesRegex(ValueError, "changed"):
            source_messages(example, input_row, baseline_row)

    def test_groups_count_paired_changes_and_draft_status(self):
        rows = [
            {"question_type": "multiple-choice", "draft_truncated": True,
             "baseline_correct": False, "correct": True},
            {"question_type": "multiple-choice", "draft_truncated": False,
             "baseline_correct": True, "correct": False},
            {"question_type": "open", "draft_truncated": False,
             "baseline_correct": True, "correct": True},
        ]
        groups = grouped_comparison(rows)
        self.assertEqual(groups["overall"]["baseline_only"], 1)
        self.assertEqual(groups["overall"]["greedy_only"], 1)
        self.assertEqual(groups["draft_truncated"]["greedy_correct"], 1)
        self.assertEqual(groups["open"]["n"], 1)

    def test_audit_reports_group_accuracy_from_saved_rows(self):
        rows = [{"question_type": "multiple-choice", "correct": True,
                 "reasoning_length_limited": False, "batch_seconds": 30,
                 "stages": [{"output_tokens": 20, "finish_reason": "stop"},
                            {"output_tokens": 1, "finish_reason": "stop"}]},
                {"question_type": "open", "correct": False,
                 "reasoning_length_limited": True, "batch_seconds": 90,
                 "stages": [{"output_tokens": 4096, "finish_reason": "length"},
                            {"output_tokens": 2, "finish_reason": "stop"}]}]
        text = report(rows)
        self.assertIn("all                      n=  2 correct=  1 accuracy=50.00%", text)
        self.assertIn("draft_finish_reasons={'stop': 1, 'length': 1}", text)


if __name__ == "__main__":
    unittest.main()
