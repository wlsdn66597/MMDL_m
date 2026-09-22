from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_output_policy import generate_policy, stage_messages, select_indices, summarize, REASON_INSTRUCTION, run
from eval_mmmu import canonical_sha256, write_json
from scripts.summarize_output_policies import report


class Params:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def chat(self, messages, sampling_params, use_tqdm):
        self.calls.append((deepcopy(messages), sampling_params))
        text, tokens, reason = next(self.responses)
        return [NS(prompt_token_ids=[1]*20, outputs=[NS(text=text, token_ids=[1]*tokens,
                                                      finish_reason=reason, stop_reason=None)])]

    def get_tokenizer(self):
        return NS(get_chat_template=lambda: 'fake template')


class OutputPolicyTests(unittest.TestCase):
    def setUp(self):
        self.args = NS(max_tokens=8192, reasoning_tokens=1024, final_tokens=16)
        self.choices = {chr(65+i): f'choice {i}' for i in range(12)}
        self.messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,fake"}},
            {"type": "text", "text": "Question/choices\nDIRECT"}]}]

    def test_constraint_uses_all_twelve_letters_and_same_input(self):
        llm = FakeLLM([('L', 1, 'stop')])
        row = generate_policy(llm, self.messages, self.choices, 'constrained', 'DIRECT', self.args, Params, Params)
        self.assertEqual(row['parsed_answer'], 'L')
        self.assertEqual(llm.calls[0][0], self.messages)
        self.assertEqual(llm.calls[0][1].structured_outputs.choice, list(self.choices))
        self.assertEqual(llm.calls[0][1].max_tokens, 16)

    def test_truncated_draft_still_selects_and_counts_both_calls(self):
        original = deepcopy(self.messages)
        llm = FakeLLM([('Unfinished draft', 1024, 'length'), ('B', 1, 'stop')])
        row = generate_policy(llm, self.messages, self.choices, 'two-stage', 'DIRECT', self.args, Params, Params)
        self.assertEqual(self.messages, original)
        self.assertEqual(row['output_tokens'], 1025)
        self.assertEqual(row['input_tokens'], 40)
        self.assertTrue(row['reasoning_length_limited'])
        self.assertEqual(row['finish_reason'], 'stop')
        self.assertEqual(row['parsed_answer'], 'B')
        final_messages = llm.calls[1][0]
        self.assertEqual(final_messages[0]['content'][0], original[0]['content'][0])
        self.assertEqual(final_messages[1]['content'], 'Unfinished draft')
        self.assertNotIn('DIRECT', final_messages[0]['content'][-1]['text'])
        self.assertIn(REASON_INSTRUCTION, final_messages[0]['content'][-1]['text'])
        self.assertFalse(hasattr(llm.calls[0][1], 'structured_outputs'))

    def test_constraint_violation_is_not_hidden_by_parser(self):
        with self.assertRaisesRegex(RuntimeError, 'contract'):
            generate_policy(FakeLLM([('Answer: B', 3, 'stop')]), self.messages, self.choices,
                            'constrained', 'DIRECT', self.args, Params, Params)

    def test_free_generation_retains_budget_and_truncation_policy(self):
        llm = FakeLLM([('Perhaps (B). I should reconsider', 8192, 'length')])
        row = generate_policy(llm, self.messages, self.choices, 'free', 'DIRECT', self.args, Params, Params)
        self.assertIsNone(row['parsed_answer'])
        self.assertEqual(llm.calls[0][1].max_tokens, 8192)
        self.assertFalse(hasattr(llm.calls[0][1], 'structured_outputs'))
        self.assertEqual(len(row['stages']), 1)

    def test_selection_is_order_independent_and_excludes_open(self):
        ids = ['a', 'b', 'c', 'd']
        kinds = ['multiple-choice', 'open', 'multiple-choice', 'multiple-choice']
        selected = [ids[i] for i in select_indices(ids, kinds, 2, 3407)]
        rev = [ids[::-1][i] for i in select_indices(ids[::-1], kinds[::-1], 2, 3407)]
        self.assertEqual(selected, rev)
        self.assertNotIn('b', selected)
        self.assertEqual(len(select_indices(ids, kinds, 0, 3407)), 3)

    def test_prompt_mismatch_fails_instead_of_appending_conflicting_instruction(self):
        with self.assertRaises(ValueError):
            stage_messages(self.messages, 'WRONG')

    def test_runner_writes_auditable_two_stage_results_without_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / 'model'
            model.mkdir()
            write_json(model/'config.json', {})
            out = root/'run'
            out.mkdir()
            args = NS(mode='two-stage', benchmark='mmmu-val', setting='standard',
                      model_path=str(model), model_revision='fixture', check_only=False,
                      max_tokens=8192, reasoning_tokens=1024, final_tokens=16,
                      max_model_len=16384, min_pixels=1003520, max_pixels=4014080,
                      gpu_memory_utilization=.85, selection_seed=3407, per_subject=4)
            ex = dict(id='q', question='Which image?', question_type='multiple-choice',
                      options=['first', 'second'], answer='B', image_1=Image.new('RGB', (4, 4)))
            llm = FakeLLM([('draft', 1024, 'length'), ('B', 1, 'stop')])
            modules = {'huggingface_hub': NS(snapshot_download=lambda *a, **k: str(model)),
                       'vllm': NS(LLM=lambda **kwargs: llm, SamplingParams=Params),
                       'vllm.sampling_params': NS(StructuredOutputsParams=Params)}
            manifest = {}
            with patch.dict(sys.modules, modules), patch('eval_output_policy.load_selection',
                       return_value=([([ex], 0, 'Math')], {'Math': 'fixture'}, 900)):
                summary = run(args, out, manifest)
            self.assertEqual(manifest['status'], 'complete')
            self.assertEqual(summary['correct'], 1)
            self.assertEqual(summary['calls'], 2)
            row = json.loads((out/'predictions.jsonl').read_text())
            inputs = json.loads((out/'inputs.jsonl').read_text())
            self.assertEqual(row['output_tokens'], 1025)
            self.assertEqual(inputs['input_sha256'], row['input_sha256'])
            self.assertIn('reasoning_messages_without_image_bytes', inputs)
            self.assertNotIn('answer', inputs)

    def test_report_checks_inputs_and_counts_draft_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for mode in ('free', 'constrained', 'two-stage'):
                folder = root / mode
                folder.mkdir()
                draft = mode == 'two-stage'
                row = dict(id='q', subject='Math', question_type='multiple-choice', answer='B',
                           choices=self.choices, input_sha256='same', parsed_answer='B', correct=True,
                           finish_reason='stop', output_tokens=1025 if draft else 1,
                           batch_seconds=3 if draft else 1, reasoning_length_limited=draft,
                           any_stage_length_limited=draft, stages=[{}, {}] if draft else [{}])
                (folder / 'predictions.jsonl').write_text(json.dumps(row)+'\n', encoding='utf-8')
                config = dict(mode=mode, benchmark='MMMU validation MC', setting='standard', sampling_recipe={})
                write_json(folder/'manifest.json', dict(status='complete', selected_n=1,
                           evaluation_signature=dict(config=config, sha256=canonical_sha256(config))))
                write_json(folder/'summary.json', dict(summarize([row]), total_seconds=5))
            text = report(root)
            self.assertIn('two-stage', text)
            result = json.loads((root/'comparison.json').read_text())
            self.assertEqual(result['summaries']['two-stage']['reasoning_length_limited'], 1)
            self.assertEqual(result['summaries']['two-stage']['length_limited'], 0)
            folder = root/'constrained'
            row['input_sha256'] = 'different'
            (folder/'predictions.jsonl').write_text(json.dumps(row)+'\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Mismatched paired input'):
                report(root)
