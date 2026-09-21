import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mc_parser import PARSER_POLICY, parse_pro
from scripts.rescore_mmmu_pro import reparse, rescore


class ParserRepairTests(unittest.TestCase):
    def setUp(self):
        self.choices = {chr(65+i): f'choice {i}' for i in range(12)}

    def test_final_formats_and_extended_choices(self):
        for raw, expected in [
            ('Discuss (A) then (B).\nAnswer: **C**','C'),
            ('So the final answer is (G).','G'),
            ('**Final Answer**\n\n(J)','J'),
            ('Answer: `K`','K'), ('Answer: (L)','L'),
            ('Answer: a','A'), ('Final Answer:\nAnswer: B','B'),
            ('Final answer: A\nAnswer: C','C')]:
            with self.subTest(raw=raw):
                self.assertEqual(parse_pro(raw,self.choices)[0],expected)

    def test_word_prefix_and_refusal_are_not_letters(self):
        for raw in ['Final answer is invalid.', 'Answer: Approximately 25',
                    '(A) might work. Answer: Cannot be determined',
                    'Answer: I cannot determine the answer.',
                    'Answer: A\nAnswer: None of the above']:
            with self.subTest(raw=raw):
                self.assertIsNone(parse_pro(raw,self.choices)[0])
        # A refusal-like phrase can be a real option; match it only with evidence.
        choices = dict(self.choices, B='None of the above')
        self.assertEqual(parse_pro('Answer: None of the above',choices)[0],'B')

    def test_truncation_never_promotes_speculative_candidate(self):
        self.assertIsNone(parse_pro('Perhaps the answer is (G). Let me try again', self.choices,'length')[0])
        self.assertEqual(parse_pro('Final answer: **G**',self.choices,'length')[0],'G')
        self.assertIsNone(parse_pro('Answer: In this case we should recalculate',self.choices,'length')[0])

    def test_legacy_vision_preserves_text_fallback(self):
        row = dict(raw_response='The result corresponds to some option text.',option_count=4,
                   parsed_answer='B',parsing={'mode':'option_text','candidates':['B']},finish_reason='stop')
        answer,info,strategy = reparse(row,None)
        self.assertEqual((answer,info),('B',row['parsing']))
        self.assertEqual(strategy,'legacy_vision_fallback_preserved')

    def test_rescore_preserves_generation_and_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)/'source'
            dest = Path(temp)/'rescored'
            source.mkdir()
            row = dict(id='synthetic',subject='Math',setting='standard-10',question_type='multiple-choice',
                       option_count=12,raw_response='So the final answer is (C).',answer='C',
                       parsed_answer='I',parsing={'mode':'explicit_final','candidates':['i']},correct=False,
                       finish_reason='stop',stop_reason=None,input_tokens=42,output_tokens=12,batch_seconds=1)
            audit = {'id':row['id'],'choices':self.choices}
            profile = {'profile_name':'test','parser_policy':'old'}
            config = {'parser_policy':'old','profile_sha256':'old'}
            manifest = {'status':'development_subset_complete',
                        'arguments':{'setting':'standard-10','model_path':'unchanged',
                                     'min_pixels':100,'max_pixels':200,'max_tokens':8192,'max_model_len':16384,'batch_size':1},
                        'evaluation_signature':{'config':config},'recipe':{'seed':3407}}
            summary = {'setting':'standard-10','n':1,'correct':0,'accuracy':0,'complete_1730':False,
                       'subjects':[{'subject':'Math','seconds':1}], 'option_count_distribution':{'12':1},
                       'total_seconds':100,'gpu':{'unchanged':True}}
            for name,value in [('manifest.json',manifest),('summary.json',summary),('evaluation_profile.json',profile)]:
                (source/name).write_text(json.dumps(value),encoding='utf-8')
            (source/'predictions.jsonl').write_text(json.dumps(row)+'\n',encoding='utf-8')
            (source/'inputs.jsonl').write_text(json.dumps(audit)+'\n',encoding='utf-8')
            before = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
            result = rescore(source,dest)
            self.assertEqual(result['wrong_to_correct'],1)
            after = json.loads((dest/'predictions.jsonl').read_text())
            for key in ('raw_response','input_tokens','output_tokens','finish_reason','batch_seconds'):
                self.assertEqual(after[key],row[key])
            new_manifest = json.loads((dest/'manifest.json').read_text())
            self.assertEqual(new_manifest['evaluation_signature']['config']['parser_policy'],PARSER_POLICY)
            self.assertEqual(json.loads((dest/'summary.json').read_text())['total_seconds'],100)
            self.assertEqual(before,{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()})
            with self.assertRaises(FileExistsError):
                rescore(source,dest)


if __name__ == '__main__':
    unittest.main()
