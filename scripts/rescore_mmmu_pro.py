#!/usr/bin/env python3
"""CPU-only parser repair of stored MMMU-Pro runs; preserves original generation."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_mmmu_pro as pro
from mc_parser import PARSER_POLICY, explicit_answer, parse_pro


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_rows(path):
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError(f'Duplicate IDs: {path}')
    return rows


def choices_for(row, audit):
    if row.get('choices') or audit.get('choices'):
        return row.get('choices') or audit['choices']
    if row['setting'] == 'vision':
        return None
    text = audit['messages_without_image_bytes'][0]['content'][-1]['text']
    if '\n\nChoices:\n' not in text or not text.endswith(pro.STANDARD_DIRECT_PROMPT):
        raise ValueError(f"Cannot reconstruct choices: {row['id']}")
    text = text.rsplit('\n\nChoices:\n', 1)[1]
    text = text[:-len(pro.STANDARD_DIRECT_PROMPT)].rstrip()
    matches = list(re.finditer(r'^\(([A-Z])\) ', text, re.M))
    if [m.group(1) for m in matches] != [chr(65+i) for i in range(row['option_count'])]:
        raise ValueError(f"Ambiguous choice layout: {row['id']}")
    return {m.group(1):text[m.end():matches[i+1].start() if i+1<len(matches) else len(text)].strip()
            for i,m in enumerate(matches)}


def reparse(row, choices):
    if choices is not None:
        return (*parse_pro(row['raw_response'], choices, row['finish_reason']), 'full_options')
    # Legacy vision logs omit option text. Preserve unchanged fallback decisions
    # rather than substituting standard-10 options or guessing their ordering.
    letters = {chr(65+i): '' for i in range(row['option_count'])}
    explicit = explicit_answer(row['raw_response'], letters, allow_answer_marker=True)
    if explicit is not None:
        answer, info = parse_pro(row['raw_response'], letters, row['finish_reason'])
        return answer, info, 'legacy_vision_explicit_repair'
    if row['parsing']['mode'] not in ('explicit_answer','explicit_final'):
        return row['parsed_answer'], row['parsing'], 'legacy_vision_fallback_preserved'
    answer, info = parse_pro(row['raw_response'], letters, row['finish_reason'])
    if answer is None and row['finish_reason'] != 'length':
        raise ValueError(f"{row['id']}: original options required to repair invalid explicit answer")
    return answer, info, 'legacy_vision_invalid_explicit_repair'


def rescore(source, destination):
    started = time.perf_counter()
    source = source.resolve()
    manifest = read_json(source/'manifest.json')
    summary = read_json(source/'summary.json')
    rows = read_rows(source/'predictions.jsonl')
    audit = {r['id']:r for r in read_rows(source/'inputs.jsonl')}
    if not rows or set(audit) != {r['id'] for r in rows} or len(rows) != summary['n']:
        raise ValueError(f'Incomplete or mismatched input/output: {source}')
    if manifest['status'] not in ('complete','development_subset_complete'):
        raise ValueError(f'Unfinished source run: {source}')
    if sum(r['correct'] for r in rows) != summary['correct']:
        raise ValueError(f'Source summary disagrees with predictions: {source}')
    for row in rows:
        if row['setting'] != manifest['arguments']['setting']:
            raise ValueError('Mixed settings')
    new_rows, changes = [], []
    strategies = Counter()
    for row in rows:
        choices = choices_for(row, audit[row['id']])
        answer, info, strategy = reparse(row, choices)
        strategies[strategy] += 1
        new = dict(row, parsed_answer=answer, parsing=info,
                   correct=answer is not None and answer == row['answer'])
        new_rows.append(new)
        if (answer, info, new['correct']) != (row['parsed_answer'], row['parsing'], row['correct']):
            changes.append({'id':row['id'], 'answer':row['answer'], 'finish_reason':row['finish_reason'],
                            'old_parsed':row['parsed_answer'], 'new_parsed':answer,
                            'old_correct':row['correct'], 'new_correct':new['correct'],
                            'old_parsing':row['parsing'], 'new_parsing':info,
                            'raw_response':row['raw_response']})
    # All validation/parsing completes before creating the destination.
    destination.mkdir(parents=True, exist_ok=False)
    for filename in ('inputs.jsonl','sampling_params.txt','chat_template.txt','environment.txt','requirements.freeze.txt'):
        if (source/filename).exists():
            shutil.copy2(source/filename,destination/filename)
    pro.write_json(destination/'generation_manifest.json',manifest)
    pro.write_json(destination/'generation_summary.json',summary)
    source_hash = hashlib.sha256((source/'predictions.jsonl').read_bytes()).hexdigest()
    provenance = {'source_directory':str(source),'source_predictions_sha256':source_hash,
                  'parser_policy':PARSER_POLICY,
                  'parser_sha256':hashlib.sha256((Path(pro.__file__).parent/'mc_parser.py').read_bytes()).hexdigest(),
                  'rescored_utc':datetime.now(timezone.utc).isoformat(),
                  'strategies':dict(strategies), 'gpu_inference_performed':False,
                  'wrong_to_correct':sum(not c['old_correct'] and c['new_correct'] for c in changes),
                  'correct_to_wrong':sum(c['old_correct'] and not c['new_correct'] for c in changes),
                  'answer_changes':sum(c['old_parsed'] != c['new_parsed'] for c in changes),
                  'metadata_or_answer_changes':len(changes)}
    # Replace the scorer signature only; generation arguments/times stay intact.
    config = dict(manifest['evaluation_signature']['config'])
    config['parser_policy'] = PARSER_POLICY
    if (source/'evaluation_profile.json').exists():
        profile = read_json(source/'evaluation_profile.json')
        profile['parser_policy'] = PARSER_POLICY
        digest = pro.canonical_sha256(profile)
        pro.write_json(destination/'evaluation_profile.json',profile)
        config['profile_sha256'] = digest
        manifest['evaluation_profile'] = {'name':profile['profile_name'], 'sha256':digest,
                                          'path':str((destination/'evaluation_profile.json').resolve())}
    manifest['parser_policy'] = PARSER_POLICY
    manifest['evaluation_signature'] = {'config':config,'sha256':pro.canonical_sha256(config)}
    manifest['rescoring'] = provenance
    subjects, domains, correct, accuracy = pro.aggregate(new_rows,{r['subject']:r['seconds'] for r in summary['subjects']})
    summary.update(subjects=subjects,domains=domains,correct=correct,accuracy=accuracy,
                   macro_subject_accuracy=sum(s['accuracy'] for s in subjects)/len(subjects),
                   unparsed=sum(r['parsed_answer'] is None for r in new_rows),
                   ambiguous_mc=sum(len(set(r['parsing']['candidates']))>1 for r in new_rows),
                   length_limited=sum(r['finish_reason']=='length' for r in new_rows))
    with (destination/'predictions.jsonl').open('w',encoding='utf-8') as out:
        for row in new_rows:
            out.write(json.dumps(row,ensure_ascii=False)+'\n')
    pro.write_json(destination/'changes.json',changes)
    pro.write_json(destination/'summary.json',summary)
    provenance['rescore_seconds'] = time.perf_counter()-started
    pro.write_json(destination/'rescore.json',provenance)
    pro.write_json(destination/'manifest.json',manifest)
    pro.make_report(destination,manifest,summary)
    return dict(setting=summary['setting'],n=summary['n'],accuracy_pct=accuracy*100,
                correct=correct,unparsed=summary['unparsed'],length_limited=summary['length_limited'],
                **{k:provenance[k] for k in ('wrong_to_correct','correct_to_wrong','answer_changes')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',nargs='+',type=Path)
    parser.add_argument('--output-root',type=Path,required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        parser.error('output-root must be new; source results are never overwritten')
    if len({p.name for p in args.runs}) != len(args.runs):
        parser.error('source directory basenames must be unique')
    results = []
    for source in args.runs:
        result = rescore(source,args.output_root/source.name)
        results.append(result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
    pro.write_json(args.output_root/'rescore_summary.json',results)
    keys = list(results[0])
    lines = ['# MMMU-Pro parser repair (no new inference)', '',
             '| Run | '+' | '.join(keys)+' |','|---|'+'---|'*len(keys)]
    for source,result in zip(args.runs,results):
        lines.append('| '+source.name+' | '+' | '.join(f'{result[k]:.4f}' if isinstance(result[k],float) else str(result[k]) for k in keys)+' |')
    (args.output_root/'rescore_summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
