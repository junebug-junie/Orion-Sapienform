#!/usr/bin/env python3
"""Leased, objectively scored code/planning/memory reasoning; no tools or writes by the model."""
import argparse
import json
from pathlib import Path

from deep_context import stream_chat, judge


def main():
    from transformers import AutoTokenizer
    from orion.evals.leased_llm import operator_lane_sync
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tokenizer', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--gateway', required=True)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--repeats', type=int, default=3)
    args = p.parse_args()
    if args.repeats < 1:
        p.error('repeats must be positive')
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    tasks = [json.loads(x) for x in Path(__file__).with_name('deep_tasks.jsonl').read_text().splitlines()]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    passed = True
    with args.out.open('x') as out:
        try:
            with operator_lane_sync('agent-deep', holder='operator:hecate-deep-quality',
                                    gateway=args.gateway, queue_wait_sec=900) as lane:
                for repeat in range(args.repeats):
                    for task in tasks:
                        messages = [{'role': 'user', 'content': task['prompt'] + '\nReturn only the requested JSON.'}]
                        count = len(tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                            enable_thinking=True, return_dict=False))
                        result = stream_chat(lane, {'messages': messages, 'temperature': 0,
                            'max_tokens': 32768, 'seed': 20261114 + repeat,
                            'chat_template_kwargs': {'enable_thinking': True}}, 1800)
                        result.update(task_id=task['id'], kind=task['kind'], repeat=repeat,
                                      expected=task['expected'], seat=lane.summary())
                        result['failures'] = judge(result, expected=task['expected'], model=args.model,
                                                   prompt_tokens=count, output_budget=32768)
                        passed &= not result['failures']
                        out.write(json.dumps(result) + '\n'); out.flush()
                        print(json.dumps({'task': task['id'], 'failures': result['failures'],
                                          'elapsed_s': result['elapsed_s']}), flush=True)
        except BaseException as exc:
            out.write(json.dumps({'status': 'INCOMPLETE', 'error': str(exc)}) + '\n'); out.flush()
            raise
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
