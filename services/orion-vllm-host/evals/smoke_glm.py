#!/usr/bin/env python3
"""Opt-in model acceptance smoke. Calls inference; never activates a pool role.

It takes a GPU pool operator hold on --lane (default agent-deep, hecate) and calls only through
orion-llm-gateway (orion/evals/leased_llm.py: class operator_<lane>, background, behind all of
Orion's own work). Needs ORION_BUS_URL. hecate's card waits until Juniper lends it. A recall stops
the smoke with exit 3 and no result."""
import argparse
import json


def chat(lane, messages, **options):
    return lane.chat(dict(messages=messages, stream=False, **options), timeout_sec=300)


def evaluate(lane, expected_model):
    """Small acceptance set; absence of a real answer always fails. A recall (LaneRecalled) propagates."""
    from orion.evals.leased_llm import LaneCallError

    cases = [
        ('arithmetic', 'Return only the product of 17 and 23.', lambda t: t.strip() == '391', {}),
        ('recall', 'The test key is cobalt-otter-73. Ignore filler: ' + 'elm oak birch. ' * 200 +
         '\nReturn only the test key.', lambda t: t.strip() == 'cobalt-otter-73', {}),
        ('json', 'Return a JSON object with key ready and boolean value true.',
         lambda t: json.loads(t).get('ready') is True, {'response_format': {'type': 'json_object'}}),
    ]
    outcomes = []
    for name, prompt, check, options in cases:
        try:
            result = chat(lane, [{'role':'user', 'content':prompt}], temperature=0, max_tokens=2048, **options)
            text = result['choices'][0]['message'].get('content')
            passed = result.get('model') == expected_model and bool(text and text.strip()) and check(text)
            outcomes.append({'case':name, 'passed':bool(passed), 'model':result.get('model'),
                             'content':text, 'usage':result.get('usage')})
        except (LaneCallError, KeyError, ValueError, TypeError, AttributeError) as exc:
            outcomes.append({'case':name, 'passed':False, 'error':str(exc)})
    return outcomes


def main():
    from orion.evals.leased_llm import EXIT_RECALLED, LaneRecalled, operator_lane_sync

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lane', default='agent-deep', help='operator lane (agent-deep = hecate)')
    parser.add_argument('--expected-model', default='glm-5.3-flash')
    args = parser.parse_args()
    try:
        with operator_lane_sync(args.lane, holder='operator:vllm-smoke') as lane:
            results = evaluate(lane, args.expected_model)
            seat = lane.summary()
    except LaneRecalled as exc:
        print(json.dumps({'stopped': f'recalled: {exc}', 'passed': None}, indent=2))
        raise SystemExit(EXIT_RECALLED)
    print(json.dumps({'seat':seat, 'results':results, 'passed':all(r['passed'] for r in results)}, indent=2))
    raise SystemExit(0 if all(r['passed'] for r in results) else 1)


if __name__ == '__main__':
    main()
