#!/usr/bin/env python3
"""Opt-in model acceptance smoke. Calls inference; never activates a pool role."""
import argparse
import json
import urllib.request


def chat(url, model, messages, **options):
    request = urllib.request.Request(
        url.rstrip('/') + '/v1/chat/completions',
        data=json.dumps(dict(model=model, messages=messages, stream=False, **options)).encode(),
        headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def evaluate(url, route, expected_model):
    """Small acceptance set; absence of a real answer always fails."""
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
            result = chat(url, route, [{'role':'user', 'content':prompt}],
                          temperature=0, max_tokens=2048, **options)
            text = result['choices'][0]['message'].get('content')
            passed = result.get('model') == expected_model and bool(text and text.strip()) and check(text)
            outcomes.append({'case':name, 'passed':bool(passed), 'model':result.get('model'),
                             'content':text, 'usage':result.get('usage')})
        except Exception as exc:
            outcomes.append({'case':name, 'passed':False, 'error':str(exc)})
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True, help='Direct Hecate server or Athena gateway base URL')
    parser.add_argument('--route', default='glm-5.3-flash', help='Use agent-deep when testing via gateway')
    parser.add_argument('--expected-model', default='glm-5.3-flash')
    args = parser.parse_args()
    results = evaluate(args.url, args.route, args.expected_model)
    print(json.dumps({'results':results, 'passed':all(r['passed'] for r in results)}, indent=2))
    raise SystemExit(0 if all(r['passed'] for r in results) else 1)


if __name__ == '__main__':
    main()
