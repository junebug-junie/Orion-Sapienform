"""Opt-in, non-persisting stance eval through the normal gateway admission path.

Supply a downstream task as UTF-8 text. This does not run the task or publish a
ThoughtEvent; it checks whether the prompt produces a grounded stance instead.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid

import httpx
from jinja2 import Environment

from orion.schemas.thought import ThoughtEventV1


def assess_response(raw: dict, correlation: str) -> dict:
    allowed = set(ThoughtEventV1.model_fields) - {'grounding_capsule', 'autonomy_slice'}
    allowed.update({'llm_profile', 'producer'})
    unexpected = sorted(set(raw) - allowed)
    # Only transport-owned metadata may be supplied by this evaluator. Missing
    # stance fields must fail, not pass through the runtime's tolerant coercers.
    thought = ThoughtEventV1.model_validate({
        **raw, 'event_id': correlation, 'correlation_id': correlation,
        'session_id': 'stance_boundary_eval', 'created_at': datetime.now(timezone.utc),
    })
    anchor = f'hub:turn:{correlation}'
    return {'unexpected_keys': unexpected, 'imperative': thought.imperative, 'valid_stance': (
        not unexpected and thought.evidence_refs == [anchor]
        and set(thought.strain_refs) <= {anchor}
        and bool(thought.imperative.strip()) and thought.disposition == 'proceed'
    )}


async def evaluate(args: argparse.Namespace) -> dict:
    task = args.task_file.read_text(encoding='utf-8')
    correlation = str(uuid.uuid4())
    anchor = f'hub:turn:{correlation}'
    prompt = Environment().from_string(args.template.read_text(encoding='utf-8')).render(
        user_message=task, stance_inputs={'user_message': task},
        association={'correlation_id': correlation, 'attended_node_ids': [anchor]},
        coalition_projection={'attended_node_ids': [anchor], 'open_loop_ids': [], 'broadcast_stale': False},
    )
    body = {
        'model': args.model, 'messages': [{'role': 'system', 'content': prompt}],
        'temperature': 0.7, 'max_tokens': 8000,
        'response_format': {'type': 'json_object'}, 'stream': True,
        'stream_options': {'include_usage': True},
    }
    report = {'correlation_id': correlation, 'prompt_chars': len(prompt),
              'reasoning_chars': 0, 'content_chars': 0, 'passed': False}
    content = []
    started = time.monotonic()

    async def collect():
        async with httpx.AsyncClient(timeout=args.timeout) as client:
            async with client.stream('POST', args.gateway_url.rstrip('/') + '/v1/chat/completions',
                                     json=body, headers={'X-Request-ID': correlation}) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith('data: ') or line == 'data: [DONE]':
                        continue
                    event = json.loads(line[6:])
                    if event.get('error'):
                        raise RuntimeError('gateway_stream_error')
                    if event.get('usage'):
                        report['usage'] = event['usage']
                    for choice in event.get('choices', []):
                        delta = choice.get('delta', {})
                        report['reasoning_chars'] += len(delta.get('reasoning_content') or delta.get('reasoning') or '')
                        text = delta.get('content') or ''
                        content.append(text)
                        report['content_chars'] += len(text)
                        if choice.get('finish_reason'):
                            report['finish_reason'] = choice['finish_reason']

    try:
        await asyncio.wait_for(collect(), timeout=args.timeout)
        raw = json.loads(''.join(content))
        report.update(assess_response(raw, correlation))
        report['passed'] = report.get('finish_reason') == 'stop' and report['valid_stance']
    except Exception as exc:
        report['error'] = type(exc).__name__
    report['elapsed_sec'] = round(time.monotonic() - started, 2)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-file', type=Path, required=True)
    parser.add_argument('--gateway-url', required=True)
    parser.add_argument('--model', default='agent')
    parser.add_argument('--timeout', type=float, default=235)
    parser.add_argument('--template', type=Path, default=Path(__file__).resolve().parents[2] / 'cognition/prompts/stance_react.j2')
    args = parser.parse_args()
    report = asyncio.run(evaluate(args))
    print(json.dumps(report), flush=True)
    return 0 if report['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
