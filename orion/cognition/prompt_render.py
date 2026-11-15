"""One Jinja environment for every verb prompt: dicts and lists print as JSON, never Python repr.

Jinja prints a dict as ``{'key': 'value'}``. Verb prompts that ask for strict JSON then showed the
model Python-literal dicts (stance_react: association, repair_bundle, coalition_projection) and the
model answered in kind: 11 stance_react Strata 502s on 2026-11-14 were all "Expecting property name
enclosed in double quotes", and their mixed quoting ('Juniper's') is past any safe repair
(services/orion-llm-gateway/app/strata_structured.py, issue #3289).

Production renders through cortex-exec (``services/orion-cortex-exec/app/executor.py``
``_render_prompt``); evals and replays that render the same templates use this too, so they test the
prompt production sends. ``finalize`` changes only ``{{ }}`` output: ``{% if %}``, filters, indexing
and ``~`` still see the Python object; ``| tojson`` output passes through. Not covered: ``| string``,
``| safe`` and sets (no template uses them), and repr built in Python before rendering.
"""
from __future__ import annotations

import json
from typing import Any

from jinja2 import Environment


def json_finalize(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError, RecursionError):
            return value
    return value


def prompt_environment(**kwargs: Any) -> Environment:
    """``Environment(autoescape=False, finalize=json_finalize)`` plus any other options."""
    kwargs.setdefault("autoescape", False)
    return Environment(finalize=json_finalize, **kwargs)
