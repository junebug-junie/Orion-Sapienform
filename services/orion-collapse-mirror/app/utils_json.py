from __future__ import annotations

import json
from typing import Any, Dict, Optional


REQUIRED_CM_KEYS = {
    "observer",
    "trigger",
    "observer_state",
    "field_resonance",
    "type",
    "emergent_entity",
    "summary",
    "mantra",
    "causal_echo",
    "timestamp",
    "environment",
}


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text or "{" not in text:
        return None

    # scan for balanced {...} blocks, try json.loads on each candidate
    starts = [i for i, ch in enumerate(text) if ch == "{"]

    for s in starts:
        depth = 0
        in_str = False
        esc = False

        for e in range(s, len(text)):
            ch = text[e]

            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue

            if ch == '"':
                in_str = True
                continue

            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[s : e + 1].strip()
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        pass

                    break

    return None


def _find_any_string(obj: Any):
    if obj is None:
        return
    if isinstance(obj, str):
        yield obj
        return
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _find_any_string(v)
        return
    if isinstance(obj, list):
        for v in obj:
            yield from _find_any_string(v)
        return


def find_collapse_entry(prior_step_results: Any) -> Optional[Dict[str, Any]]:
    """
    Accepts:
      - raw orion-exec:result:* dicts
      - StepExecutionResult dicts
      - any nested structure
    Returns a dict that looks like a Collapse Mirror JSON payload.
    """
    if prior_step_results is None:
        return None

    # If someone already passed a dict that is the entry itself
    if isinstance(prior_step_results, dict) and REQUIRED_CM_KEYS.issubset(prior_step_results.keys()):
        return prior_step_results

    # Walk all strings and try to pull a JSON object out
    for s in _find_any_string(prior_step_results):
        obj = _extract_first_json_object(s)
        if isinstance(obj, dict) and REQUIRED_CM_KEYS.issubset(obj.keys()):
            return obj

    # Also try: any dicts nested that look like entry-ish
    if isinstance(prior_step_results, dict):
        for _, v in prior_step_results.items():
            found = find_collapse_entry(v)
            if found:
                return found
    elif isinstance(prior_step_results, list):
        for v in prior_step_results:
            found = find_collapse_entry(v)
            if found:
                return found

    return None
