from __future__ import annotations

import json
from typing import Any


def extract_final_text(result_payload: dict[str, Any]) -> str:
    """Pull the plan's final_text out of a decoded CortexExecResultPayload.

    Mirrors services/orion-actions/app/main.py's _extract_plan_final_text --
    duplicated rather than imported since that function is private to
    orion-actions and not a shared library export.
    """
    result = result_payload.get("result") if isinstance(result_payload, dict) else None
    if isinstance(result, dict):
        final = result.get("final_text")
        if isinstance(final, str) and final.strip():
            return final.strip()
        steps = result.get("steps")
        if isinstance(steps, list):
            for step in reversed(steps):
                step_result = step.get("result") if isinstance(step, dict) else None
                if not isinstance(step_result, dict):
                    continue
                for payload in step_result.values():
                    if not isinstance(payload, dict):
                        continue
                    text = payload.get("text") or payload.get("content")
                    if isinstance(text, str) and text.strip():
                        return text.strip()
    return ""


def parse_structured_observation(final_text: str) -> dict[str, Any]:
    """Parse a substrate.* verb's structured JSON output.

    Expected shape: {"observation": str, "salient_facts": list[str], "confidence": float}.
    Never raises -- an unparseable or non-conforming payload degrades to an
    empty observation, which the caller treats as status="empty" (never
    fabricated as success). This is the empty-shell-cognition rule applied
    at the dispatch-result boundary.
    """
    if not final_text.strip():
        return {"observation": "", "salient_facts": [], "confidence": 0.0}
    try:
        data = json.loads(final_text)
    except (json.JSONDecodeError, ValueError):
        return {"observation": "", "salient_facts": [], "confidence": 0.0}
    if not isinstance(data, dict):
        return {"observation": "", "salient_facts": [], "confidence": 0.0}
    observation = data.get("observation")
    if not isinstance(observation, str):
        observation = ""
    salient_facts = data.get("salient_facts")
    if not isinstance(salient_facts, list):
        salient_facts = []
    confidence = data.get("confidence")
    if not isinstance(confidence, (int, float)):
        confidence = 0.0
    return {
        "observation": observation.strip(),
        "salient_facts": [str(f) for f in salient_facts],
        "confidence": float(confidence),
    }
