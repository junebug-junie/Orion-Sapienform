"""The reading turn's output contract: the draft must hold one JSON object.

Reading Stage 1/2 turns (``reading_only``) are machine-to-machine. The governor's
finalize validates and canonicalizes their draft here instead of running prose
repair (``preserve_structured_output``), and the FCC motor asks the model once
for the JSON object when its final answer fails this same check. One parser
(``orion.core.llm_json.parse_json_object``) on both sides, so the motor never
resumes a draft finalize would have accepted, nor passes one it would reject.
"""

from __future__ import annotations

import json

from orion.core.llm_json import parse_json_object


def canonicalize_structured_output(text: str) -> str:
    """Return a compact JSON object or raise when a machine draft is not JSON."""
    return json.dumps(parse_json_object(text), ensure_ascii=False)


def is_structured_output(text: object) -> bool:
    """True when finalize's ``canonicalize_structured_output`` would accept ``text``."""
    if not isinstance(text, str) or not text.strip():
        return False
    try:
        parse_json_object(text)
    except ValueError:
        return False
    return True
