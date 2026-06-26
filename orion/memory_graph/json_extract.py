"""Deterministic extraction of the first top-level JSON object from model text.

Used by Hub memory-graph suggest parsing. No LLM or network I/O.
"""


def extract_first_json_object_text(text: str) -> str | None:
    """Return the substring of the first balanced `{ ... }` object, or None."""
    candidate = (text or "").strip()
    if not candidate:
        return None
    start = candidate.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(candidate)):
        ch = candidate[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return candidate[start : idx + 1]
    return None


def salvage_truncated_json_object_text(text: str) -> str | None:
    """Best-effort repair when finish_reason=length left an unclosed JSON object."""
    import json

    balanced = extract_first_json_object_text(text)
    if balanced:
        return balanced

    raw = (text or "").strip()
    start = raw.find("{")
    if start < 0:
        return None
    fragment = raw[start:]

    for trim in range(0, min(400, len(fragment))):
        base = fragment[: len(fragment) - trim].rstrip().rstrip(",") if trim else fragment.rstrip().rstrip(",")
        if not base or not base.startswith("{"):
            continue
        for suffix in ("", "}", "]}", "\"}", "null}", "null]}", "false}", "true}"):
            for close in ("", "}", "]}", "}]}", "}]}]", "}]}]}"):
                candidate = f"{base}{suffix}{close}"
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    return candidate
    return None
