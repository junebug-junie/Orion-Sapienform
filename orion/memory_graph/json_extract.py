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
