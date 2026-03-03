from __future__ import annotations

import json
import re


_MAX_PREVIEW_CHARS = 2000


def repair_json(text: str) -> str:
    if text is None:
        return ""

    text = text.strip()

    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]

    text = text.replace(" None", " null").replace(":None", ":null")
    text = text.replace(" True", " true").replace(":True", ":true")
    text = text.replace(" False", " false").replace(":False", ":false")
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


def _strip_outer_quotes(value: str) -> str:
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {'"', "'"}:
        return stripped[1:-1]
    return value


def _try_json_loads(candidate: str) -> dict | None:
    try:
        parsed = json.loads(candidate)
    except Exception:
        return None

    if isinstance(parsed, dict):
        return parsed

    if isinstance(parsed, str):
        inner_candidate = repair_json(parsed)
        try:
            inner_parsed = json.loads(inner_candidate)
        except Exception:
            return None
        if isinstance(inner_parsed, dict):
            return inner_parsed

    return None


def parse_json_object(text: str) -> dict:
    original = text if isinstance(text, str) else str(text)
    candidate = repair_json(original)

    parsed = _try_json_loads(candidate)
    if parsed is not None:
        return parsed

    if "\\\"" in candidate or "\\n" in candidate or "\\t" in candidate:
        try:
            unescaped = bytes(candidate, "utf-8").decode("unicode_escape")
        except Exception:
            unescaped = ""
        if unescaped:
            unescaped = repair_json(unescaped)
            parsed = _try_json_loads(unescaped)
            if parsed is not None:
                return parsed

    stripped_quotes = _strip_outer_quotes(candidate)
    if stripped_quotes != candidate:
        parsed = _try_json_loads(repair_json(stripped_quotes))
        if parsed is not None:
            return parsed

        if "\\\"" in stripped_quotes or "\\n" in stripped_quotes or "\\t" in stripped_quotes:
            try:
                unescaped = bytes(stripped_quotes, "utf-8").decode("unicode_escape")
            except Exception:
                unescaped = ""
            if unescaped:
                parsed = _try_json_loads(repair_json(unescaped))
                if parsed is not None:
                    return parsed

    preview = original[:_MAX_PREVIEW_CHARS]
    raise ValueError(f"Could not parse JSON object from LLM text: {preview!r}")
