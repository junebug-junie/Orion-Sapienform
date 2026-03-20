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


def _json_to_dict(candidate: str) -> dict | None:
    try:
        parsed = json.loads(candidate)
    except Exception:
        return None

    if isinstance(parsed, dict):
        return parsed

    if isinstance(parsed, str):
        try:
            nested = json.loads(repair_json(parsed))
        except Exception:
            return None
        if isinstance(nested, dict):
            return nested

    return None


def _try_candidate(candidate: str) -> dict | None:
    if not candidate:
        return None

    parsed = _json_to_dict(candidate)
    if parsed is not None:
        return parsed

    repaired = repair_json(candidate)
    if repaired != candidate:
        parsed = _json_to_dict(repaired)
        if parsed is not None:
            return parsed

    stripped = _strip_outer_quotes(candidate)
    if stripped != candidate:
        parsed = _json_to_dict(stripped)
        if parsed is not None:
            return parsed
        stripped_repaired = repair_json(stripped)
        if stripped_repaired != stripped:
            parsed = _json_to_dict(stripped_repaired)
            if parsed is not None:
                return parsed

    needs_unescape = any(token in candidate for token in (r"\"", r"\n", r"\t", r"\r"))
    if needs_unescape:
        try:
            unescaped = bytes(candidate, "utf-8").decode("unicode_escape")
        except Exception:
            unescaped = ""
        if unescaped:
            parsed = _json_to_dict(unescaped)
            if parsed is not None:
                return parsed
            unescaped_repaired = repair_json(unescaped)
            if unescaped_repaired != unescaped:
                parsed = _json_to_dict(unescaped_repaired)
                if parsed is not None:
                    return parsed
            unescaped_stripped = _strip_outer_quotes(unescaped)
            if unescaped_stripped != unescaped:
                parsed = _json_to_dict(unescaped_stripped)
                if parsed is not None:
                    return parsed
                parsed = _json_to_dict(repair_json(unescaped_stripped))
                if parsed is not None:
                    return parsed

    return None


def parse_json_object(text: str) -> dict:
    original = text if isinstance(text, str) else str(text)

    for candidate in (
        original,
        repair_json(original),
        _strip_outer_quotes(original),
        repair_json(_strip_outer_quotes(original)),
    ):
        parsed = _try_candidate(candidate)
        if parsed is not None:
            return parsed

    preview = original[:_MAX_PREVIEW_CHARS]
    raise ValueError(f"Could not parse JSON object from LLM text: {preview!r}")
