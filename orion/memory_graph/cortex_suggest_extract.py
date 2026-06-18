"""Extract memory_graph_suggest draft JSON from CortexClientResult-shaped payloads."""

from __future__ import annotations

import json
from typing import Any

from orion.memory_graph.json_extract import extract_first_json_object_text
from orion.memory_graph.suggest_validate import parse_json_object


def _openai_choice_message_text(raw: Any) -> list[str]:
    out: list[str] = []
    if not isinstance(raw, dict):
        return out
    choices = raw.get("choices")
    if not isinstance(choices, list):
        return out
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        msg = choice.get("message")
        if not isinstance(msg, dict):
            continue
        for field in ("content", "reasoning_content", "reasoning", "reasoning_text"):
            val = msg.get(field)
            if isinstance(val, str) and val.strip():
                out.append(val.strip())
    return out


def _service_block_text_candidates(block: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for field in (
        "content",
        "final_text",
        "text",
        "reasoning_content",
        "inline_think_content",
    ):
        val = block.get(field)
        if isinstance(val, str) and val.strip():
            out.append(val.strip())
    raw = block.get("raw")
    if isinstance(raw, dict):
        out.extend(_openai_choice_message_text(raw))
    return out


def _step_text_candidates(step: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for container_key in ("result", "detail"):
        container = step.get(container_key)
        if not isinstance(container, dict):
            continue
        for block in container.values():
            if isinstance(block, dict):
                out.extend(_service_block_text_candidates(block))
            elif isinstance(block, str) and block.strip():
                out.append(block.strip())
        output = container.get("output") if isinstance(container.get("output"), dict) else None
        if output is not None:
            out.extend(_service_block_text_candidates(output))
        for field in ("text", "content"):
            val = container.get(field)
            if isinstance(val, str) and val.strip():
                out.append(val.strip())
    return out


def _sorted_steps(steps: list[Any]) -> list[dict[str, Any]]:
    typed: list[dict[str, Any]] = [s for s in steps if isinstance(s, dict)]

    def _rank(step: dict[str, Any]) -> tuple[int, int]:
        name = str(step.get("step_name") or "")
        prefer = 0 if name == "llm_memory_graph_suggest" else 1
        order = step.get("order")
        return prefer, int(order) if isinstance(order, int) else 0

    return sorted(typed, key=_rank)


def extract_suggest_text_from_cortex_payload(raw: dict[str, Any]) -> str:
    """Return best-effort model text containing a suggest draft from a cortex payload."""
    if not isinstance(raw, dict):
        return ""

    if isinstance(raw.get("draft"), dict):
        return json.dumps(raw["draft"])

    for field in ("final_text", "text", "content"):
        val = raw.get(field)
        if isinstance(val, str) and val.strip():
            json_blob = extract_first_json_object_text(val)
            return json_blob if json_blob else val.strip()

    steps = _sorted_steps(list(raw.get("steps") or raw.get("step_results") or []))
    best = ""
    best_score = -1
    for step in reversed(steps):
        for candidate in _step_text_candidates(step):
            json_blob = extract_first_json_object_text(candidate)
            pick = json_blob if json_blob else candidate
            score = len(pick) + (1_000_000 if json_blob else 0)
            if score > best_score:
                best = pick
                best_score = score
    if best:
        return best

    nested = raw.get("result")
    if isinstance(nested, dict):
        return extract_suggest_text_from_cortex_payload(nested)

    meta = raw.get("metadata")
    if isinstance(meta, dict):
        preview = meta.get("structured_rejection_preview")
        if isinstance(preview, str) and preview.strip():
            json_blob = extract_first_json_object_text(preview)
            if json_blob:
                return json_blob
            return preview.strip()

    return ""


def extract_suggest_draft_dict_from_cortex_payload(raw: dict[str, Any]) -> dict[str, Any]:
    """Parse a SuggestDraftV1 dict from a CortexClientResult-shaped payload."""
    if isinstance(raw.get("draft"), dict):
        return raw["draft"]

    text = extract_suggest_text_from_cortex_payload(raw)
    data, parse_err = parse_json_object(text)
    if data is not None:
        return data
    raise ValueError(parse_err or "memory_graph_suggest_draft_not_found")
