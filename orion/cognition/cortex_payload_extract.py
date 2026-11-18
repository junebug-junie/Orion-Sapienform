"""Extract model text / JSON-bearing strings from Cortex PlanExecutionResult-shaped dicts."""

from __future__ import annotations

import re
from typing import Any

# Canonical home for this check (2026-08-19), promoted here from
# services/orion-hub/scripts/endogenous_outreach.py -- that module found the
# real incident this exists for (2026-08-14: a llamacpp 400 arrived as a
# perfectly non-empty final_text, sailed past an emptiness-only check, and
# was delivered/persisted into a real chat thread as if Orion had said it)
# and built the fix, but only for its own bare cortex_client.chat() path.
# `extract_cortex_payload_text()`/`extract_response_repair_text()`/
# `extract_finalize_reflection_payload()` in orion/harness/finalize.py --
# the SHARED code path every real unified turn's finalize chain runs
# through, not just outreach's -- had the identical "if text: return text"
# vulnerability with no detection at all. Confirmed live, 2026-08-19: a real
# circe-worker outage produced the exact string
# "[Error: llamacpp timed out after waiting]" as `orion_response_repair`'s
# own `final_text`, which the harness governor would have delivered as
# Orion's real spoken answer had outreach's OWN defense-in-depth backstop
# (a second copy of this exact check) not happened to also be in the call
# path that day. Moved here, the shared cortex-payload-extraction module
# both callers already depend on, so there is one definition instead of
# two drifting copies.
_ERROR_TEXT_PREFIXES = (
    "[error",
    "error:",
    "traceback (most recent call last)",
    "internal server error",
)
_ERROR_TEXT_MARKERS = (
    "llamacpp failed",
    "llamacpp timed out",
    "client error '4",
    "client error '5",
    "server error '5",
    "connection refused",
    "read timeout",
)


def looks_like_error_text(text: str) -> bool:
    """True when generated 'prose' is really a plumbing error report.

    Backstop only -- a caller's own ok/error result fields are the primary
    gate; this exists because an upstream can report failure purely in the
    text (see the module-level comment above for the two real incidents
    that made this necessary). Kept deliberately narrow: it matches error
    *framing*, not the mere presence of the word "error", so real prose
    reflecting on an error genuinely encountered elsewhere is not
    swallowed -- e.g. "the codebase is throwing errors I can't map yet" is
    not this.
    """
    stripped = str(text or "").strip().lower()
    if not stripped:
        return False
    if stripped.startswith(_ERROR_TEXT_PREFIXES):
        return True
    # Markers only count near the start; a long reflective passage that
    # happens to mention a timeout deep in the body is not an error report.
    head = stripped[:200]
    return any(marker in head for marker in _ERROR_TEXT_MARKERS)


# Fields that carry a reasoning model's hidden chain-of-thought, not its answer.
# Kept separate so an answer-only extractor can drop them (see
# extract_cortex_answer_text): live 2026-09-24..30, orion_response_repair on
# the thinking-on agent lane spent all 8000 tokens reasoning, returned
# content="", and extract_cortex_payload_text() fell back to the 26-36k-char
# reasoning_content -- which the harness shipped as Orion's reply (17 turns in
# 7 days, 32 all-time; harness_turn_trace final_text "We need answer user's
# request: repair Orion's draft reply to Juniper...").
_REASONING_MESSAGE_FIELDS = ("reasoning_content", "reasoning", "reasoning_text")
_REASONING_BLOCK_FIELDS = ("reasoning_content", "inline_think_content")


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)


def strip_inline_think(text: str) -> str:
    """Drop inline ``<think>`` reasoning from a text field (same rules as cortex-exec's
    router ``_strip_think_content``): whole blocks removed, a dangling ``</think>`` keeps only
    what follows it, an unclosed ``<think>`` keeps only what precedes it."""
    raw = str(text or "")
    if "<think>" not in raw.lower() and not _THINK_CLOSE_RE.search(raw):
        return raw.strip()
    cleaned = _THINK_BLOCK_RE.sub(" ", raw)
    if "<think>" not in cleaned.lower():
        close = _THINK_CLOSE_RE.search(cleaned)
        if close:
            cleaned = cleaned[close.end():]
    lowered = cleaned.lower()
    if "<think>" in lowered:
        cleaned = cleaned[: lowered.find("<think>")]
    return cleaned.strip()


def _answer_only(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        stripped = strip_inline_think(value)
        if stripped:
            out.append(stripped)
    return out


def _openai_choice_message_text(raw: Any, *, include_reasoning: bool = True) -> list[str]:
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
        fields = ("content", *_REASONING_MESSAGE_FIELDS) if include_reasoning else ("content",)
        for field in fields:
            val = msg.get(field)
            if isinstance(val, str) and val.strip():
                out.append(val.strip())
    return out


def _service_block_text_candidates(block: dict[str, Any], *, include_reasoning: bool = True) -> list[str]:
    out: list[str] = []
    for field in (
        "content",
        "final_text",
        "text",
        "reasoning_content",
        "inline_think_content",
        "structured",
        "json",
        "payload",
    ):
        if not include_reasoning and field in _REASONING_BLOCK_FIELDS:
            continue
        val = block.get(field)
        if isinstance(val, str) and val.strip():
            out.append(val.strip())
        elif isinstance(val, dict):
            out.append(str(val))
    raw = block.get("raw")
    if isinstance(raw, dict):
        out.extend(_openai_choice_message_text(raw, include_reasoning=include_reasoning))
    return out


def _step_text_candidates(step: dict[str, Any], *, include_reasoning: bool = True) -> list[str]:
    out: list[str] = []
    for container_key in ("result", "detail"):
        container = step.get(container_key)
        if not isinstance(container, dict):
            continue
        for block in container.values():
            if isinstance(block, dict):
                out.extend(_service_block_text_candidates(block, include_reasoning=include_reasoning))
            elif isinstance(block, str) and block.strip():
                out.append(block.strip())
        output = container.get("output") if isinstance(container.get("output"), dict) else None
        if output is not None:
            out.extend(_service_block_text_candidates(output, include_reasoning=include_reasoning))
        for field in ("text", "content", "final_text", "structured", "json", "payload"):
            val = container.get(field)
            if isinstance(val, str) and val.strip():
                out.append(val.strip())
    return out


def _sorted_steps(steps: list[Any]) -> list[dict[str, Any]]:
    typed: list[dict[str, Any]] = [s for s in steps if isinstance(s, dict)]

    def _rank(step: dict[str, Any]) -> tuple[int, int]:
        order = step.get("order")
        return 0, int(order) if isinstance(order, int) else 0

    return sorted(typed, key=_rank)


def extract_cortex_payload_text(raw: dict[str, Any], *, include_reasoning: bool = True) -> str:
    """Return best-effort model text from a cortex exec payload (may be JSON-ish prose).

    ``include_reasoning=True`` (default, existing callers) falls back to a
    reasoning model's ``reasoning_content`` / think blocks when the answer is
    empty -- JSON verbs rely on their own parser to reject that. A caller whose
    text is shown or stored AS Orion's words must use
    :func:`extract_cortex_answer_text` instead."""
    if not isinstance(raw, dict):
        return ""

    for field in ("final_text", "text", "content"):
        val = raw.get(field)
        if isinstance(val, str) and val.strip():
            if include_reasoning:
                return val.strip()
            answer = strip_inline_think(val)
            if answer:
                return answer

    steps = _sorted_steps(list(raw.get("steps") or raw.get("step_results") or []))
    for step in reversed(steps):
        candidates = _step_text_candidates(step, include_reasoning=include_reasoning)
        if not include_reasoning:
            # Inline <think> text in an answer field is reasoning too (llama.cpp with
            # reasoning_format=none, or a template that keeps the tags in content).
            candidates = _answer_only(candidates)
        if candidates:
            return candidates[-1]

    nested = raw.get("result")
    if isinstance(nested, dict):
        nested_text = extract_cortex_payload_text(nested, include_reasoning=include_reasoning)
        if nested_text:
            return nested_text

    meta = raw.get("metadata")
    if isinstance(meta, dict):
        preview = meta.get("structured_rejection_preview")
        if isinstance(preview, str) and preview.strip():
            return preview.strip() if include_reasoning else strip_inline_think(preview)

    return ""


def extract_cortex_answer_text(raw: dict[str, Any]) -> str:
    """The model's ANSWER only -- never its chain-of-thought.

    Same lookup as :func:`extract_cortex_payload_text` with every reasoning
    field excluded, so a reasoning model that spent its budget thinking and
    returned ``content=""`` yields ``""`` here (the caller's failure path),
    not its reasoning dressed up as prose."""
    return extract_cortex_payload_text(raw, include_reasoning=False)


def cortex_payload_truncated(raw: dict[str, Any]) -> bool:
    """True when any provider finish_reason in a cortex payload is ``length``
    (completion cut off at max_tokens) or runtime diagnostics flag truncation.
    Mirrors services/orion-durable-runs/app/runner.py ``_finish_reasons``."""
    if not isinstance(raw, dict):
        return False
    diagnostics = (raw.get("metadata") or {}).get("runtime_response_diagnostics") if isinstance(raw.get("metadata"), dict) else None
    if isinstance(diagnostics, dict) and diagnostics.get("truncation_detected") is True:
        return True
    for step in list(raw.get("steps") or raw.get("step_results") or []):
        if not isinstance(step, dict):
            continue
        for container_key in ("result", "detail"):
            container = step.get(container_key)
            if not isinstance(container, dict):
                continue
            for block in container.values():
                if not isinstance(block, dict):
                    continue
                if block.get("finish_reason") == "length":
                    return True
                block_raw = block.get("raw") if isinstance(block.get("raw"), dict) else {}
                for choice in block_raw.get("choices") or []:
                    if isinstance(choice, dict) and choice.get("finish_reason") == "length":
                        return True
    nested = raw.get("result")
    if isinstance(nested, dict):
        return cortex_payload_truncated(nested)
    return False


def cortex_exec_failure_detail(result: dict[str, Any]) -> str | None:
    """Summarize why a cortex exec payload has no usable model text."""
    if not isinstance(result, dict):
        return "cortex exec returned non-dict payload"

    status = str(result.get("status") or "").strip().lower()
    error = result.get("error")
    if isinstance(error, str) and error.strip():
        return error.strip()

    steps = list(result.get("steps") or [])
    step_errors: list[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        step_error = step.get("error")
        if isinstance(step_error, str) and step_error.strip():
            step_errors.append(step_error.strip())
    if step_errors:
        return step_errors[-1]

    meta = result.get("metadata")
    if isinstance(meta, dict) and meta.get("structured_output_rejected"):
        preview = str(meta.get("structured_rejection_preview") or "")[:240]
        return f"structured_output_rejected preview={preview!r}"

    if status in {"fail", "partial", "error"}:
        return f"cortex exec status={status or 'unknown'} with empty final_text"

    return None
