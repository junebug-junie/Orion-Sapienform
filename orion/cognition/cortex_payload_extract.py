"""Extract model text / JSON-bearing strings from Cortex PlanExecutionResult-shaped dicts."""

from __future__ import annotations

from typing import Any

# Canonical home for this check (2026-08-19), promoted here from
# services/orion-hub/scripts/endogenous_outreach.py -- that module found the
# real incident this exists for (2026-08-14: a llamacpp 400 arrived as a
# perfectly non-empty final_text, sailed past an emptiness-only check, and
# was delivered/persisted into a real chat thread as if Orion had said it)
# and built the fix, but only for its own bare cortex_client.chat() path.
# `extract_cortex_payload_text()`/`extract_voice_finalize_text()`/
# `extract_finalize_reflection_payload()` in orion/harness/finalize.py --
# the SHARED code path every real unified turn's finalize chain runs
# through, not just outreach's -- had the identical "if text: return text"
# vulnerability with no detection at all. Confirmed live, 2026-08-19: a real
# circe-worker outage produced the exact string
# "[Error: llamacpp timed out after waiting]" as `orion_voice_finalize`'s
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
        "structured",
        "json",
        "payload",
    ):
        val = block.get(field)
        if isinstance(val, str) and val.strip():
            out.append(val.strip())
        elif isinstance(val, dict):
            out.append(str(val))
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


def extract_cortex_payload_text(raw: dict[str, Any]) -> str:
    """Return best-effort model text from a cortex exec payload (may be JSON-ish prose)."""
    if not isinstance(raw, dict):
        return ""

    for field in ("final_text", "text", "content"):
        val = raw.get(field)
        if isinstance(val, str) and val.strip():
            return val.strip()

    steps = _sorted_steps(list(raw.get("steps") or raw.get("step_results") or []))
    for step in reversed(steps):
        candidates = _step_text_candidates(step)
        if candidates:
            return candidates[-1]

    nested = raw.get("result")
    if isinstance(nested, dict):
        nested_text = extract_cortex_payload_text(nested)
        if nested_text:
            return nested_text

    meta = raw.get("metadata")
    if isinstance(meta, dict):
        preview = meta.get("structured_rejection_preview")
        if isinstance(preview, str) and preview.strip():
            return preview.strip()

    return ""


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
