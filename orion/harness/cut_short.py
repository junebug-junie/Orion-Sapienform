"""Cut-short turns: build the draft from what the turn actually found.

When the FCC motor stops a still-working turn (time budget, stalled step,
oversized stream line, context ceiling) it hands back only the LAST assistant
text fragment. On a long investigation that fragment is almost always a
lead-in -- live 2026-10-01: "Let me check what `outcome_from_followup`
produces:" -- and finalize/repair then turned it into an "answer", or into
"I cannot complete this investigation... hit a context wall". Both are empty
shells (CLAUDE.md, "No empty-shell cognition").

This module keeps an ordered, bounded record of the turn's real findings --
the assistant's own interim text and every tool result, paired with the call
that produced it -- and, when a turn is cut short, builds a draft that says
plainly it was cut short and then lists those findings. No synthesis is
invented here: every line is something the turn itself saw or wrote.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# Motor error codes that mean "the motor stopped a turn that was still working".
# Pre-spawn refusals, MCP preflight failures, provider errors and non-zero exits
# are not cut-short turns -- nothing (or nothing trustworthy) ran.
CUT_SHORT_CODES = frozenset(
    {
        "fcc_context_ceiling_exceeded",
        # Pre-rename spelling; kept so a mixed-version deploy still routes here.
        "fcc_draft_length_ceiling_exceeded",
        "fcc_timeout",
        "fcc_stream_stalled",
        "fcc_stream_line_limit",
    }
)

CUT_SHORT_MARKER = "[Cut short - not a finished answer.]"

_REASON_TEXT = {
    "fcc_context_ceiling_exceeded": "its context filled past the lane's window",
    "fcc_draft_length_ceiling_exceeded": "its context filled past the lane's window",
    "fcc_timeout": "its time budget ran out",
    "fcc_stream_stalled": "one step stalled past the per-step limit",
    "fcc_stream_line_limit": "one step produced output too large to read",
}

_TEXT_CAP = 600
_TOOL_RESULT_CAP = 500
_TOOL_ARGS_CAP = 160
_LAST_TEXT_CAP = 4000
# Total budget for the findings section. Finalize/reflect/repair read this
# draft, so it must stay far below a lane window; the tail is kept because the
# latest findings are the ones the turn was acting on when it stopped.
DEFAULT_FINDINGS_CHAR_BUDGET = 8000


def is_cut_short_code(code: str | None) -> bool:
    return str(code or "").strip() in CUT_SHORT_CODES


def _raw(step: dict[str, Any]) -> dict[str, Any]:
    raw = step.get("raw") if isinstance(step.get("raw"), dict) else step
    return raw if isinstance(raw, dict) else {}


def _one_line(text: str, cap: int) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= cap else flat[: cap - 3] + "..."


def _result_text(body: Any) -> str:
    if isinstance(body, str):
        return body
    if isinstance(body, list):
        return "\n".join(
            str(b.get("text"))
            for b in body
            if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str)
        )
    return ""


@dataclass
class TurnFindings:
    """Ordered record of what a turn saw and said, fed one stream step at a time."""

    entries: list[str] = field(default_factory=list)
    tool_calls: dict[str, str] = field(default_factory=dict)
    tool_result_count: int = 0

    def observe(self, step: Any) -> None:
        if not isinstance(step, dict):
            return
        raw = _raw(step)
        rtype = str(raw.get("type") or "")
        if rtype not in ("assistant", "user"):
            return
        message = raw.get("message") if isinstance(raw.get("message"), dict) else {}
        content = message.get("content")
        if not isinstance(content, list):
            # A string-content user message is the prompt or the CLI's own
            # compaction summary -- neither is a finding of this turn.
            return
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text" and rtype == "assistant":
                text = str(block.get("text") or "").strip()
                if text:
                    self.entries.append(f"- Orion noted: {_one_line(text, _TEXT_CAP)}")
            elif btype == "tool_use":
                name = str(block.get("name") or "tool")
                args = block.get("input")
                arg_str = ""
                if isinstance(args, dict) and args:
                    arg_str = _one_line(json.dumps(args, default=str, ensure_ascii=False), _TOOL_ARGS_CAP)
                tool_id = str(block.get("id") or "")
                if tool_id:
                    self.tool_calls[tool_id] = f"{name} {arg_str}".strip()
            elif btype == "tool_result":
                body = _result_text(block.get("content")).strip()
                call = self.tool_calls.get(str(block.get("tool_use_id") or ""), "tool")
                err = " (error)" if block.get("is_error") else ""
                if not body:
                    body = "(empty result)"
                self.tool_result_count += 1
                self.entries.append(f"- {call}{err} returned: {_one_line(body, _TOOL_RESULT_CAP)}")

    def has_findings(self) -> bool:
        return bool(self.entries)


def build_cut_short_draft(
    *,
    error_code: str,
    step_count: int,
    findings: TurnFindings,
    last_text: str = "",
    char_budget: int = DEFAULT_FINDINGS_CHAR_BUDGET,
) -> str:
    """Plainly-marked draft from the turn's own findings; "" when it has none.

    Returning "" for a turn with no findings is deliberate: the caller then
    treats the motor as failed rather than shipping a marker with nothing
    behind it.
    """
    if not findings.has_findings():
        return ""
    reason = _REASON_TEXT.get(str(error_code or "").strip(), "the motor stopped it")
    kept: list[str] = []
    used = 0
    for entry in reversed(findings.entries):
        if kept and used + len(entry) + 1 > char_budget:
            break
        kept.append(entry)
        used += len(entry) + 1
    kept.reverse()
    omitted = len(findings.entries) - len(kept)
    lines = [
        CUT_SHORT_MARKER,
        (
            f"This turn was stopped after {step_count} steps because {reason} "
            f"({error_code}), before Orion wrote a conclusion. What follows is what the turn "
            f"actually recorded ({findings.tool_result_count} tool results), in order. "
            "None of it has been synthesized into an answer."
        ),
        "",
    ]
    if omitted:
        lines.append(f"({omitted} earlier entries omitted for length.)")
    lines.extend(kept)
    last = str(last_text or "").strip()
    if len(last) > _TEXT_CAP:
        # The findings list caps each note; a long in-progress write-up is the
        # closest thing to a conclusion the turn has, so keep it whole (bounded).
        lines += ["", "The last thing Orion was writing when it was stopped:", "", last[:_LAST_TEXT_CAP]]
    return "\n".join(lines)


def ensure_cut_short_marked(final_text: str | None) -> str | None:
    """Re-attach the marker if finalize/repair rewrote it away."""
    if not final_text or CUT_SHORT_MARKER in final_text:
        return final_text
    return f"{CUT_SHORT_MARKER}\n{final_text}"
