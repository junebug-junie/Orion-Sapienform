"""What Claude sees for each decision, output caps and redaction, and the footer under Claude's reply.

The words follow the design's "What Claude sees" table (spec section 3). Every refusal says plainly
that nothing was shared, so Claude cannot read a refusal as an empty result.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Iterable, Mapping, Optional

from orion.core.redact import REDACTED, redact_secrets

OUTPUT_CAP = 4000
_CUT_NOTE = "\n(cut at 4000 characters)"

# Token shapes a tool result might carry (a turn step's Bash output, a config dump) that no env value
# names: Anthropic/OpenAI keys, GitHub tokens, Slack tokens, AWS key ids, JWTs, bearer headers and
# key=value secrets. Patterns are conservative on purpose: a false positive costs a word of Claude's
# view, a miss sends a credential to a third party.
_TOKEN_SHAPES = (
    re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{12,}"),
)
_SECRET_KV = re.compile(
    r"(?i)(?<![a-z0-9])(\w*(?:token|secret|api[_-]?key|access[_-]?key|private[_-]?key|credential)s?)"
    r"(\s*=\s*)(?:'[^']{8,}'|\"[^\"]{8,}\"|[^\s\"'),;}\]]{8,})"
)


def redact_tool_output(text: str, secret_values: Optional[Iterable[str]] = None) -> str:
    """orion.core.redact (DSN userinfo, *password=, credential-named env values) plus token shapes."""
    out = redact_secrets(text or "", secret_values)
    for pattern in _TOKEN_SHAPES:
        out = pattern.sub(REDACTED, out)
    return _SECRET_KV.sub(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}", out)


def cap_output(text: str, cap: int = OUTPUT_CAP) -> str:
    text = text or ""
    if len(text) <= cap:
        return text
    room = cap - len(_CUT_NOTE)
    head = text[:room]
    cut = head.rfind("\n")
    if cut > room // 2:
        head = head[:cut]
    return head + _CUT_NOTE


def _quote(text: Optional[str], n: int = 200) -> str:
    words = " ".join((text or "").split())
    return words if len(words) <= n else words[: n - 1] + "…"


def deny_text(category: str, set_at: Optional[datetime], reason: Optional[str]) -> str:
    when = set_at.date().isoformat() if set_at else "earlier"
    return f'Orion does not share {category} with you (set {when}: "{_quote(reason)}").'


def no_text(reason: Optional[str]) -> str:
    return f'Orion said no: "{_quote(reason)}".'


def not_shared_text(why: str) -> str:
    return f"Nothing was shared: {why}."


def yes_header(reason: Optional[str], scope: str) -> str:
    tail = " (for this conversation)" if scope == "conversation" else ""
    return f'Orion said yes{tail}: "{_quote(reason)}"'


def timeout_text(deadline_sec: float) -> str:
    return f"Orion did not answer within {deadline_sec:g} s, so nothing was shared."


def unavailable_text(error: str) -> str:
    return f"Not available: {_quote(error, 160).rstrip('.')}."


# --- footer under Claude's reply -------------------------------------------------------------------

def _tool_label(tool: str) -> str:
    return tool.replace("_", " ")


def footer_part(entry: Mapping[str, object]) -> str:
    """One call, as Juniper reads it under Claude's reply."""
    label = _tool_label(str(entry.get("tool") or "tool"))
    path = entry.get("decision_path")
    answer = entry.get("orion_answer")
    reason = _quote(str(entry.get("orion_reason") or ""), 80)
    if path == "standing_allow":
        return f"{label} (Orion allows it)"
    if path == "session_allow":
        return f"{label} (Orion: yes for this conversation)"
    if path == "standing_deny":
        return f"{label} (Orion: no, standing)"
    if path in ("asked", "repeat_no") and answer in ("yes", "no"):
        return f'{label} (Orion: {answer}, "{reason}")' if reason else f"{label} (Orion: {answer})"
    if path == "asked":
        return f"{label} (Orion did not answer)"
    return f"{label} (not available)"


def render_footer(entries: Iterable[Mapping[str, object]]) -> Optional[str]:
    parts = [footer_part(e) for e in entries]
    return ("Looked at: " + "; ".join(parts) + ".") if parts else None


__all__ = [
    "OUTPUT_CAP",
    "cap_output",
    "deny_text",
    "footer_part",
    "no_text",
    "not_shared_text",
    "redact_tool_output",
    "render_footer",
    "timeout_text",
    "unavailable_text",
    "yes_header",
]
