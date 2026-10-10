"""Cursor failure classification for contractor-peer transport order."""

from __future__ import annotations

import re
from typing import Literal

CursorFailureKind = Literal["token_unavailable", "other"]

# Narrow markers — bare "token" matched unrelated prose ("tokenizing",
# "take a token turn") and falsely routed to Claude fallback.
_TOKEN_MARKERS = (
    "unauthorized",
    "401",
    "403",
    "quota",
    "rate limit",
    "ratelimit",
    "usage limit",
    "insufficient",
    "billing",
    "payment required",
    "api key",
    "apikey",
    "access token",
    "api_token",
    "authentication",
    "auth failed",
    "not authenticated",
    # Desktop Cursor Agent CLI login (host auth.json), not API-key billing.
    "not logged in",
    "login required",
    "please log in",
    "please login",
    "authentication required",
    "run `agent login`",
    "run agent login",
    "cursor agent binary not found",
)

# Regex forms that need a bit more structure than a substring.
_TOKEN_REGEXES = (
    re.compile(r"please run\b.*\blogin\b", re.IGNORECASE),
    re.compile(r"\blogin\b.*\brequired\b", re.IGNORECASE),
)


class TokenUnavailable(Exception):
    """Cursor tokens / auth / quota unavailable — Claude fallback is allowed once."""


def classify_cursor_failure(exc: BaseException) -> CursorFailureKind:
    """Classify a Cursor invoker failure as token/unavailable vs other."""
    if isinstance(exc, TokenUnavailable):
        return "token_unavailable"
    msg = str(exc or "").lower()
    if any(marker in msg for marker in _TOKEN_MARKERS):
        return "token_unavailable"
    if any(rx.search(msg) for rx in _TOKEN_REGEXES):
        return "token_unavailable"
    return "other"


# --- Plain-English cause for the brief Orion reads -------------------------
#
# 2026-10-10: every HelpRequest since ~09-28 was refused because Cursor hit its
# monthly usage limit, but the brief only said `claude_budget_unobserved` (the
# second link of the chain). Orion built a "regime break" theory around what
# was a billing cap. These helpers name the first link too.

CursorTokenCause = Literal["usage_limit", "auth", "binary_missing", "unknown"]

# Each group reuses markers already in `_TOKEN_MARKERS`; no new vocabulary.
_USAGE_LIMIT_MARKERS = (
    "usage limit",
    "quota",
    "rate limit",
    "ratelimit",
    "insufficient",
    "billing",
    "payment required",
)
_BINARY_MISSING_MARKERS = (
    "cursor agent binary not found",
    "curiosity_peer_agent_bin missing",
)

# Cursor CLI: "Your usage limits will reset when your monthly cycle ends on
# 10/14/2026." Only an explicit M/D/YYYY after "reset ... on" counts.
_RESET_DATE_RX = re.compile(
    r"\breset\b[^.\n]{0,120}?\bon\s+(\d{1,2})/(\d{1,2})/(\d{4})\b",
    re.IGNORECASE,
)


def parse_cursor_reset_date(message: str) -> str | None:
    """ISO date (YYYY-MM-DD) from a Cursor usage-limit message, else None."""
    m = _RESET_DATE_RX.search(message or "")
    if not m:
        return None
    month, day, year = (int(g) for g in m.groups())
    try:
        from datetime import date

        return date(year, month, day).isoformat()
    except ValueError:
        return None


def cursor_token_cause(exc: BaseException) -> CursorTokenCause:
    """Why Cursor was token-unavailable. Call only after classify == token_unavailable."""
    msg = str(exc or "").lower()
    if any(marker in msg for marker in _BINARY_MISSING_MARKERS):
        return "binary_missing"
    if any(marker in msg for marker in _USAGE_LIMIT_MARKERS):
        return "usage_limit"
    # Text match only: a bare TokenUnavailable("...") classifies as
    # token_unavailable by type, but its text may name no cause at all.
    if any(marker in msg for marker in _TOKEN_MARKERS) or any(
        rx.search(msg) for rx in _TOKEN_REGEXES
    ):
        return "auth"
    return "unknown"


def describe_cursor_unavailable(exc: BaseException) -> str:
    """One plain sentence naming why Cursor could not take the request."""
    cause = cursor_token_cause(exc)
    if cause == "usage_limit":
        reset = parse_cursor_reset_date(str(exc or ""))
        when = f"resets {reset}" if reset else "reset date unknown"
        return f"Cursor unavailable: it hit its usage limit ({when})."
    if cause == "binary_missing":
        return "Cursor unavailable: the Cursor agent program is not installed in this service."
    if cause == "auth":
        return "Cursor unavailable: it is not logged in or its credentials were rejected."
    return "Cursor unavailable: reason not recognised (see cursor said)."


# Plain sentences for `orion.autonomy.ask_claude_trigger.RefusalReason` budget
# values the Claude fallback gate can return.
_CLAUDE_REFUSAL_TEXT = {
    "budget_unobserved": (
        "this service cannot see Claude's usage meter, so it refuses rather "
        "than spend blind"
    ),
    "budget_limited": "Claude's usage meter reads limited",
    "budget_unknown": "Claude's usage meter state is unknown",
    "budget_observation_missing": "no Claude usage reading was available",
    "budget_observation_incoherent": "Claude's usage reading contradicted itself",
}


def describe_claude_refusal(refusal: str) -> str:
    text = _CLAUDE_REFUSAL_TEXT.get(refusal, f"budget gate returned {refusal}")
    return f"Claude fallback refused: {text}."


CURSOR_SAID_EXCERPT_CHARS = 400


def cursor_token_reason(exc: BaseException, *, after_text: str, after_code: str) -> str:
    """Full refusal chain Orion reads: plain sentences, then codes, then Cursor's own text.

    ``after_text`` / ``after_code`` describe what happened once Cursor was out
    (Claude refused / failed / unwired), e.g. ``describe_claude_refusal(r)`` and
    ``"claude_budget_unobserved"``.
    """
    cause = cursor_token_cause(exc)
    raw = " ".join(str(exc or "").split())
    excerpt = raw[:CURSOR_SAID_EXCERPT_CHARS]
    if len(raw) > CURSOR_SAID_EXCERPT_CHARS:
        excerpt += "..."
    return (
        f"{describe_cursor_unavailable(exc)} {after_text} "
        f"[cursor_token_unavailable:{cause}; {after_code}] "
        f"cursor said: {excerpt or '(empty)'}"
    )
