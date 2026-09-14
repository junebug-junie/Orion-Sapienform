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
