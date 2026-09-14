"""Cursor failure classification for contractor-peer transport order."""

from __future__ import annotations

from typing import Literal

CursorFailureKind = Literal["token_unavailable", "other"]

_TOKEN_MARKERS = (
    "token",
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
    "authentication",
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
    return "other"
