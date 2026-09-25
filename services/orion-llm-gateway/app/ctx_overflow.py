"""Detect a prompt that did not fit the context of the role that served it.

Detection only. The old escalation ladder (probe every route's /props, then POST straight to the
next-larger route's URL) is gone: it placed work on a GPU without asking orion-gpu-pool. Now
main.py releases the lease that overflowed and re-leases ONCE with a larger ``min_ctx_tokens``,
so the pool picks a role whose per-slot context can hold the prompt (spec 2026-09-24, stage 3).

Why a retry at all rather than trusting the pre-flight estimate: the estimate is chars/4, and a
chars-per-token guess is exactly the assumption that has been wrong here before. The upstream's
own 400 is ground truth.
"""
from __future__ import annotations

from typing import Any

# raw.error value for an overflowed call (llm_backend._context_overflow_result).
CONTEXT_OVERFLOW_ERROR = "context_overflow"

# llama.cpp's phrasing varies by build; match on the substrings that survive across them rather
# than an exact message. Deliberately narrow: a generic 400 must NOT be treated as an overflow,
# or a malformed request would be re-leased onto a bigger role for nothing.
_OVERFLOW_MARKERS = (
    "exceed",              # "the request exceeds the available context size"
    "context size",
    "context length",
    "n_ctx",
    "too many tokens",
    "prompt is too long",
)


def is_context_overflow(status_code: int, body: Any) -> bool:
    """True only for a prompt that did not fit. Narrow on purpose.

    A context overflow is DETERMINISTIC -- the same prompt on the same lane fails identically
    forever -- so it is the one error worth escalating rather than repeating. Every other 4xx is
    left alone: retrying a malformed request on a bigger model just wastes a bigger model.
    """
    if status_code not in (400, 413, 422, 500):
        return False
    text = ""
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            text = str(err.get("message") or err.get("type") or "")
        elif err is not None:
            text = str(err)
        if not text:
            text = str(body.get("message") or "")
    else:
        text = str(body or "")
    low = text.lower()
    return any(marker in low for marker in _OVERFLOW_MARKERS)
