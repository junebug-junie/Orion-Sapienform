"""Degraded RPC replies: a reply that arrived but is not the service's real answer.

Why this exists (2026-10-13, live): during the 2026-10-08 cabinet-heat GPU shed every
``journal.compose`` call on ``cortex-orch -> orion:cortex:exec:request:background``
(263 of 263, 15:00-19:00 UTC, ``cognition_traces``) failed with
``gpu_pool_unavailable``: orion-llm-gateway refused in well under a second and
cortex-exec replied with a failed plan just as fast. ``rpc_request()`` recorded
every one of those replies as a *success*, so the per-hop transport baseline
(``orion/metacog/transport_baseline.py``), whose floor follows improvements with a
90 s half-life, learned "this hop takes ~1.3 s" from refusals and then read real
~8 s calls as 6x saturated for days. A refusal is not a round trip.

The marker -- one reason string, no taxonomy:

- **Explicit, set by the service that degraded.** The reply envelope's
  ``trace[DEGRADED_TRACE_KEY]`` holds a short reason (``"gpu_pool_unavailable"``,
  ``"metacog_draft_fallback:empty_response"``). ``BaseEnvelope.trace`` is an open
  dict, so no envelope schema change is needed and an old caller simply ignores it.
- **The existing failure kind.** A reply whose ``kind`` is in ``SYSTEM_ERROR_KINDS``
  is already, by repo-wide convention, "this reply is a failure"
  (``bus_schemas.py``); it reads as degraded with reason ``"system_error"``.

Nothing is inferred from payload content by the caller.

What counts as degraded -- one rule: **the replier declined or skipped the work**
(a GPU-pool / lease / capacity refusal, an unusable request, a canned fallback produced
without asking the model). A reply that *failed after real work* -- a backend timeout,
an upstream 5xx, a model that ran and returned junk -- is NOT degraded: its wait was
real, and it is exactly the slowness a latency baseline must see.

How a service marks: a Rabbit (``bus_service_chassis.Rabbit``) handler opens a
collector around every request; anything that runs inside the handler calls
``note_degraded(reason)`` and the chassis stamps the reply. A service that
publishes its reply itself (outside the chassis) stamps with ``mark_degraded``.
Relays are explicit: a service whose answer IS a downstream reply (cortex-orch
relaying cortex-exec) calls ``note_degraded`` itself when that reply was degraded --
``rpc_request()`` does not propagate on its own, because a degraded side lookup
(a state fetch, an optional recall) does not make the caller's own answer fake.
"""

from __future__ import annotations

import contextvars
from typing import Any, Optional

from orion.core.bus.bus_schemas import SYSTEM_ERROR_KINDS

DEGRADED_TRACE_KEY = "rpc_degraded"
# Reasons are labels for logs/tests, not payloads: bounded so a caller-supplied
# string cannot bloat every reply envelope.
MAX_REASON_CHARS = 120

_COLLECTOR: contextvars.ContextVar[Optional[list[str]]] = contextvars.ContextVar(
    "orion_rpc_degraded_collector", default=None
)


def _clean(reason: Any) -> Optional[str]:
    if not isinstance(reason, str):
        return None
    text = " ".join(reason.split())[:MAX_REASON_CHARS]
    return text or None


def degraded_reason(envelope: Any, *, request_trace: Optional[dict] = None) -> Optional[str]:
    """Why ``envelope`` (an RPC reply) is degraded, or None if it is a real answer.

    ``request_trace``: the trace the caller sent. A replier that copies the request
    trace into its reply (cortex-exec's ``_derive_envelope`` does) would echo a
    marker the request already carried; an echoed marker identical to the
    request's is not this reply's own verdict and is ignored. Never raises.
    """
    try:
        trace = getattr(envelope, "trace", None)
        if isinstance(trace, dict):
            reason = _clean(trace.get(DEGRADED_TRACE_KEY))
            if reason is not None:
                inherited = (
                    _clean(request_trace.get(DEGRADED_TRACE_KEY)) if isinstance(request_trace, dict) else None
                )
                if reason != inherited:
                    return reason
        if getattr(envelope, "kind", None) in SYSTEM_ERROR_KINDS:
            return "system_error"
    except Exception:
        return None
    return None


def mark_degraded(envelope: Any, reason: str) -> Any:
    """Return ``envelope`` with ``trace[DEGRADED_TRACE_KEY] = reason`` (envelopes are
    frozen, so this is a copy). An existing reason is kept: the first cause wins.
    Returns ``envelope`` unchanged on a blank reason or a non-envelope."""
    clean = _clean(reason)
    trace = getattr(envelope, "trace", None)
    if clean is None or not isinstance(trace, dict) or not hasattr(envelope, "model_copy"):
        return envelope
    if _clean(trace.get(DEGRADED_TRACE_KEY)) is not None:
        return envelope
    return envelope.model_copy(update={"trace": {**trace, DEGRADED_TRACE_KEY: clean}})


def strip_degraded(trace: Any) -> dict:
    """A copy of ``trace`` without the marker -- for code that copies an incoming
    request's trace into a new envelope."""
    if not isinstance(trace, dict):
        return {}
    return {k: v for k, v in trace.items() if k != DEGRADED_TRACE_KEY}


def note_degraded(reason: str) -> None:
    """Record that the reply being built for the current request is degraded. A no-op
    outside a collector (a Hunter handler, a background task with no reply)."""
    bucket = _COLLECTOR.get()
    clean = _clean(reason)
    if bucket is not None and clean is not None and clean not in bucket:
        bucket.append(clean)


def current_degraded_reason() -> Optional[str]:
    """The first reason noted for the current request, or None."""
    bucket = _COLLECTOR.get()
    return bucket[0] if bucket else None


def begin_collecting() -> contextvars.Token:
    """Open a fresh collector for one request; pair with ``end_collecting``."""
    return _COLLECTOR.set([])


def end_collecting(token: contextvars.Token) -> None:
    _COLLECTOR.reset(token)
