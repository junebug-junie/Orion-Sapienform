"""Tries that never reached the model (2026-11-03, moved from orion/prime/state.py on 2026-11-04).

A try that got no pool seat, no gateway reply, or an upstream 5xx (after one immediate retry) is not a
thought. ``note_no_model`` records it without starting the refractory interval, and the next tick re-bids
(that is the retry). ``NO_MODEL_FREE_TRIES`` in a row are free; after that each one backs off (refractory x
2^n, capped at ``NO_MODEL_BACKOFF_CAP``), and a gateway shed answer's own ``retry_after_sec`` is always
honoured. A no-seat try also sends the next one on the ``starved`` pool priority (``wants_starved_route``).

The functions work on any loop state model with the four try fields (``orion/mindloop/runtime.py``
``LOOP_STATE_FIELDS``, checked when an instance is built): ``no_model_streak``, ``no_model_tries_total``, ``last_try_kind``, ``retry_not_before``.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal, Optional, TypeVar

from pydantic import BaseModel

# Tries that never reached the model: this many in a row re-bid on the next tick, then back off.
NO_MODEL_FREE_TRIES = 3
NO_MODEL_BACKOFF_CAP = timedelta(hours=1)

TryKind = Literal["model", "no_seat", "no_reply", "upstream_5xx"]
# The gateway's typed errors (services/orion-llm-gateway app/pool_placement.py, llm_backend.py).
GATEWAY_NO_SEAT_ERRORS = frozenset({"gpu_pool_unavailable", "gpu_pool_recalled"})
GATEWAY_UPSTREAM_5XX = "upstream_http_5xx"

S = TypeVar("S", bound=BaseModel)


def classify_try(err: Optional[str], gateway_error: Optional[dict]) -> TryKind:
    """Did this model call reach the model? ``gateway_error`` is the gateway's typed error on an empty
    reply (orion-mind app/llm_client.py ``meta["gateway_error"]``).

    * ``no_seat``: the pool never granted a seat (queue deadline, heat shed, nothing serving the
      class) or took it back before any answer;
    * ``upstream_5xx``: the model server failed the request;
    * ``no_reply``: no gateway answer at all (the bus RPC timed out; the RPC covers seat wait +
      generation + a margin, so this is the gateway or the bus, not the model thinking long);
    * ``model``: everything else -- an answer, an empty or unparseable one, or generation that ran out
      of time. Those are the model's (and so the instance's) own outcome: a failed thought."""
    code = str((gateway_error or {}).get("error") or "")
    if code in GATEWAY_NO_SEAT_ERRORS:
        return "no_seat"
    if code == GATEWAY_UPSTREAM_5XX:
        return "upstream_5xx"
    if not code and err and ("RPC timeout" in err or err.startswith("TimeoutError")):
        return "no_reply"
    return "model"


def no_model_backoff(streak: int, *, base: timedelta) -> timedelta:
    """How long to hold the next try after ``streak`` no-model tries in a row: 0 for the first
    NO_MODEL_FREE_TRIES, then base x 2^n, capped at NO_MODEL_BACKOFF_CAP."""
    if streak <= NO_MODEL_FREE_TRIES:
        return timedelta(0)
    exp = min(streak - NO_MODEL_FREE_TRIES - 1, 16)
    return min(base * (2 ** exp), NO_MODEL_BACKOFF_CAP)


def note_no_model(state: S, kind: TryKind, now: datetime, *, base: timedelta,
                  retry_after_sec: Optional[float] = None) -> S:
    """A try that never reached the model: not a thought. The refractory clock and the pending changes
    are untouched; the streak grows and may hold the next try (see ``no_model_backoff``)."""
    streak = state.no_model_streak + 1
    hold = no_model_backoff(streak, base=base)
    if isinstance(retry_after_sec, (int, float)) and retry_after_sec > 0:
        hold = max(hold, timedelta(seconds=float(retry_after_sec)))
    return state.model_copy(update={
        "no_model_streak": streak, "no_model_tries_total": state.no_model_tries_total + 1,
        "last_try_kind": kind, "retry_not_before": (now + hold) if hold > timedelta(0) else None,
    })


def note_model_reached(state: S) -> S:
    return state.model_copy(update={"no_model_streak": 0, "last_try_kind": "model", "retry_not_before": None})


def wants_starved_route(state: BaseModel) -> bool:
    """The next try goes on the ``starved`` pool priority when the last one got no seat at all.
    Only no_seat: a dead gateway (no_reply) or a failing model server (5xx) is not helped by a place
    further up the queue."""
    return state.last_try_kind == "no_seat"


__all__ = [
    "GATEWAY_NO_SEAT_ERRORS",
    "GATEWAY_UPSTREAM_5XX",
    "NO_MODEL_BACKOFF_CAP",
    "NO_MODEL_FREE_TRIES",
    "TryKind",
    "classify_try",
    "no_model_backoff",
    "note_model_reached",
    "note_no_model",
    "wants_starved_route",
]
