"""Cross-process, cross-restart storage for "when did the user/Orion last
turn in this session" -- the raw inputs `situation.py::_build_conversation_phase`
buckets into `same_breath`/`short_pause`/.../`stale_thread` framing.

BUG THIS REPLACES (confirmed live 2026-08-21, do not reopen as an
investigation): `situation.py` used to hold this in two module-level,
in-process Python dicts (`_SESSION_LAST_USER_TURN`, `_SESSION_LAST_ORION_TURN`).
There are FOUR separate `orion-cortex-exec` containers
(`orion-athena-cortex-exec`, `-chat`, `-background`, `-spark`), each an
independent process with its own private copy of those dicts, and a single
chat turn's pipeline can touch more than one of them (the initial
stance/imperative RPC vs. later background-verb RPCs dispatched separately
over the bus). "How long since we last talked" was therefore neither durable
across restarts nor consistent across the service's own replicas. Confirmed
live: all four containers restarted simultaneously at 2026-08-21T03:51:2x (an
unrelated infra incident), wiping every copy of this state at once, and
reproduced the exact symptom Juniper reported -- a reply mid-active-
conversation carrying "haven't talked in a while" framing.

Storage-backend change only: the phase-bucketing thresholds themselves
(`_build_conversation_phase`'s own logic) are unchanged and untested here --
this module owns *where the timestamps live*, not how they're interpreted.

Precedent: mirrors `orion/harness/last_tool_fetch_cache.py` (single JSON
payload per key, one SETEX call, fail-open, "not found" vs "read/write
failed" logged distinctly) for the storage shape, and
`pre_turn_appraisal.py`/`current_turn_llm_signals.py` (module-level
`_BUS`, a `bind_*_bus()` setter called once from `main.py` at startup, a
`reset_*_bus_for_tests()` helper) for how this module gets a Redis handle --
`chat_stance.py`/`situation.py` never receive a `bus` parameter through their
own call chains (no `ctx["bus"]` is ever set in this service), so the bind
pattern already used by this service's own sibling modules is the real
precedent, not an ad hoc new one.

Key: ``orion:cortex-exec:session_turn_phase:{session_id}``
Payload: ``{"last_user_turn_at": "<ISO8601 UTC>" | null, "last_orion_turn_at": "<ISO8601 UTC>" | null}``
TTL: 7 days by default (`ttl_seconds` kwarg) -- comfortably longer than the
48h `stale_thread` threshold in `situation.py`, so the key never expires out
from under a genuinely-stale-but-still-tracked thread and gets silently
misread as "unknown" instead of correctly classified as stale.

Fail-open by contract: every function here degrades to "no prior state"
(`None`, `None`) on any read/write failure, an unbound bus, or a malformed
payload -- never raises. A Redis hiccup here must not break a chat turn.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import NamedTuple

from orion.core.bus.async_service import OrionBusAsync

logger = logging.getLogger("orion.cortex.session_turn_phase")

_KEY_PREFIX = "orion:cortex-exec:session_turn_phase"
_DEFAULT_TTL_SECONDS = 7 * 24 * 3600  # 7 days


class SessionTurnState(NamedTuple):
    last_user_turn_at: datetime | None
    last_orion_turn_at: datetime | None


_NO_STATE = SessionTurnState(last_user_turn_at=None, last_orion_turn_at=None)


def _key(session_id: str) -> str:
    return f"{_KEY_PREFIX}:{session_id}"


def _parse_iso(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


_BUS: OrionBusAsync | None = None


def bind_session_turn_phase_bus(bus: OrionBusAsync) -> None:
    global _BUS
    _BUS = bus


def reset_session_turn_phase_bus_for_tests() -> None:
    global _BUS
    _BUS = None


async def read_session_turn_state(session_id: str) -> SessionTurnState:
    """Fail-open read of the last known user/Orion turn timestamps for this
    session, shared across every cortex-exec replica.

    Returns `(None, None)` -- never raises -- when the bus is unbound, the
    key is genuinely absent (a new/never-tracked session; not a failure,
    logged at INFO not WARNING), Redis errors, or the payload is
    malformed/incomplete.
    """
    bus = _BUS
    if bus is None:
        logger.warning("session_turn_phase_read_bus_unbound session_id=%s", session_id)
        return _NO_STATE

    key = _key(session_id)
    try:
        raw = await bus.redis.get(key)
    except Exception:
        logger.warning("session_turn_phase_read_failed key=%s redis_error", key, exc_info=True)
        return _NO_STATE

    if raw is None:
        logger.info("session_turn_phase_read key=%s found=False", key)
        return _NO_STATE

    try:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        parsed = json.loads(raw)
    except Exception:
        logger.warning("session_turn_phase_read_failed key=%s malformed_json", key, exc_info=True)
        return _NO_STATE

    if not isinstance(parsed, dict):
        logger.warning("session_turn_phase_read_failed key=%s not_a_dict", key)
        return _NO_STATE

    last_user = _parse_iso(parsed.get("last_user_turn_at"))
    last_orion = _parse_iso(parsed.get("last_orion_turn_at"))
    logger.info("session_turn_phase_read key=%s found=True", key)
    return SessionTurnState(last_user_turn_at=last_user, last_orion_turn_at=last_orion)


async def write_session_turn_state(
    session_id: str,
    *,
    last_user_turn_at: datetime | None,
    last_orion_turn_at: datetime | None,
    ttl_seconds: int = _DEFAULT_TTL_SECONDS,
) -> None:
    """Fail-open write of both timestamps as a single JSON payload via one
    SETEX call (not SET + EXPIRE -- see `last_tool_fetch_cache.py`'s own
    reasoning for why: two calls leave a window where the key exists
    without a TTL if the process dies between them).

    Callers are responsible for read-modify-write: this always writes both
    fields, so a caller updating only one timestamp must pass through the
    other one it already has (typically from a prior `read_session_turn_state`
    call in the same turn).
    """
    bus = _BUS
    if bus is None:
        logger.warning("session_turn_phase_write_bus_unbound session_id=%s", session_id)
        return

    key = _key(session_id)
    payload = json.dumps(
        {
            "last_user_turn_at": last_user_turn_at.astimezone(timezone.utc).isoformat() if last_user_turn_at else None,
            "last_orion_turn_at": last_orion_turn_at.astimezone(timezone.utc).isoformat() if last_orion_turn_at else None,
        }
    )
    try:
        await bus.redis.setex(key, ttl_seconds, payload)
        logger.info("session_turn_phase_write key=%s ttl_seconds=%s", key, ttl_seconds)
    except Exception:
        logger.warning("session_turn_phase_write_failed key=%s", key, exc_info=True)
