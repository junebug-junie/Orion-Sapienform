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
TTL: 7 days by default (`ttl_seconds` kwarg) -- comfortably covers the 48h
`stale_thread` threshold in `situation.py` with room to spare (a gap longer
than 7 days is still possible -- `stale_thread` has no upper bound -- and
in that case the key legitimately expires and the thread reads "unknown"
rather than "stale"; the TTL bounds the window this can misclassify, it
does not eliminate it entirely).

Fail-open by contract: every function here degrades to "no prior state"
(`None`, `None`) on any read/write failure, an unbound bus, or a malformed
payload -- never raises. A Redis hiccup here must not break a chat turn.

IMPORTANT for callers doing read-modify-write (both `situation.py`
functions that use this module do): `SessionTurnState.ok` distinguishes a
read that genuinely confirmed "no prior data" (`ok=True`, both fields
`None` -- safe to write both fields fresh) from a read that FAILED and
therefore has an UNKNOWN true state (`ok=False`, both fields `None`
because we don't know them, not because they're empty). Blindly writing
`(None, None)` back on `ok=False` would silently clobber a real,
previously-good timestamp on the field the caller wasn't even trying to
update -- worse than the bug this module fixes. Callers must check `.ok`
and skip the write entirely when it's `False`.

KNOWN, ACCEPTED LIMITATION: the read and its matching write are not
atomic (no Redis transaction/WATCH, no hash field ops), so two different
replicas racing on the exact same session within the same sub-millisecond
window could lose one field's update. Self-heals on the next turn either
way, and one process only ever writes the field it's actively updating
while carrying forward the other field it just read -- worth closing with
a Redis hash (one field per writer, no read-modify-write at all) if this
ever proves to matter in practice, not before.
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
    # True: read genuinely completed (key found and parsed, or confirmed
    # absent) -- both fields are trustworthy as-is, safe to write back.
    # False: read FAILED (unbound bus, Redis error, malformed payload) --
    # both fields are None because the true state is unknown, not because
    # it's empty. See the module docstring's read-modify-write warning.
    ok: bool = True


_UNKNOWN_STATE = SessionTurnState(last_user_turn_at=None, last_orion_turn_at=None, ok=False)
_CONFIRMED_EMPTY_STATE = SessionTurnState(last_user_turn_at=None, last_orion_turn_at=None, ok=True)


def _key(session_id: str) -> str:
    return f"{_KEY_PREFIX}:{session_id}"


def _parse_iso(value: object) -> datetime | None:
    """Parse an ISO8601 string, normalizing a tz-naive result to UTC.

    `write_session_turn_state` always writes a tz-aware UTC string, but a
    foreign/hand-edited payload could contain a naive one. Left unnormalized,
    a naive value later hits `now_utc - last_user` in
    `situation._build_conversation_phase` and raises `TypeError: can't
    subtract offset-naive and offset-aware datetimes` -- which would escape
    this "never raises" module and, worse, propagate out of
    `build_situation_for_ctx` into `executor.py`'s broad except block,
    discarding the ENTIRE situation brief (weather/presence/perception/
    runtime too, not just phase) over one bad timestamp field. Confirmed
    live before this fix. `mark_orion_turn`'s `.astimezone(timezone.utc)`
    on a naive value doesn't raise either -- it silently reinterprets it as
    local time and writes a shifted timestamp back, which is worse (a loud
    failure would at least have been visible).
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


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

    Returns a `SessionTurnState` with `ok=True` and both fields `None` when
    the key is genuinely absent (a new/never-tracked session; not a
    failure, logged at INFO not WARNING) OR when the payload was read
    successfully but is malformed/not-a-dict -- in both cases the read
    itself succeeded and we affirmatively know there is no usable prior
    data, so it is safe for a caller to overwrite fresh (a malformed
    payload must self-heal on the next write, not self-poison for the
    remaining TTL by permanently blocking every future write too). Returns
    `ok=False` (also both fields `None`, but meaning "unknown", not
    "confirmed empty") -- never raises -- only when the read itself could
    not be trusted at all: the bus is unbound or Redis errored.
    """
    bus = _BUS
    if bus is None:
        logger.warning("session_turn_phase_read_bus_unbound session_id=%s", session_id)
        return _UNKNOWN_STATE

    key = _key(session_id)
    try:
        raw = await bus.redis.get(key)
    except Exception:
        logger.warning("session_turn_phase_read_failed key=%s redis_error", key, exc_info=True)
        return _UNKNOWN_STATE

    if raw is None:
        logger.info("session_turn_phase_read key=%s found=False", key)
        return _CONFIRMED_EMPTY_STATE

    try:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        parsed = json.loads(raw)
    except Exception:
        # The GET itself succeeded -- this is a confirmed-unusable payload,
        # not an unknown read outcome, so ok=True (see docstring above: a
        # caller must be able to overwrite this on its next write rather
        # than have it wedge for the rest of the TTL).
        logger.warning("session_turn_phase_read_failed key=%s malformed_json", key, exc_info=True)
        return _CONFIRMED_EMPTY_STATE

    if not isinstance(parsed, dict):
        logger.warning("session_turn_phase_read_failed key=%s not_a_dict", key)
        return _CONFIRMED_EMPTY_STATE

    last_user = _parse_iso(parsed.get("last_user_turn_at"))
    last_orion = _parse_iso(parsed.get("last_orion_turn_at"))
    logger.info("session_turn_phase_read key=%s found=True", key)
    return SessionTurnState(last_user_turn_at=last_user, last_orion_turn_at=last_orion, ok=True)


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
    try:
        # Payload construction lives inside the try too, not just the
        # setex call -- .astimezone() can raise on a pathological/out-of-
        # range datetime, and this module's whole contract is "never
        # raises," not "never raises past the network call."
        payload = json.dumps(
            {
                "last_user_turn_at": last_user_turn_at.astimezone(timezone.utc).isoformat() if last_user_turn_at else None,
                "last_orion_turn_at": last_orion_turn_at.astimezone(timezone.utc).isoformat() if last_orion_turn_at else None,
            }
        )
        await bus.redis.setex(key, ttl_seconds, payload)
        logger.info("session_turn_phase_write key=%s ttl_seconds=%s", key, ttl_seconds)
    except Exception:
        logger.warning("session_turn_phase_write_failed key=%s", key, exc_info=True)
