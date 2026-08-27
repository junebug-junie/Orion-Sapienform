"""Cross-process storage for "what did Orion most recently see of Juniper's
facial+vocal affect" -- the raw input `context.py::_build_affect_context`
gates on age and folds into the situation brief.

**Why this exists.** `orion-affectgpt-worker` (circe GPU1, real AffectGPT +
Whisper inference) already produces a real, grounded affect read every time
Hub's "Check now" button or ambient toggle fires a capture --
`orion-juniper-affective-state` relays it as `JuniperMultimodalAffectV1` on
the `orion:affectgpt:assessment` bus channel. As of 2026-08-25, NOTHING
downstream ever consumed that channel except a manual debug CLI
(`services/orion-juniper-affective-state/scripts/tap_assessments.py`) --
Orion's own chat turns never found out. This module is the seam that closes
that loop: the write side lets the producer mirror its latest successful
read into one Redis key; the read side lets `orion/situational/context.py`
(which runs inside orion-hub's and orion-cortex-exec's own event loops, not
orion-juniper-affective-state's) pick it back up.

**Storage shape and precedent.** Single JSON payload per key, one SETEX
call, fail-open, "not found" vs "read/write failed" logged distinctly --
same shape as `session_turn_phase.py` (see that module's own docstring for
why: `orion/harness/last_tool_fetch_cache.py` is the original precedent).
Unlike `session_turn_phase.py`, this is NOT per-session -- Juniper is the
sole subject, one physical camera/mic, so one global key
(`orion:juniper_affect:latest`) is the honest shape, not a fabricated
per-session dimension.

**Bus handle.** The WRITE side takes an explicit `bus` parameter rather than
using a module-level bind: `orion-juniper-affective-state`'s
`AffectStateService` already carries `self.bus` at every call site that
would write here, so a global bind adds ceremony with no caller that needs
it. The READ side DOES use the bind pattern
(`bind_juniper_affect_state_bus`/`reset_juniper_affect_state_bus_for_tests`),
mirroring `session_turn_phase.py` exactly -- `context.py`'s call chain
(`build_situation_for_ctx` -> `_build_affect_context`) never receives a bus
parameter, same reason cited in that module's docstring.

**TTL is a safety net, not the freshness gate.** The write always sets a
generous outer TTL (`_WRITE_TTL_SECONDS`, 1h) so a crashed/misconfigured
producer cannot leave a stale read alive indefinitely. The actual "is this
fresh enough to color a turn" decision is a separate, much tighter
`observation_age_seconds` check the CALLER (`_build_affect_context`) makes
against its own configured `orion_situation_affect_max_age_seconds` -- same
split perception_reader.py uses (`fetch_latest_percept` always returns the
newest row regardless of age; `context.py` owns the staleness gate).

**Privacy.** `summary` is caller-supplied. Callers (only
`orion-juniper-affective-state`'s `_publish_event` as of this writing) must
pass an already-truncated excerpt of the model's `raw_response`, never the
verbatim transcript -- see `AffectContextV1`'s docstring
(`orion/schemas/situation.py`) for the contract this key's reader exposes
downstream. This module itself does not enforce the truncation; it is a
plain fail-open KV store, same trust boundary `session_turn_phase.py` has
with its own callers.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import NamedTuple

from orion.core.bus.async_service import OrionBusAsync

logger = logging.getLogger("orion.situational.juniper_affect_state")

_KEY = "orion:juniper_affect:latest"
_WRITE_TTL_SECONDS = 3600  # 1h outer safety net -- see module docstring.


class JuniperAffectState(NamedTuple):
    summary: str | None
    observed_at: datetime | None
    trigger: str | None
    subtitle_source: str | None
    # The producing backend's own confidence in this read, 0.0-1.0, or None
    # for a read written before backends were distinguished (2026-08-26) or
    # by the legacy affectgpt path, which had no confidence to report.
    # Present here so the PROMPT can hedge proportionally -- the write-side
    # gate in orion-juniper-affective-state has already rejected anything
    # below AFFECT_MIRROR_MIN_CONFIDENCE, so a value that arrives here is
    # above the bar but not necessarily near 1.0.
    confidence: float | None = None
    # "vision" | "affectgpt" | None (pre-2026-08-26 payload).
    backend: str | None = None
    # True: read genuinely completed (key found and parsed, or confirmed
    # absent) -- the other fields are trustworthy as-is. False: read FAILED
    # (unbound bus, Redis error, malformed payload) -- all other fields are
    # None because the true state is unknown, not because it's empty. Same
    # ok-vs-fields-are-None distinction as `session_turn_phase.SessionTurnState`.
    ok: bool = True


_UNKNOWN_STATE = JuniperAffectState(
    summary=None, observed_at=None, trigger=None, subtitle_source=None, ok=False
)
_CONFIRMED_EMPTY_STATE = JuniperAffectState(
    summary=None, observed_at=None, trigger=None, subtitle_source=None, ok=True
)


def _parse_iso(value: object) -> datetime | None:
    """Parse an ISO8601 string, normalizing a tz-naive result to UTC.

    Same reasoning as `session_turn_phase._parse_iso`: a naive datetime
    reaching `percept_age_seconds`-style subtraction later raises
    `TypeError: can't subtract offset-naive and offset-aware datetimes`,
    which must not escape this "never raises" module.
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


_BUS: OrionBusAsync | None = None


def bind_juniper_affect_state_bus(bus: OrionBusAsync) -> None:
    global _BUS
    _BUS = bus


def reset_juniper_affect_state_bus_for_tests() -> None:
    global _BUS
    _BUS = None


async def read_latest_juniper_affect() -> JuniperAffectState:
    """Fail-open read of the most recent affect capture, or the confirmed-
    absent/unknown states described on `JuniperAffectState`. Never raises."""
    bus = _BUS
    if bus is None:
        logger.warning("juniper_affect_state_read_bus_unbound")
        return _UNKNOWN_STATE

    try:
        raw = await bus.redis.get(_KEY)
    except Exception:
        logger.warning("juniper_affect_state_read_failed key=%s redis_error", _KEY, exc_info=True)
        return _UNKNOWN_STATE

    if raw is None:
        logger.info("juniper_affect_state_read key=%s found=False", _KEY)
        return _CONFIRMED_EMPTY_STATE

    try:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        parsed = json.loads(raw)
    except Exception:
        logger.warning("juniper_affect_state_read_failed key=%s malformed_json", _KEY, exc_info=True)
        return _CONFIRMED_EMPTY_STATE

    if not isinstance(parsed, dict):
        logger.warning("juniper_affect_state_read_failed key=%s not_a_dict", _KEY)
        return _CONFIRMED_EMPTY_STATE

    logger.info("juniper_affect_state_read key=%s found=True", _KEY)
    return JuniperAffectState(
        summary=parsed.get("summary") if isinstance(parsed.get("summary"), str) else None,
        observed_at=_parse_iso(parsed.get("observed_at")),
        trigger=parsed.get("trigger") if isinstance(parsed.get("trigger"), str) else None,
        subtitle_source=parsed.get("subtitle_source")
        if isinstance(parsed.get("subtitle_source"), str)
        else None,
        # `not isinstance(..., bool)`: bool is a subclass of int, so a payload
        # carrying `"confidence": true` would otherwise read back as 1.0 --
        # maximum confidence from a value that expressed none (review finding,
        # 2026-08-26).
        confidence=float(parsed["confidence"])
        if isinstance(parsed.get("confidence"), (int, float))
        and not isinstance(parsed.get("confidence"), bool)
        else None,
        backend=parsed.get("backend") if isinstance(parsed.get("backend"), str) else None,
        ok=True,
    )


async def write_latest_juniper_affect(
    bus: OrionBusAsync,
    *,
    summary: str,
    observed_at: datetime,
    trigger: str,
    subtitle_source: str | None,
    confidence: float | None = None,
    backend: str | None = None,
    ttl_seconds: int = _WRITE_TTL_SECONDS,
) -> None:
    """Fail-open write of the latest affect read. Never raises -- a failed
    write here must not break the caller's own bus-publish path."""
    try:
        payload = json.dumps(
            {
                "summary": summary,
                "observed_at": observed_at.astimezone(timezone.utc).isoformat(),
                "trigger": trigger,
                "subtitle_source": subtitle_source,
                "confidence": confidence,
                "backend": backend,
            }
        )
        await bus.redis.setex(_KEY, ttl_seconds, payload)
        logger.info("juniper_affect_state_write key=%s ttl_seconds=%s", _KEY, ttl_seconds)
    except Exception:
        logger.warning("juniper_affect_state_write_failed key=%s", _KEY, exc_info=True)
