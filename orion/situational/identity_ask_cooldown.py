"""Cross-process cooldown for the "is that you?" identity clarifying
question, so Orion asks it at most once per sit-down instead of once per
turn.

Built 2026-08-26 alongside `PerceptionContextV1.presence_identity_uncertain`
(Juniper, direct ask: "have Orion say something ... friendly, hi i'm having
trouble recognizing you. is that juniper? ... we need something solid to
know if it is broke, running, or not running and not make speech akward").
The "not make speech awkward" half of that ask is THIS module's whole job --
without it, a genuinely persistent unsure reading (bad lighting, an odd
angle) would re-surface on essentially every turn for as long as it lasts,
since `_build_perception_context` rebuilds the situation brief fresh most
turns.

Precedent, deliberately reused rather than reinvented: `session_turn_phase.py`
already had to fix the exact failure class an in-process flag would repeat
here -- there are FOUR independent `orion-cortex-exec` replicas
(`orion-athena-cortex-exec`, `-chat`, `-background`, `-spark`), each with its
own process memory, and a single conversation can be routed across more than
one of them. An in-process "already asked" dict would let a different
replica ask again on the very next turn. This module is the same
bind-bus-at-startup / Redis-backed shape as `session_turn_phase.py`, just
simpler: one boolean per stream (not two timestamps), so a single SET-with-
TTL is the whole write path -- no read-modify-write race to worry about.

Keyed by camera STREAM, not chat session -- the thing being asked about
("do I recognize the person at this camera right now") is a property of the
camera, not of which typed/spoken surface Juniper happens to be talking
through.

Fail-open toward asking, not toward silence: every read failure here returns
False (not in cooldown), so a Redis hiccup costs at most one redundant ask,
never a feature that goes permanently mute because of an infra blip.
"""

from __future__ import annotations

import logging

from orion.core.bus.async_service import OrionBusAsync

logger = logging.getLogger("orion.cortex.identity_ask_cooldown")

_KEY_PREFIX = "orion:cortex-exec:identity_ask_cooldown"
# 20 minutes: long enough that "ask once per sit-down" is the felt
# experience in a normal conversation, short enough that someone who fixes
# their lighting or turns their laptop lid up isn't stuck being silently
# mis-recognized-and-never-re-asked for the rest of the day.
_DEFAULT_TTL_SECONDS = 20 * 60

_BUS: OrionBusAsync | None = None


def bind_identity_ask_cooldown_bus(bus: OrionBusAsync) -> None:
    global _BUS
    _BUS = bus


def reset_identity_ask_cooldown_bus_for_tests() -> None:
    global _BUS
    _BUS = None


def _key(stream_id: str) -> str:
    return f"{_KEY_PREFIX}:{stream_id}"


async def identity_ask_in_cooldown(stream_id: str) -> bool:
    """True if Orion has already offered the clarifying question for this
    stream within the cooldown window and should stay quiet about it now.

    Never raises. Bus unbound or a Redis error both degrade to False (see
    module docstring for why fail-open points toward asking, not silence).
    """
    bus = _BUS
    if bus is None:
        logger.warning("identity_ask_cooldown_read_bus_unbound stream_id=%s", stream_id)
        return False
    key = _key(stream_id)
    try:
        raw = await bus.redis.get(key)
    except Exception:
        logger.warning("identity_ask_cooldown_read_failed key=%s redis_error", key, exc_info=True)
        return False
    return raw is not None


async def mark_identity_ask_offered(stream_id: str, *, ttl_seconds: int = _DEFAULT_TTL_SECONDS) -> None:
    """Best-effort, never raises. Call right after deciding to surface
    `presence_identity_uncertain=True` for this stream in a turn, so every
    replica sees the cooldown starting on the very next read."""
    bus = _BUS
    if bus is None:
        logger.warning("identity_ask_cooldown_write_bus_unbound stream_id=%s", stream_id)
        return
    key = _key(stream_id)
    try:
        await bus.redis.setex(key, ttl_seconds, "1")
        logger.info("identity_ask_cooldown_write key=%s ttl_seconds=%s", key, ttl_seconds)
    except Exception:
        logger.warning("identity_ask_cooldown_write_failed key=%s", key, exc_info=True)
