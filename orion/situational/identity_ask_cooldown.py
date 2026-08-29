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

Precedent, mirrored (not code-shared -- see below) from `session_turn_phase.py`,
which already had to fix the exact failure class an in-process flag would
repeat here: there are FOUR independent `orion-cortex-exec` replicas
(`orion-athena-cortex-exec`, `-chat`, `-background`, `-spark`), each with its
own process memory, and a single conversation can be routed across more than
one of them. An in-process "already asked" dict would let a different
replica ask again on the very next turn. This module uses the same
bind-bus-at-startup / Redis-backed shape, but is NOT a shared abstraction
with `session_turn_phase.py` (or `last_tool_fetch_cache.py`, which that
module itself cites as ITS precedent) -- each is its own small copy with its
own semantics (that one does two-field read-modify-write with an `.ok`
flag; this one is a single boolean claim). A shared "namespaced Redis TTL
flag" helper would be a reasonable follow-up if a fourth copy of this shape
shows up, not before.

**Single atomic claim, not check-then-set** (review finding, 2026-08-26): an
earlier version of this module exposed a separate `identity_ask_in_cooldown`
(GET) and `mark_identity_ask_offered` (SETEX) as two independent round-trips.
With four concurrent replicas, two could both read "not in cooldown" before
either had written its mark, asking twice -- exactly the awkward repetition
this module exists to prevent. `try_claim_identity_ask` below is the single
atomic `SET key val NX EX ttl` that replaces both: only the caller that
actually sets the key gets `True` back.

Keyed by camera STREAM, not chat session -- the thing being asked about
("do I recognize the person at this camera right now") is a property of the
camera, not of which typed/spoken surface Juniper happens to be talking
through. Also, deliberately, NOT keyed by subject: `identity_face`'s gallery
is capped at exactly one enrolled subject by contract
(`config/vision_profiles.yaml`), so "uncertain" only ever means one thing --
"not confidently Juniper" -- and there is no second subject identity this
cooldown could wrongly suppress an ask about.

Fail-open toward asking, not toward silence: every read/write failure here
returns True (claim succeeded, go ahead and ask), so a Redis hiccup costs at
most one redundant ask, never a feature that goes permanently mute because
of an infra blip.
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


def _key(stream_id: str, reason: str) -> str:
    """Keyed by (reason, stream), not stream alone (2026-08-29).

    There are now two distinct reasons to ask, with very different natural
    repeat rates, and a single shared key would let the common one starve the
    rare one. `unmatched_face` means a face was seen and did not match -- a
    strong, specific, usually transient signal worth re-raising within the
    hour. `no_visual_confirmation` means Orion simply has no fresh confirmed
    read at all (lid closed, camera off, nobody in frame) -- a background
    CONDITION rather than an event, true for hours at a stretch, and asking
    about it on the old 20-minute cadence would be roughly nine questions
    across an evening of lid-closed chat. Separate keys let each carry its
    own TTL (see SituationSettings' two cooldown fields).

    Changing the key shape retires any cooldown claimed under the old
    stream-only key; the one-time cost is at most one extra ask per camera
    on the first turn after deploy.
    """
    return f"{_KEY_PREFIX}:{reason}:{stream_id}"


async def try_claim_identity_ask(
    stream_id: str,
    *,
    reason: str = "unmatched_face",
    ttl_seconds: int = _DEFAULT_TTL_SECONDS,
) -> bool:
    """Atomically claim the "ask about this camera's identity mismatch"
    slot for `stream_id`. Returns True if THIS call claimed it -- safe to
    surface the clarifying question now, and the cooldown is already
    started. Returns False if another call (this replica moments ago, or a
    different one racing right now) already holds the claim.

    One `SET key val NX EX ttl` Redis round-trip, not a separate check then
    a separate write -- see module docstring for why that split let two
    replicas both ask.

    Never raises. Bus unbound or a Redis error both fail open to True (see
    module docstring for why fail-open points toward asking, not silence).
    """
    bus = _BUS
    if bus is None:
        logger.warning(
            "identity_ask_cooldown_claim_bus_unbound stream_id=%s reason=%s", stream_id, reason
        )
        return True
    key = _key(stream_id, reason)
    try:
        claimed = await bus.redis.set(key, "1", nx=True, ex=ttl_seconds)
    except Exception:
        logger.warning("identity_ask_cooldown_claim_failed key=%s redis_error", key, exc_info=True)
        return True
    result = bool(claimed)
    logger.info("identity_ask_cooldown_claim key=%s claimed=%s ttl_seconds=%s", key, result, ttl_seconds)
    return result
