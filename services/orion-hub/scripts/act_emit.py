"""Hub producers for the act ledger (rung 2, "doing on one track").

Two hub acts publish here (the chat reply's ``say`` lives next to the reply in
orion/hub/turn_orchestrator.py):

- Endogenous outreach -> ``say`` (or ``ask`` when the delivered text contains a
  question mark: a syntactic test of whether the message asks Juniper
  something, not a guess at intent). Expects Juniper to send a chat turn within
  OUTREACH_REPLY_HORIZON_SEC; orion-sql-writer's resolver closes it.
- World-pulse reading, stage 1 -> ``look``. Opened once, when the read's durable
  run starts real work (not when it is queued: the GPU queue wait is not the read).
  Expects a read with real evidence (stage1_completed) within READ_HORIZON_SEC of
  that start; closed here on stage1_completed (1) or stage1_failed (0), expired at
  0 by the writer if neither arrives. A read refused before work never opens.

Every function is best-effort and never raises: the act is a trace of doing,
and losing one must never block the doing itself. Gated by
HUB_ACT_LEDGER_ENABLED.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("orion-hub.act_emit")

# Outreach: 36000 s. Measured 2026-10-16 over 60 days of chat_history_log
# (n=238 unsolicited outreach rows): median time from an outreach message to
# Juniper's next chat turn is 36792 s; P(turn <= 1800 s) = 0.04, <= 4 h = 0.18,
# <= 12 h = 0.63. Re-measured the same day on the resolver's clock (when her next
# turn STARTED, see orion/grammar/act_ledger.py _JUNIPER_REPLY_SQL): median 36554 s,
# P(<= 36000 s) = 0.49 on both clocks, so the horizon stands. The same "horizon at the median" rule as chat replies, so the
# binary outcome is not saturated. Known confound: at this horizon the outcome
# is dominated by Juniper's daily rhythm, not the message (recorded in the PR).
OUTREACH_REPLY_HORIZON_SEC = 36000.0
OUTREACH_REPLY_BASIS = (
    "juniper_replied by 36000s: median time from an outreach message to Juniper's next chat turn "
    "start over 60d (36554s, n=238, measured 2026-10-16); same horizon-at-median rule as chat replies"
)

# Reading: 3600 s from the moment the read's durable run starts work (its first
# run.started event). Re-measured 2026-10-16 over 30 days of reading.turn runs on
# that clock: started -> completed p50 484 s, p90 1840 s, p99 3757 s (n=386). The
# horizon is the p90 rounded up to the hour: "a read finishes in the time Orion's
# own reads usually take". The first measurement (p90 2734 s) was taken from run
# creation, which counts the GPU queue wait; on 2026-10-16 that wait alone was
# 0.5 to 3.3 h, so 6 of 7 reads that produced were graded late. The queue wait is
# still recorded per run (durable-runs queue_wait_seconds); it is not the read.
READ_HORIZON_SEC = 3600.0
READ_BASIS = (
    "artifact_produced (stage1_completed with read evidence) by 3600s after the read's durable "
    "run starts work: p90 of reading.turn run.started->completed over 30d is 1840s "
    "(n=386, measured 2026-10-16), rounded up to the hour; GPU queue wait before the first start "
    "excluded (a retried read's re-queue still counts: the act is the seed, retries included)"
)


def _enabled() -> bool:
    try:
        from scripts.settings import settings

        return bool(getattr(settings, "HUB_ACT_LEDGER_ENABLED", True))
    except Exception:  # noqa: BLE001
        return True


async def publish_outreach_act(
    bus: Any,
    *,
    text: str,
    message_id: str,
    correlation_id: str,
    session_id: str | None,
    source_tag: str | None,
) -> bool:
    if bus is None or not _enabled():
        return False
    try:
        from orion.grammar.acts import act_opened_event, build_act
        from orion.grammar.publish import publish_grammar_event

        act = build_act(
            actor="hub.outreach",
            verb="ask" if "?" in (text or "") else "say",
            key=str(message_id),
            signal_id="juniper_replied",
            horizon_sec=OUTREACH_REPLY_HORIZON_SEC,
            basis=OUTREACH_REPLY_BASIS,
            object_ref="juniper",
            test_before=f"outreach:{source_tag or 'self'}",
        )
        event = act_opened_event(
            act, source_service="orion-hub", source_component="endogenous_outreach.act",
            correlation_id=str(correlation_id), session_id=session_id,
        )
        await publish_grammar_event(bus, event, source_name="orion-hub")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("outreach_act_publish_failed message_id=%s error=%s", message_id, exc)
        return False


def _read_act(seed: Any):
    from orion.grammar.acts import build_act

    return build_act(
        actor="hub.world_pulse_read",
        verb="look",
        key=str(getattr(seed, "seed_id", "")),
        signal_id="artifact_produced",
        horizon_sec=READ_HORIZON_SEC,
        basis=READ_BASIS,
        object_ref=str(getattr(seed, "url", "") or "")[:500] or None,
        test_before=f"read_seed:{getattr(seed, 'seed_id', '')}",
    )


def _seed_correlation(seed: Any) -> str | None:
    request = getattr(seed, "request", None)
    rid = getattr(request, "request_id", None)
    return str(rid) if rid else None


async def publish_read_opened(bus: Any, seed: Any, *, at: Any = None) -> bool:
    """Open the seed's `look` act. ``at`` is when the read's work started (the
    writer sets opened_at and the horizon from it); None means now."""
    if bus is None or not _enabled():
        return False
    try:
        from orion.grammar.acts import act_opened_event
        from orion.grammar.publish import publish_grammar_event

        event = act_opened_event(
            _read_act(seed), source_service="orion-hub", source_component="world_pulse_read.act",
            correlation_id=_seed_correlation(seed), at=at,
        )
        await publish_grammar_event(bus, event, source_name="orion-hub")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("read_act_open_failed seed=%s error=%s", getattr(seed, "seed_id", None), exc)
        return False


async def publish_read_closed(bus: Any, seed: Any, *, produced: bool, note: str = "") -> bool:
    if bus is None or not _enabled():
        return False
    try:
        from orion.grammar.acts import act_closed_event
        from orion.grammar.publish import publish_grammar_event

        event = act_closed_event(
            _read_act(seed), produced=produced, note=note,
            source_service="orion-hub", source_component="world_pulse_read.act",
            correlation_id=_seed_correlation(seed),
        )
        await publish_grammar_event(bus, event, source_name="orion-hub")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("read_act_close_failed seed=%s error=%s", getattr(seed, "seed_id", None), exc)
        return False
