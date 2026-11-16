"""Hub side of the intention ledger (D1 view + D2 acts), SHADOW until #3531.

Design: docs/superpowers/specs/2026-11-11-intention-ledger-design.md and its 2026-11-16 amendments.

Before one of Juniper's turns, Hub reads (``promise_view_for_turn``):

* the open promises: ``hub.promise`` acts with no close and a horizon still ahead, oldest first, up to
  ``PROMISE_JUDGE_VIEW_MAX``; each quote is read from the promising turn's own record
  (``harness_turn_trace``), so the act never holds a copy;
* the pending offers: the offers in Orion's previous reply in this session, when that reply answered a
  Juniper turn (an outreach message's offers are out of v1).

Both ride ``HarnessRunRequestV1`` to the 5b judge ONLY. After the reply is delivered,
``publish_promise_acts`` opens one act per firm promise in it, one per earlier offer this message
accepted, and closes each promise the judge marked kept with a delivering quote
(orion/grammar/promise_acts.py ``plan_promise_events``).

The read is orion/grammar/promise_view.py (sync SQLAlchemy), run here on a worker thread with a timeout.
Fail-open everywhere: a read that fails shows the judge nothing and opens no act from an offer; the
turn itself never waits more than ``VIEW_TIMEOUT_SEC`` on it.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional, Sequence

from orion.grammar.promise_acts import plan_promise_events
from orion.grammar.promise_view import PromiseView, fetch_promise_view
from orion.schemas.harness_finalize import FinalizeReflectionV1

logger = logging.getLogger("orion.hub.promise_ledger")

VIEW_TIMEOUT_SEC = 2.0


async def promise_view_for_turn(session_id: Optional[str], correlation_id: str, *, open_too: bool = True,
                                timeout_sec: float = VIEW_TIMEOUT_SEC) -> PromiseView:
    """Never raises, never waits longer than ``timeout_sec``; an unread view is empty."""
    try:
        view = await asyncio.wait_for(
            asyncio.to_thread(fetch_promise_view, session_id, correlation_id, open_too=open_too), timeout=timeout_sec)
    except Exception as exc:  # noqa: BLE001 -- incl. the timeout
        logger.warning("promise_view_skipped corr=%s reason=%s", correlation_id, type(exc).__name__)
        return PromiseView()
    if view is None:
        return PromiseView()
    if view.open_promises or view.offers or view.unreadable:
        logger.info("promise_view corr=%s open_shown=%d offers=%d unreadable=%d", correlation_id,
                    len(view.open_promises), len(view.offers), view.unreadable)
    return view


async def publish_promise_acts(
    bus: Any,
    *,
    correlation_id: str,
    session_id: Optional[str],
    reflection: Optional[FinalizeReflectionV1],
    prior_offers: Sequence[tuple[str, str]] = (),
) -> dict[str, int]:
    """Publish the turn's promise act events. Best-effort: the reply is already delivered."""
    from orion.grammar.publish import publish_grammar_event

    plan = plan_promise_events(correlation_id=correlation_id, session_id=session_id, reflection=reflection,
                               prior_offers=prior_offers)
    published = failed = 0
    for event in plan.events:
        try:
            await publish_grammar_event(bus, event, source_name="orion-hub")
            published += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            logger.warning("promise_act_publish_failed corr=%s act=%s err=%s", correlation_id,
                           event.act.act_id if event.act else "-", exc)
    counts = {**plan.counts(), "published": published, "failed": failed}
    if plan.events or any(v for k, v in counts.items() if k not in ("published", "failed")):
        logger.info("promise_ledger corr=%s %s", correlation_id, " ".join(f"{k}={v}" for k, v in counts.items()))
    return counts


__all__ = ["PromiseView", "fetch_promise_view", "promise_view_for_turn", "publish_promise_acts"]
