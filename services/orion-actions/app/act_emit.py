"""orion-actions producers for the act ledger (body economy Patch A5).

Two acts, one per attempt that starts real work (after the dedupe gate and the semaphore wait):

* ``actions.daily_metacog`` / ``infer``: one scheduled or manual daily_metacog_v1 attempt
  (``_execute_daily``). Produced = the report was built and delivered (notify accepted).
  daily_metacog_v1 is OFF live on purpose (ACTIONS_DAILY_METACOG_ENABLED=false, Juniper 10-11):
  this emits nothing until the job runs.
* ``actions.journal`` / ``make``: one in-process journal compose (``_dispatch_journal`` ->
  ``_run_journal`` -> cortex-orch ``journal.compose``). Produced = a draft came back and its
  ``journal.entry.write.v1`` was published.

Journal ownership rule: the process that runs the journal's compose owns its act.
``_dispatch_journal`` composes in-process here, so orion-actions emits ``actions.journal``.
A journal submitted as a durable ``journal.compose`` run (today: world_pulse_digest, via
``app/world_pulse_journal.py`` -> cortex-orch durable ingress) is composed by orion-durable-runs,
which owns that act at its run seam; that path never calls ``_dispatch_journal``, so no journal is
counted by both.

Act key = (the job's dedupe key, the attempt envelope's correlation id). A redelivered trigger
carries the same correlation id, so it maps to the same act id (the writer is first-close-wins).
A scheduler retry is a new attempt with a new correlation id: a new act, because it spends its
own GPU time. A restart mid-attempt leaves the act open; orion-sql-writer expires it at 0 at its
horizon.

correlation_id = the attempt envelope's correlation id. cortex-orch / cortex-exec take their GPU
leases under it (verified live 10-23 for journal composes: lease turn_correlation_id = the journal's
correlation_id), so the ledger's close-time cost fill joins.

The event is built here, not with orion.grammar.acts.act_opened_event: that helper only knows
the hub and thought producer identities. ``build_act`` is reused as is. Gated by
ACTIONS_ACT_EMIT_ENABLED. Every publish is best-effort: losing the trace never fails the job.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional
from uuid import uuid4

from orion.schemas.grammar import ActCostV1, ActOutcomeV1, ActV1, GrammarEventV1, GrammarProvenanceV1

logger = logging.getLogger("orion-actions.act_emit")

SOURCE_SERVICE = "orion-actions"
PUBLISH_TIMEOUT_SEC = 5.0
# Slack on top of the RPC budget for the notify / journal-write publish after the reply.
_PUBLISH_SLACK_SEC = 60.0

DAILY_METACOG_ACTOR = "actions.daily_metacog"
JOURNAL_ACTOR = "actions.journal"


def daily_metacog_horizon_sec(exec_timeout_sec: float) -> float:
    # _execute_daily: up to 2 generation attempts (truncation / skill-validation retry), each a
    # _rpc_request_with_retry with max_attempts=2, each bounded by ACTIONS_EXEC_TIMEOUT_SECONDS.
    return 4.0 * float(exec_timeout_sec) + _PUBLISH_SLACK_SEC


def journal_horizon_sec(exec_timeout_sec: float) -> float:
    # _run_journal: one cortex-orch RPC bounded by ACTIONS_EXEC_TIMEOUT_SECONDS.
    return float(exec_timeout_sec) + _PUBLISH_SLACK_SEC


def _daily_basis(h: float) -> str:
    return (
        f"artifact_produced (the daily metacog report built and delivered) by {h:.0f}s: the job's own "
        "deadline, 2 generation attempts x 2 RPC attempts x ACTIONS_EXEC_TIMEOUT_SECONDS + "
        f"{_PUBLISH_SLACK_SEC:.0f}s for notify; not measured (job off since 2026-09-18)"
    )


def _journal_basis(h: float) -> str:
    return (
        f"artifact_produced (a composed journal draft whose journal.entry.write.v1 was published) by "
        f"{h:.0f}s: the compose RPC's own deadline, ACTIONS_EXEC_TIMEOUT_SECONDS + "
        f"{_PUBLISH_SLACK_SEC:.0f}s for the write publish"
    )


def daily_metacog_act(dedupe_key: str, correlation_id: str, *, horizon: float) -> ActV1:
    from orion.grammar.acts import build_act

    return build_act(
        actor=DAILY_METACOG_ACTOR, verb="infer", key=f"{dedupe_key}|{correlation_id}",
        signal_id="artifact_produced", horizon_sec=horizon, basis=_daily_basis(horizon),
        object_ref=dedupe_key, test_before=f"daily_metacog:{dedupe_key}",
    )


def journal_act(dedupe_key: str, correlation_id: str, *, trigger_kind: str, horizon: float) -> ActV1:
    from orion.grammar.acts import build_act

    return build_act(
        actor=JOURNAL_ACTOR, verb="make", key=f"{dedupe_key}|{correlation_id}",
        signal_id="artifact_produced", horizon_sec=horizon, basis=_journal_basis(horizon),
        object_ref=f"journal_trigger:{trigger_kind}", test_before=f"journal:{dedupe_key}",
    )


def _event(kind: str, act: ActV1, *, component: str, correlation_id: str, at: datetime) -> GrammarEventV1:
    # Same shape as orion.grammar.acts._event; the provenance literal is what
    # tests/test_grammar_event_producer_catalog.py reads to keep channels.yaml honest.
    return GrammarEventV1(
        event_id=f"{kind}:{act.act_id}:{uuid4().hex[:8]}",
        event_kind=kind,  # type: ignore[arg-type]
        trace_id=f"act:{act.act_id}",
        correlation_id=correlation_id,
        emitted_at=datetime.now(timezone.utc),
        observed_at=at,
        layer="act",
        dimensions=["act", act.verb, act.level],
        act=act,
        provenance=GrammarProvenanceV1(source_service="orion-actions", source_component=component),
    )


Publish = Callable[[GrammarEventV1], Awaitable[None]]


class ActTrace:
    """One attempt's act: opened once, closed once. Never raises.

    The close is sent whenever an open was attempted, even if the open publish failed: the writer
    then builds the row from the close (close-before-open) and still grades it."""

    def __init__(self, publish: Optional[Publish], act: ActV1, *, component: str, correlation_id: str) -> None:
        self._publish = publish
        self.act = act
        self.component = component
        self.correlation_id = correlation_id
        self.opened_at: Optional[datetime] = None
        self.closed = False

    async def _send(self, event: GrammarEventV1, what: str) -> None:
        try:
            await asyncio.wait_for(self._publish(event), timeout=PUBLISH_TIMEOUT_SEC)  # type: ignore[misc]
        except Exception:  # noqa: BLE001 -- the act is a trace; never fail the job
            logger.warning("act_%s_failed actor=%s act_id=%s", what, self.act.actor, self.act.act_id, exc_info=True)

    async def open(self) -> None:
        if self._publish is None or self.opened_at is not None:
            return
        self.opened_at = datetime.now(timezone.utc)
        await self._send(_event("act_opened", self.act, component=self.component,
                                correlation_id=self.correlation_id, at=self.opened_at), "open")

    async def close(self, produced: bool, note: str) -> None:
        if self._publish is None or self.closed or self.opened_at is None:
            return
        self.closed = True
        at = datetime.now(timezone.utc)
        closed = self.act.model_copy(update={
            "outcome": ActOutcomeV1(value=1.0 if produced else 0.0, observed_at=at, source="producer",
                                    note=note[:500]),
            "cost": ActCostV1(seconds=round(max(0.0, (at - self.opened_at).total_seconds()), 3)),
        })
        await self._send(_event("act_closed", closed, component=self.component,
                                correlation_id=self.correlation_id, at=at), "close")


def bus_publisher(bus_getter: Callable[[], Any]) -> Publish:
    async def _publish(event: GrammarEventV1) -> None:
        from orion.grammar.publish import publish_grammar_event

        await publish_grammar_event(bus_getter(), event, source_name=SOURCE_SERVICE)

    return _publish
