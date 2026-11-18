"""A topic-foundry training/labelling run as one act in the act ledger (body economy Patch A5).

One ``_run_training`` call (app/services/training.py) is one ``maintain`` act, actor
``topic_foundry.run``: opened when the run starts work (status=running), closed once when it ends,
1 when the run completes (segments written, topics labelled, artifacts recorded), 0 when it fails
(including "No documents available for training", 37 of 38 failures over 7 days to 2026-10-23) or
raises anything at all.

Act key = the run id, so the act id is stable for the run. A process restart mid-run (e.g. every
post-pull rebuild of this service) leaves the act open; app/services/run_recovery.py reaps the run
row at startup (possibly as `complete`, if it has segments), and orion-sql-writer expires the act at
0 at its horizon. Deploy-interrupted runs grade 0 on purpose: the run did not finish in one go.

GPU cost join: NOT POSSIBLE YET. The act carries a correlation id derived from the run id, but the
run's LLM calls each keep their own per-call envelope id (live 10-16..23: 17,449 orion-topic-foundry
leases, 17,449 distinct ids), and orion-llm-gateway leases under the envelope id. Sharing one
envelope id across a run's ~200 sequential calls was tried and rejected in review: orion-bus-mirror
would read each gateway reply -> next request as a cross-organ causal hop with think time as its
latency, for up to an hour per run, polluting bus_synaptic_prediction_error's hop baselines.
Follow-up: the gateway forwards an explicit turn id (e.g. options.turn_correlation_id) to the pool
as the lease's turn_correlation_id, with the envelope id still fresh per call; then pass the run's
correlation there.

The event is built here, not with orion.grammar.acts.act_opened_event: that helper only knows the
hub and thought producer identities. ``build_act`` is reused as is. Gated by
TOPIC_FOUNDRY_ACT_EMIT_ENABLED. Every publish is best-effort: losing the trace never fails a run.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Optional
from uuid import NAMESPACE_URL, uuid4, uuid5

from orion.schemas.grammar import ActCostV1, ActOutcomeV1, ActV1, GrammarEventV1, GrammarProvenanceV1

logger = logging.getLogger("topic-foundry.act_emit")

ACTOR = "topic_foundry.run"
VERB = "maintain"
SOURCE_COMPONENT = "training_run.act"
PUBLISH_TIMEOUT_SEC = 5.0

# Measured 2026-10-23 over 30 days of topic_foundry_runs, status=complete, started_at -> completed_at
# (n=256): p50 221 s, p90 2335 s, p99 3143 s, max 3640 s. The horizon is the p90 rounded up to the
# hour, the same rule as the hub's reading act ("finishes in the time Orion's own runs usually
# take"). The run has no deadline of its own to cite. The longest observed runs (max 3640 s) exceed
# it and grade 0 (late) by design.
RUN_HORIZON_SEC = 3600.0
RUN_BASIS = (
    "artifact_produced (the run reaches status=complete: segments written, topics labelled, artifacts "
    "recorded) by 3600s after the run starts: p90 of complete runs started->completed over 30d is "
    "2335s (n=256, measured 2026-10-23), rounded up to the hour (max 3640s exceeds it: the longest "
    "runs grade late); the run has no deadline of its own"
)

def run_correlation_id(run_id: Any) -> str:
    """A UUID derived only from the run id (the act's correlation; no lease carries it yet)."""
    return str(uuid5(NAMESPACE_URL, f"orion:topic_foundry.run:{run_id}"))


def run_act(run_id: Any, *, model_ref: Optional[str] = None) -> ActV1:
    from orion.grammar.acts import build_act

    return build_act(
        actor=ACTOR, verb=VERB, key=str(run_id), signal_id="artifact_produced",
        horizon_sec=RUN_HORIZON_SEC, basis=RUN_BASIS,
        object_ref=model_ref or f"topic_foundry_run:{run_id}", test_before=f"topic_foundry_run:{run_id}",
    )


def _event(kind: str, act: ActV1, *, correlation_id: str, at: datetime) -> GrammarEventV1:
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
        provenance=GrammarProvenanceV1(source_service="orion-topic-foundry", source_component=SOURCE_COMPONENT),
    )


def _bus_publish(event: GrammarEventV1) -> None:
    """Synchronous publish from the run's worker thread (no running loop there)."""
    from orion.core.bus.async_service import OrionBusAsync
    from orion.grammar.publish import publish_grammar_event

    from app.settings import settings

    async def _go() -> None:
        bus = OrionBusAsync(url=settings.orion_bus_url, enabled=settings.orion_bus_enabled)
        await bus.connect()
        try:
            await publish_grammar_event(bus, event, source_name="orion-topic-foundry")
        finally:
            await bus.close()

    asyncio.run(asyncio.wait_for(_go(), timeout=PUBLISH_TIMEOUT_SEC))


class RunActTrace:
    """One training run's act: opened once, closed once. Never raises.

    Synchronous: ``_run_training`` runs in a FastAPI BackgroundTasks worker thread."""

    def __init__(self, run_id: Any, *, publish: Optional[Callable[[GrammarEventV1], None]],
                 model_ref: Optional[str] = None) -> None:
        self._publish = publish
        self.run_id = run_id
        self.act = run_act(run_id, model_ref=model_ref)
        self.correlation_id = run_correlation_id(run_id)
        self.opened_at: Optional[datetime] = None
        self.closed = False

    def _send(self, event: GrammarEventV1, what: str) -> None:
        try:
            self._publish(event)  # type: ignore[misc]
        except Exception:  # noqa: BLE001 -- the act is a trace; never fail the run
            logger.warning("topic_foundry_run_act_%s_failed run_id=%s", what, self.run_id, exc_info=True)

    def open(self) -> None:
        if self._publish is None or self.opened_at is not None:
            return
        self.opened_at = datetime.now(timezone.utc)
        self._send(_event("act_opened", self.act, correlation_id=self.correlation_id, at=self.opened_at), "open")

    def close(self, produced: bool, note: str) -> None:
        if self._publish is None or self.closed or self.opened_at is None:
            return
        self.closed = True
        at = datetime.now(timezone.utc)
        closed = self.act.model_copy(update={
            "outcome": ActOutcomeV1(value=1.0 if produced else 0.0, observed_at=at, source="producer",
                                    note=note[:500]),
            "cost": ActCostV1(seconds=round(max(0.0, (at - self.opened_at).total_seconds()), 3)),
        })
        self._send(_event("act_closed", closed, correlation_id=self.correlation_id, at=at), "close")


class _NoTrace:
    def open(self) -> None:
        pass

    def close(self, produced: bool, note: str) -> None:
        pass


def start_run_act(run_id: Any, *, model_ref: Optional[str] = None) -> Any:
    """Build and open the run's act. Never raises: a broken act build must not strand a run that
    is already marked `running` (the caller is outside the run's try)."""
    try:
        trace = RunActTrace(run_id, publish=default_publisher(), model_ref=model_ref)
        trace.open()
        return trace
    except Exception:  # noqa: BLE001
        logger.warning("topic_foundry_run_act_build_failed run_id=%s", run_id, exc_info=True)
        return _NoTrace()


def default_publisher() -> Optional[Callable[[GrammarEventV1], None]]:
    from app.settings import settings

    if not settings.topic_foundry_act_emit_enabled or not settings.orion_bus_enabled:
        return None
    return _bus_publish
