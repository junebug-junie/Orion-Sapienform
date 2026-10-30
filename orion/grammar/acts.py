"""Producer helpers for the act ledger (rung 2, "doing on one track").

A producer calls ``act_opened_event`` before it acts and ``act_closed_event``
when the outcome is known, then publishes each with
``orion.grammar.publish.publish_grammar_event``. Both events share
``trace_id = "act:<act_id>"`` so they land on one sql-writer grammar lane, in
order. orion-sql-writer persists them (orion/grammar/act_ledger.py).

act_id is deterministic from (actor, verb, key) so a retry of the same act
re-uses the same identity and the writer's first-close-wins rule holds.

Expectation horizons for each world-facing act live HERE, next to their
derivation, so every producer reads one number and the derivation is checked
in once:

- ``JUNIPER_REPLY_HORIZON_SEC`` (say / ask): 1800 s. Measured 2026-10-16 over
  60 days of chat_history_log hub turns (n=241 gaps): median gap between
  consecutive Juniper turns 2015 s, P(next turn <= 1800 s) = 0.49. A horizon
  at the median keeps the binary outcome near maximum entropy, so the
  signal is not saturated at 0 or 1 by construction.
  Re-measured 2026-10-16 on the resolver's own clock (when Juniper's next turn
  STARTED: hub ingress hop, logged created_at fallback; see act_ledger
  _JUNIPER_REPLY_SQL): median from Orion's stored reply to her next turn start
  1785 s, P(<= 1800 s) = 0.50 (n=250). The horizon still sits at the median.
- Artifact horizons (make / look) come from the producer's own existing
  deadline (the run's timeout), passed in by the caller and named in
  ``basis``: "produced by the deadline Orion already gives itself".
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from orion.schemas.grammar import (
    ActCostV1,
    ActExpectV1,
    ActLevel,
    ActOutcomeV1,
    ActSignal,
    ActV1,
    ActVerb,
    GrammarEventV1,
    GrammarProvenanceV1,
)

JUNIPER_REPLY_HORIZON_SEC = 1800.0
JUNIPER_REPLY_BASIS = (
    "juniper_replied by 1800s: median time from Orion's reply to Juniper's next turn start over 60d "
    "(1785s, n=250, measured 2026-10-16); P(reply<=1800s)=0.50 so the outcome is not saturated; "
    "reply time = when Juniper's next turn started (hub ingress; logged created_at fallback)"
)


def stable_act_id(actor: str, verb: str, key: str) -> str:
    digest = hashlib.sha256(f"{actor}|{verb}|{key}".encode("utf-8")).hexdigest()[:24]
    return f"act_{digest}"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def build_act(
    *,
    actor: str,
    verb: ActVerb,
    key: str,
    signal_id: ActSignal,
    horizon_sec: float,
    basis: str,
    level: ActLevel = "deliberate",
    object_ref: str | None = None,
    test_before: str | None = None,
    parent_act_id: str | None = None,
    goal_ref: str | None = None,
    workspace_ref: str | None = None,
    direction: str = "increase",
) -> ActV1:
    return ActV1(
        act_id=stable_act_id(actor, verb, key),
        parent_act_id=parent_act_id,
        level=level,
        actor=actor,
        verb=verb,
        object_ref=object_ref,
        test_before=test_before,
        expect=ActExpectV1(signal_id=signal_id, direction=direction, horizon_sec=horizon_sec, basis=basis),
        goal_ref=goal_ref,
        workspace_ref=workspace_ref,
    )


def _event(
    kind: str,
    act: ActV1,
    *,
    source_service: str,
    source_component: str,
    correlation_id: str | None,
    session_id: str | None,
    at: datetime | None,
) -> GrammarEventV1:
    ts = at or _now()
    return GrammarEventV1(
        event_id=f"{kind}:{act.act_id}:{uuid4().hex[:8]}",
        event_kind=kind,  # type: ignore[arg-type]
        trace_id=f"act:{act.act_id}",
        correlation_id=correlation_id,
        session_id=session_id,
        emitted_at=_now(),
        observed_at=ts,
        layer="act",
        dimensions=["act", act.verb, act.level],
        act=act,
        provenance=_provenance(source_service, source_component),
    )


def _provenance(source_service: str, source_component: str) -> GrammarProvenanceV1:
    # Literal per producer, so tests/test_grammar_event_producer_catalog.py can see
    # every act producer identity statically. A new act producer adds a branch here
    # and its name to orion:grammar:event producer_services in channels.yaml.
    if source_service == "orion-hub":
        return GrammarProvenanceV1(source_service="orion-hub", source_component=source_component)
    if source_service == "orion-thought":
        return GrammarProvenanceV1(source_service="orion-thought", source_component=source_component)
    if source_service == "orion-durable-runs":
        return GrammarProvenanceV1(source_service="orion-durable-runs", source_component=source_component)
    if source_service == "orion-embodiment":
        return GrammarProvenanceV1(source_service="orion-embodiment", source_component=source_component)
    raise ValueError(f"unregistered act producer {source_service!r}")


def act_opened_event(
    act: ActV1,
    *,
    source_service: str,
    source_component: str,
    correlation_id: str | None = None,
    session_id: str | None = None,
    at: datetime | None = None,
) -> GrammarEventV1:
    return _event("act_opened", act, source_service=source_service, source_component=source_component,
                  correlation_id=correlation_id, session_id=session_id, at=at)


def act_closed_event(
    act: ActV1,
    *,
    produced: bool,
    source_service: str,
    source_component: str,
    note: str = "",
    # V1 only until every reader of orion:grammar:event runs ActCostV2-aware code (consumer-first,
    # orion/schemas/grammar.py ActCostV2). The act ledger fills V2 fields itself on close.
    cost: ActCostV1 | None = None,
    correlation_id: str | None = None,
    session_id: str | None = None,
    at: datetime | None = None,
) -> GrammarEventV1:
    ts = at or _now()
    closed = act.model_copy(update={
        "outcome": ActOutcomeV1(value=1.0 if produced else 0.0, observed_at=ts, source="producer", note=note[:500]),
        "cost": cost,
    })
    return _event("act_closed", closed, source_service=source_service, source_component=source_component,
                  correlation_id=correlation_id, session_id=session_id, at=ts)


def publish_act_event_sync(bus: Any, event: GrammarEventV1, *, source_name: str) -> None:
    """Publish from sync code paths that hold a sync bus client with ``publish``."""
    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

    envelope = BaseEnvelope(
        kind="grammar.event.v1",
        source=ServiceRef(name=source_name),
        payload=event.model_dump(mode="json"),
    )
    bus.publish("orion:grammar:event", envelope)
