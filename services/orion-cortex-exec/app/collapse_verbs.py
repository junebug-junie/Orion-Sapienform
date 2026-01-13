from __future__ import annotations

from typing import Any, Dict, List, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field, RootModel

from orion.core.bus.bus_schemas import BaseEnvelope
from orion.core.verbs.base import BaseVerb, VerbContext
from orion.core.verbs.models import VerbEffectV1
from orion.core.verbs.registry import verb
from orion.collapse import create_entry_from_v2, enrich_entry, score_causal_density


class CollapseMirrorLogPayload(RootModel[Dict[str, Any]]):
    pass


class CollapseMirrorLogResult(BaseModel):
    event_id: str
    timestamp: str
    snapshot_kind: str
    causal_density: Dict[str, Any]


class CollapseMirrorEventRequest(BaseModel):
    event_id: str


class CollapseMirrorEventResult(BaseModel):
    event_id: str
    snapshot_kind: str
    causal_density: Dict[str, Any]
    is_causally_dense: bool


@verb("orion.collapse.log")
class LogCollapseMirrorVerb(BaseVerb[CollapseMirrorLogPayload, CollapseMirrorLogResult]):
    input_model = CollapseMirrorLogPayload
    output_model = CollapseMirrorLogResult

    async def execute(
        self,
        ctx: VerbContext,
        payload: CollapseMirrorLogPayload,
    ) -> Tuple[CollapseMirrorLogResult, List[VerbEffectV1]]:
        bus = ctx.meta.get("bus")
        source = ctx.meta.get("source")

        entry = create_entry_from_v2(
            payload.root,
            source_service=getattr(source, "name", None),
            source_node=getattr(source, "node", None),
        )

        if bus is not None:
            envelope = BaseEnvelope(
                kind="collapse.mirror.intake",
                source=source,
                correlation_id=str(ctx.request_id or uuid4()),
                payload=entry.model_dump(mode="json"),
            )
            await bus.publish("orion:collapse:intake", envelope)

        effects = [
            VerbEffectV1(kind="sql", payload=entry.model_dump(mode="json")),
            VerbEffectV1(kind="rdf", payload=entry.model_dump(mode="json")),
            VerbEffectV1(kind="vector", payload=entry.model_dump(mode="json")),
        ]

        return (
            CollapseMirrorLogResult(
                event_id=entry.event_id,
                timestamp=entry.timestamp,
                snapshot_kind=entry.snapshot_kind,
                causal_density=entry.causal_density.model_dump(mode="json"),
            ),
            effects,
        )


@verb("orion.collapse.enrich")
class EnrichCollapseMirrorVerb(BaseVerb[CollapseMirrorEventRequest, CollapseMirrorEventResult]):
    input_model = CollapseMirrorEventRequest
    output_model = CollapseMirrorEventResult

    async def execute(
        self,
        ctx: VerbContext,
        payload: CollapseMirrorEventRequest,
    ) -> Tuple[CollapseMirrorEventResult, List[VerbEffectV1]]:
        bus = ctx.meta.get("bus")
        source = ctx.meta.get("source")

        entry = enrich_entry(payload.event_id)

        effects = [
            VerbEffectV1(
                kind="telemetry",
                payload={"event": "collapse_mirror.enriched", "event_id": entry.event_id},
            )
        ]

        if bus is not None:
            envelope = BaseEnvelope(
                kind="collapse.mirror.event",
                source=source,
                correlation_id=str(ctx.request_id or uuid4()),
                payload={"event": "collapse_mirror.enriched", "event_id": entry.event_id},
            )
            await bus.publish("orion:collapse:events", envelope)

        return (
            CollapseMirrorEventResult(
                event_id=entry.event_id,
                snapshot_kind=entry.snapshot_kind,
                causal_density=entry.causal_density.model_dump(mode="json"),
                is_causally_dense=entry.is_causally_dense,
            ),
            effects,
        )


@verb("orion.collapse.score")
class ScoreCausalDensityVerb(BaseVerb[CollapseMirrorEventRequest, CollapseMirrorEventResult]):
    input_model = CollapseMirrorEventRequest
    output_model = CollapseMirrorEventResult

    async def execute(
        self,
        ctx: VerbContext,
        payload: CollapseMirrorEventRequest,
    ) -> Tuple[CollapseMirrorEventResult, List[VerbEffectV1]]:
        bus = ctx.meta.get("bus")
        source = ctx.meta.get("source")

        entry = score_causal_density(payload.event_id)

        effects: List[VerbEffectV1] = []
        if entry.is_causally_dense:
            effects.append(
                VerbEffectV1(
                    kind="spark",
                    payload={
                        "event": "collapse_mirror.causally_dense",
                        "event_id": entry.event_id,
                        "score": entry.causal_density.score,
                        "label": entry.causal_density.label,
                    },
                )
            )

        if bus is not None:
            envelope = BaseEnvelope(
                kind="collapse.mirror.event",
                source=source,
                correlation_id=str(ctx.request_id or uuid4()),
                payload={
                    "event": "collapse_mirror.scored",
                    "event_id": entry.event_id,
                    "score": entry.causal_density.score,
                    "label": entry.causal_density.label,
                    "is_causally_dense": entry.is_causally_dense,
                },
            )
            await bus.publish("orion:collapse:events", envelope)

        return (
            CollapseMirrorEventResult(
                event_id=entry.event_id,
                snapshot_kind=entry.snapshot_kind,
                causal_density=entry.causal_density.model_dump(mode="json"),
                is_causally_dense=entry.is_causally_dense,
            ),
            effects,
        )
