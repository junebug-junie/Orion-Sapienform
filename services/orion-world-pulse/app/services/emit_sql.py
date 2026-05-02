from __future__ import annotations

from typing import Any

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.world_pulse import (
    ArticleClusterV1,
    ArticleRecordV1,
    ClaimRecordV1,
    DailyWorldPulseV1,
    EntityRecordV1,
    EventRecordV1,
    SituationChangeV1,
    TopicSituationBriefV1,
    WorldContextCapsuleV1,
    WorldLearningDeltaV1,
    WorldPulseRunResultV1,
)


def build_sql_envelopes(
    *,
    source_ref: ServiceRef,
    run_result: WorldPulseRunResultV1,
    claims: list[ClaimRecordV1],
    events: list[EventRecordV1],
    entities: list[EntityRecordV1],
    briefs: list[TopicSituationBriefV1],
    changes: list[SituationChangeV1],
    learning: list[WorldLearningDeltaV1],
) -> list[tuple[str, BaseEnvelope]]:
    envs: list[tuple[str, BaseEnvelope]] = []
    envs.append(("orion:world_pulse:run:result", BaseEnvelope(kind="world.pulse.run.result.v1", source=source_ref, payload=run_result.model_dump(mode="json"))))
    if run_result.digest:
        digest: DailyWorldPulseV1 = run_result.digest
        envs.append(("orion:world_pulse:digest:created", BaseEnvelope(kind="world.pulse.digest.created.v1", source=source_ref, payload=digest.model_dump(mode="json"))))
        for item in digest.items:
            envs.append(
                (
                    "orion:world_pulse:digest:item",
                    BaseEnvelope(
                        kind="world.pulse.digest.item.v1",
                        source=source_ref,
                        payload=item.model_dump(mode="json"),
                    ),
                )
            )
        for reading in digest.things_worth_reading:
            envs.append(
                (
                    "orion:world_pulse:worth:reading",
                    BaseEnvelope(
                        kind="world.pulse.worth.reading.v1",
                        source=source_ref,
                        payload=reading.model_dump(mode="json"),
                    ),
                )
            )
        for watching in digest.things_worth_watching:
            envs.append(
                (
                    "orion:world_pulse:worth:watching",
                    BaseEnvelope(
                        kind="world.pulse.worth.watching.v1",
                        source=source_ref,
                        payload=watching.model_dump(mode="json"),
                    ),
                )
            )
    if run_result.capsule:
        capsule: WorldContextCapsuleV1 = run_result.capsule
        envs.append(("orion:world_context:daily_capsule", BaseEnvelope(kind="world.context.daily.capsule.v1", source=source_ref, payload=capsule.model_dump(mode="json"))))
    for claim in claims:
        envs.append(("orion:world_pulse:claim:emit", BaseEnvelope(kind="world.pulse.claim.emit.v1", source=source_ref, payload=claim.model_dump(mode="json"))))
    for event in events:
        envs.append(("orion:world_pulse:event:emit", BaseEnvelope(kind="world.pulse.event.emit.v1", source=source_ref, payload=event.model_dump(mode="json"))))
    for entity in entities:
        envs.append(("orion:world_pulse:entity:emit", BaseEnvelope(kind="world.pulse.entity.emit.v1", source=source_ref, payload=entity.model_dump(mode="json"))))
    for brief in briefs:
        envs.append(("orion:world_pulse:situation:brief:upsert", BaseEnvelope(kind="world.pulse.situation.brief.upsert.v1", source=source_ref, payload=brief.model_dump(mode="json"))))
    for change in changes:
        envs.append(("orion:world_pulse:situation:change:emit", BaseEnvelope(kind="world.pulse.situation.change.emit.v1", source=source_ref, payload=change.model_dump(mode="json"))))
    for delta in learning:
        envs.append(("orion:world_pulse:learning:emit", BaseEnvelope(kind="world.pulse.learning.emit.v1", source=source_ref, payload=delta.model_dump(mode="json"))))
    for article in (run_result.publish_status or {}).get("articles", []):
        article_model = article if isinstance(article, ArticleRecordV1) else ArticleRecordV1.model_validate(article)
        envs.append(
            (
                "orion:world_pulse:article:emit",
                BaseEnvelope(kind="world.pulse.article.emit.v1", source=source_ref, payload=article_model.model_dump(mode="json")),
            )
        )
    for cluster in (run_result.publish_status or {}).get("clusters", []):
        cluster_model = cluster if isinstance(cluster, ArticleClusterV1) else ArticleClusterV1.model_validate(cluster)
        envs.append(
            (
                "orion:world_pulse:cluster:emit",
                BaseEnvelope(kind="world.pulse.cluster.emit.v1", source=source_ref, payload=cluster_model.model_dump(mode="json")),
            )
        )
    envs.append(
        (
            "orion:world_pulse:publish:status",
            BaseEnvelope(
                kind="world.pulse.publish.status.v1",
                source=source_ref,
                payload={
                    "status_id": f"publish:{run_result.run.run_id}",
                    "run_id": run_result.run.run_id,
                    "channel": "sql",
                    "state": run_result.run.sql_emit_status,
                    "detail": "world pulse sql emission",
                    "publish_status": run_result.publish_status,
                },
            ),
        )
    )
    return envs
