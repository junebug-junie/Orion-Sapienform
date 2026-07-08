"""Autonomy goal execute verb (ported from deleted orion-planner-react)."""

from __future__ import annotations

import logging
import os
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from orion.autonomy.goal_actions import (
    build_goal_graph_query_client,
    fetch_goal_by_artifact_id,
    new_goal_task_id,
    update_goal_planned_task_id,
)
from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.schemas.drives import AutonomyGoalPlannedV1

from .settings import settings

logger = logging.getLogger("orion.cortex.exec.autonomy_goal")

AUTONOMY_GOAL_EXECUTE_VERB = "autonomy.goal.execute.v1"
BUS_AUTONOMY_GOAL_PLANNED_OUT = os.getenv(
    "BUS_AUTONOMY_GOAL_PLANNED_OUT",
    "orion:autonomy:goal:planned",
)


class AutonomyGoalExecuteInputV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal_artifact_id: str
    goal_statement: str = ""
    drive_origin: str = "supervisor"


class AutonomyGoalExecuteOutputV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    verb: str = AUTONOMY_GOAL_EXECUTE_VERB
    task_id: str
    goal_artifact_id: str
    proposal_status: str = "executing"


def _persist_goal_executing(*, artifact_id: str, task_id: str) -> bool:
    client = build_goal_graph_query_client()
    if client is None:
        logger.warning(
            "autonomy_goal_execute graph_not_configured artifact_id=%s task_id=%s",
            artifact_id,
            task_id,
        )
        return False
    update_goal_planned_task_id(client, artifact_id, task_id, proposal_status="executing")
    return True


async def _publish_goal_planned_supervisor_event(
    bus: OrionBusAsync,
    *,
    payload: AutonomyGoalExecuteInputV1,
    task_id: str,
    correlation_id: str,
) -> None:
    planned = AutonomyGoalPlannedV1(
        goal_artifact_id=payload.goal_artifact_id,
        goal_statement=payload.goal_statement,
        drive_origin=payload.drive_origin,
        task_id=task_id,
        proposal_status="executing",
        source_verb=AUTONOMY_GOAL_EXECUTE_VERB,
    )
    env = BaseEnvelope(
        kind=planned.kind,
        source=ServiceRef(
            name=settings.service_name,
            version=settings.service_version,
            node=settings.node_name,
        ),
        correlation_id=correlation_id,
        payload=planned.model_dump(mode="json"),
    )
    await bus.publish(BUS_AUTONOMY_GOAL_PLANNED_OUT, env)


async def execute_autonomy_goal_v1(
    payload: AutonomyGoalExecuteInputV1,
    *,
    bus: Optional[OrionBusAsync] = None,
    task_id: str | None = None,
    skip_planned_status_check: bool = False,
    correlation_id: str | None = None,
) -> AutonomyGoalExecuteOutputV1:
    goal_artifact_id = payload.goal_artifact_id.strip()
    if not goal_artifact_id:
        raise ValueError("goal_artifact_id_required")

    if not skip_planned_status_check:
        client = build_goal_graph_query_client()
        if client is None:
            raise ValueError("graph_not_configured")
        fetched = fetch_goal_by_artifact_id(client, goal_artifact_id)
        if fetched is None:
            raise ValueError("goal_not_found")
        goal, _subject = fetched
        if goal.proposal_status != "planned":
            raise ValueError(f"goal_not_planned:{goal.proposal_status}")
        resolved_task_id = goal.planned_task_id or task_id or new_goal_task_id()
        if not payload.goal_statement:
            payload = payload.model_copy(update={"goal_statement": goal.goal_statement or ""})
    else:
        resolved_task_id = task_id or new_goal_task_id()

    corr = correlation_id or str(uuid4())
    graph_persisted = _persist_goal_executing(artifact_id=goal_artifact_id, task_id=resolved_task_id)

    bus_client = bus
    owns_bus = False
    if bus_client is None and settings.orion_bus_enabled:
        bus_client = OrionBusAsync(url=settings.orion_bus_url)
        await bus_client.connect()
        owns_bus = True

    try:
        if bus_client is not None:
            await _publish_goal_planned_supervisor_event(
                bus_client,
                payload=payload,
                task_id=resolved_task_id,
                correlation_id=corr,
            )
    finally:
        if owns_bus and bus_client is not None:
            await bus_client.close()

    logger.info(
        "autonomy_goal_execute artifact_id=%s task_id=%s drive_origin=%s graph_persisted=%s",
        goal_artifact_id,
        resolved_task_id,
        payload.drive_origin,
        str(graph_persisted).lower(),
    )

    return AutonomyGoalExecuteOutputV1(
        task_id=resolved_task_id,
        goal_artifact_id=goal_artifact_id,
    )
