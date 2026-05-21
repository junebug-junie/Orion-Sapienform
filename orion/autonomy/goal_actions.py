from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from orion.autonomy.models import AutonomyGoalHeadlineV1
from orion.autonomy.repository import AUTONOMY_GOALS_GRAPH, _escape_sparql, _literal
from orion.core.schemas.reasoning import ClaimV1, ReasoningProvenanceV1
from orion.core.schemas.reasoning_io import ReasoningWriteContextV1, ReasoningWriteRequestV1
from orion.core.schemas.reasoning_policy import PromotionEvaluationRequestV1, PromotionEvaluationResultV1
from orion.reasoning.promotion import PromotionEngine
from orion.reasoning.repository import InMemoryReasoningRepository
from orion.spark.concept_induction.graph_query import GraphQueryClient, GraphQueryConfig, GraphQueryError

logger = logging.getLogger("orion.autonomy.goal_actions")

GoalActionKind = Literal["promote", "dismiss", "complete"]


class GoalActionError(Exception):
    def __init__(self, code: str, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True)
class GoalActionResult:
    artifact_id: str
    action: GoalActionKind
    proposal_status: str
    reasoning_outcome: str | None = None
    reasoning_claim_id: str | None = None
    hitl_satisfied: bool = False
    completed_at: str | None = None
    planned_task_id: str | None = None


def _autonomy_goal_execution_enabled() -> bool:
    return str(os.getenv("AUTONOMY_GOAL_EXECUTION_ENABLED", "false")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def new_goal_task_id() -> str:
    return f"goal-task-{uuid.uuid4()}"


def update_goal_planned_task_id(
    client: GraphQueryClient,
    artifact_id: str,
    task_id: str,
    *,
    proposal_status: str = "planned",
) -> None:
    safe_id = _escape_sparql(artifact_id.strip())
    safe_task = _escape_sparql(task_id.strip())
    safe_status = _escape_sparql(proposal_status.strip())
    update = f"""
PREFIX orion: <http://conjourney.net/orion#>
DELETE {{ ?a orion:plannedTaskId ?old_task . ?a orion:proposalStatus ?old_status . }}
INSERT {{ ?a orion:plannedTaskId "{safe_task}" . ?a orion:proposalStatus "{safe_status}" . }}
WHERE {{
  GRAPH <{AUTONOMY_GOALS_GRAPH}> {{
    ?a orion:artifactId "{safe_id}" .
    OPTIONAL {{ ?a orion:plannedTaskId ?old_task . }}
    OPTIONAL {{ ?a orion:proposalStatus ?old_status . }}
  }}
}}
""".strip()
    client.update(update)


def plan_promoted_goal(
    *,
    goal: AutonomyGoalHeadlineV1,
    graph_client: GraphQueryClient,
) -> str:
    if not _autonomy_goal_execution_enabled():
        raise GoalActionError("autonomy_goal_execution_disabled", "autonomy goal execution is disabled", status_code=503)
    if goal.proposal_status != "planned":
        raise GoalActionError(
            "goal_not_planned",
            f"goal must be planned before task allocation: {goal.proposal_status}",
            status_code=409,
        )
    task_id = new_goal_task_id()
    update_goal_planned_task_id(graph_client, goal.artifact_id, task_id, proposal_status="planned")
    return task_id


def build_goal_graph_query_client() -> GraphQueryClient | None:
    from orion.graph.backend_config import resolve_autonomy_read_query_url, resolve_graph_update_url, resolve_rdf_store_auth

    query_url, _src = resolve_autonomy_read_query_url()
    if not query_url:
        base = os.getenv("GRAPHDB_URL", "").strip()
        if not base:
            return None
        repo = (os.getenv("GRAPHDB_REPO") or "collapse").strip()
        query_url = base if "/repositories/" in base else f"{base.rstrip('/')}/repositories/{repo}"
        update_url = query_url
        user = (os.getenv("GRAPHDB_USER") or os.getenv("CONCEPT_PROFILE_GRAPHDB_USER") or "").strip() or None
        password = (os.getenv("GRAPHDB_PASS") or os.getenv("CONCEPT_PROFILE_GRAPHDB_PASS") or "").strip() or None
    else:
        update_url = resolve_graph_update_url() or query_url
        user, password = resolve_rdf_store_auth()

    return GraphQueryClient(
        GraphQueryConfig(
            endpoint=query_url,
            update_endpoint=update_url,
            graph_uri=AUTONOMY_GOALS_GRAPH,
            timeout_sec=float(os.getenv("AUTONOMY_GRAPH_TIMEOUT_SEC", "30")),
            user=user,
            password=password,
        )
    )


def fetch_goal_by_artifact_id(client: GraphQueryClient, artifact_id: str) -> tuple[AutonomyGoalHeadlineV1, str] | None:
    safe_id = _escape_sparql(artifact_id.strip())
    sparql = f"""
PREFIX orion: <http://conjourney.net/orion#>
SELECT ?artifact_id ?goal_statement ?drive_origin ?priority ?cooldown_until ?proposal_signature ?created_at ?proposal_status ?planned_task_id ?completed_at ?subject_key
WHERE {{
  GRAPH <{AUTONOMY_GOALS_GRAPH}> {{
    ?artifact a orion:ProposedGoal ;
      orion:artifactId ?artifact_id ;
      orion:goalStatement ?goal_statement ;
      orion:driveOrigin ?drive_origin ;
      orion:proposalPriority ?priority ;
      orion:proposalSignature ?proposal_signature .
    OPTIONAL {{ ?artifact orion:cooldownUntil ?cooldown_until . }}
    OPTIONAL {{ ?artifact orion:proposalStatus ?proposal_status . }}
    OPTIONAL {{ ?artifact orion:plannedTaskId ?planned_task_id . }}
    OPTIONAL {{ ?artifact orion:completedAt ?completed_at . }}
    OPTIONAL {{ ?artifact orion:timestamp ?created_at . }}
    OPTIONAL {{ ?artifact orion:subjectKey ?subject_key . }}
    FILTER(?artifact_id = "{safe_id}")
  }}
}}
LIMIT 1
""".strip()
    rows = client.select(sparql)
    if not rows:
        return None
    row = rows[0]
    try:
        goal = AutonomyGoalHeadlineV1(
            artifact_id=_literal(row, "artifact_id") or artifact_id,
            goal_statement=_literal(row, "goal_statement") or "",
            drive_origin=_literal(row, "drive_origin") or "",
            priority=float(_literal(row, "priority") or 0.0),
            cooldown_until=_literal(row, "cooldown_until"),
            proposal_signature=_literal(row, "proposal_signature") or "",
            proposal_status=_literal(row, "proposal_status") or "proposed",
            planned_task_id=_literal(row, "planned_task_id"),
            completed_at=_literal(row, "completed_at"),
        )
    except Exception as exc:
        raise GoalActionError("goal_row_invalid", str(exc), status_code=500) from exc
    subject = _literal(row, "subject_key") or "orion"
    return goal, subject


def _goal_to_reasoning_claim(*, goal: AutonomyGoalHeadlineV1, subject: str, observed_at: datetime | None = None) -> ClaimV1:
    when = observed_at or datetime.now(timezone.utc)
    return ClaimV1(
        artifact_id=f"goal-reasoning-{goal.artifact_id}",
        anchor_scope=subject,  # type: ignore[arg-type]
        subject_ref=f"goal:{goal.proposal_signature}",
        status="proposed",
        authority="human_verified",
        confidence=max(0.2, min(goal.priority, 1.0)),
        salience=goal.priority,
        novelty=0.4,
        risk_tier="medium",
        observed_at=when,
        provenance=ReasoningProvenanceV1(
            evidence_refs=[goal.artifact_id],
            source_channel="orion:autonomy",
            source_kind="AutonomyGoalHeadlineV1",
            producer="hub_goal_action",
        ),
        claim_text=goal.goal_statement,
        claim_kind="goal_proposal_headline",
        qualifiers={
            "drive_origin": goal.drive_origin,
            "priority": goal.priority,
            "proposal_status": goal.proposal_status,
            "planned_task_id": goal.planned_task_id,
            "completed_at": goal.completed_at.isoformat() if goal.completed_at else None,
            "cooldown_until": goal.cooldown_until.isoformat() if goal.cooldown_until else None,
        },
    )


def apply_operator_goal_reasoning_promotion(
    *,
    goal: AutonomyGoalHeadlineV1,
    subject: str,
    operator: str,
    reasoning_repo: InMemoryReasoningRepository,
) -> PromotionEvaluationResultV1:
    claim = _goal_to_reasoning_claim(goal=goal, subject=subject)
    existing = reasoning_repo.get_by_id(claim.artifact_id)
    if existing is None:
        reasoning_repo.write_artifacts(
            ReasoningWriteRequestV1(
                context=ReasoningWriteContextV1(
                    source_family="manual",
                    source_kind="hub_goal_action",
                    source_channel="orion:hub",
                    producer="hub_goal_action",
                ),
                artifacts=[claim],
            )
        )

    engine = PromotionEngine(reasoning_repo)
    artifact = reasoning_repo.get_by_id(claim.artifact_id)
    if artifact is None:
        raise GoalActionError("reasoning_claim_missing", "reasoning claim could not be stored", status_code=500)

    if artifact.status == "proposed":
        provisional = engine.evaluate(
            PromotionEvaluationRequestV1(
                artifact_ids=[claim.artifact_id],
                target_status="provisional",
                actor=operator,
            )
        )
        if provisional.items[0].outcome not in {"promoted", "no_change"}:
            raise GoalActionError(
                "reasoning_promotion_blocked",
                "; ".join(provisional.items[0].reasons) or "provisional promotion blocked",
                status_code=409,
            )

    canonical = engine.evaluate(
        PromotionEvaluationRequestV1(
            artifact_ids=[claim.artifact_id],
            target_status="canonical",
            actor=operator,
        )
    )
    item = canonical.items[0]
    if item.outcome == "escalated_hitl" and "autonomy_goal_requires_hitl" in item.reasons:
        reasoning_repo.update_status(claim.artifact_id, "canonical")
        return canonical
    if item.outcome == "promoted":
        return canonical
    raise GoalActionError(
        "reasoning_promotion_blocked",
        "; ".join(item.reasons) or "canonical promotion blocked",
        status_code=409,
    )


def update_goal_proposal_status(
    client: GraphQueryClient,
    artifact_id: str,
    proposal_status: str,
    *,
    completed_at: datetime | None = None,
) -> None:
    safe_id = _escape_sparql(artifact_id.strip())
    completed_clause = ""
    if completed_at is not None:
        completed_iso = completed_at.astimezone(timezone.utc).isoformat()
        completed_clause = f"""
DELETE {{ ?a orion:completedAt ?old_completed . }}
INSERT {{ ?a orion:completedAt "{_escape_sparql(completed_iso)}" . }}
"""
    update = f"""
PREFIX orion: <http://conjourney.net/orion#>
DELETE {{ ?a orion:proposalStatus ?old . }}
INSERT {{ ?a orion:proposalStatus "{_escape_sparql(proposal_status)}" . }}
{completed_clause}
WHERE {{
  GRAPH <{AUTONOMY_GOALS_GRAPH}> {{
    ?a orion:artifactId "{safe_id}" .
    OPTIONAL {{ ?a orion:proposalStatus ?old . }}
    {"OPTIONAL { ?a orion:completedAt ?old_completed . }" if completed_at is not None else ""}
  }}
}}
""".strip()
    client.update(update)


def promote_goal(
    *,
    artifact_id: str,
    operator: str,
    graph_client: GraphQueryClient | None = None,
    reasoning_repo: InMemoryReasoningRepository | None = None,
) -> GoalActionResult:
    client = graph_client or build_goal_graph_query_client()
    if client is None:
        raise GoalActionError("graph_not_configured", "autonomy goals graph is not configured", status_code=503)

    fetched = fetch_goal_by_artifact_id(client, artifact_id)
    if fetched is None:
        raise GoalActionError("goal_not_found", f"goal artifact not found: {artifact_id}", status_code=404)
    goal, subject = fetched
    if goal.proposal_status in {"archived", "completed", "superseded"}:
        raise GoalActionError(
            "goal_terminal_status",
            f"goal is in terminal status: {goal.proposal_status}",
            status_code=409,
        )

    repo = reasoning_repo or InMemoryReasoningRepository()
    promotion = apply_operator_goal_reasoning_promotion(goal=goal, subject=subject, operator=operator, reasoning_repo=repo)
    item = promotion.items[0]
    hitl_satisfied = item.outcome == "escalated_hitl" or item.human_review_required

    update_goal_proposal_status(client, artifact_id, "planned")
    planned_goal = AutonomyGoalHeadlineV1(
        artifact_id=goal.artifact_id,
        goal_statement=goal.goal_statement,
        drive_origin=goal.drive_origin,
        priority=goal.priority,
        cooldown_until=goal.cooldown_until,
        proposal_signature=goal.proposal_signature,
        proposal_status="planned",
        planned_task_id=goal.planned_task_id,
        completed_at=goal.completed_at,
    )
    task_id = plan_promoted_goal(goal=planned_goal, graph_client=client)
    logger.info(
        "autonomy_goal_promote artifact_id=%s operator=%s proposal_status=planned task_id=%s",
        artifact_id,
        operator,
        task_id,
    )
    return GoalActionResult(
        artifact_id=artifact_id,
        action="promote",
        proposal_status="planned",
        reasoning_outcome=item.outcome,
        reasoning_claim_id=f"goal-reasoning-{artifact_id}",
        hitl_satisfied=hitl_satisfied,
        planned_task_id=task_id,
    )


def dismiss_goal(
    *,
    artifact_id: str,
    operator: str,
    graph_client: GraphQueryClient | None = None,
) -> GoalActionResult:
    del operator
    client = graph_client or build_goal_graph_query_client()
    if client is None:
        raise GoalActionError("graph_not_configured", "autonomy goals graph is not configured", status_code=503)

    fetched = fetch_goal_by_artifact_id(client, artifact_id)
    if fetched is None:
        raise GoalActionError("goal_not_found", f"goal artifact not found: {artifact_id}", status_code=404)

    update_goal_proposal_status(client, artifact_id, "archived")
    logger.info("autonomy_goal_dismiss artifact_id=%s proposal_status=archived", artifact_id)
    return GoalActionResult(artifact_id=artifact_id, action="dismiss", proposal_status="archived")


def complete_goal(
    *,
    artifact_id: str,
    operator: str,
    graph_client: GraphQueryClient | None = None,
    completed_at: datetime | None = None,
) -> GoalActionResult:
    del operator
    client = graph_client or build_goal_graph_query_client()
    if client is None:
        raise GoalActionError("graph_not_configured", "autonomy goals graph is not configured", status_code=503)

    fetched = fetch_goal_by_artifact_id(client, artifact_id)
    if fetched is None:
        raise GoalActionError("goal_not_found", f"goal artifact not found: {artifact_id}", status_code=404)

    when = completed_at or datetime.now(timezone.utc)
    update_goal_proposal_status(client, artifact_id, "completed", completed_at=when)
    logger.info("autonomy_goal_complete artifact_id=%s proposal_status=completed", artifact_id)
    return GoalActionResult(
        artifact_id=artifact_id,
        action="complete",
        proposal_status="completed",
        completed_at=when.isoformat(),
    )


def execute_goal_action(
    action: GoalActionKind,
    *,
    artifact_id: str,
    operator: str,
    graph_client: GraphQueryClient | None = None,
    reasoning_repo: InMemoryReasoningRepository | None = None,
) -> GoalActionResult:
    if action == "promote":
        return promote_goal(
            artifact_id=artifact_id,
            operator=operator,
            graph_client=graph_client,
            reasoning_repo=reasoning_repo,
        )
    if action == "dismiss":
        return dismiss_goal(artifact_id=artifact_id, operator=operator, graph_client=graph_client)
    if action == "complete":
        return complete_goal(artifact_id=artifact_id, operator=operator, graph_client=graph_client)
    raise GoalActionError("unsupported_action", f"unsupported goal action: {action}", status_code=400)
