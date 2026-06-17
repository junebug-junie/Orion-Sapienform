"""Contract-gated probe plan for investigation_v2 (permissions = ceiling, contract = warrant)."""

from __future__ import annotations

from dataclasses import dataclass

from orion.cognition.answer_contract_normalize import (
    heuristic_answer_contract,
    investigation_state_for_contract,
)
from orion.schemas.cognition.answer_contract import AnswerContract
from orion.schemas.context_exec import ContextExecRequestV1


@dataclass(frozen=True)
class InvestigationProbePlan:
    investigation_status: str
    run_repo: bool
    run_traces: bool
    run_recall: bool
    run_memory: bool
    run_runtime: bool
    run_health: bool
    skip_repo_reason: str | None = None
    skip_traces_reason: str | None = None
    skip_runtime_reason: str | None = None


def _contract_for_request(request: ContextExecRequestV1) -> AnswerContract:
    if request.answer_contract is not None:
        return request.answer_contract
    return heuristic_answer_contract(request.text or "")


def probe_plan_for_request(request: ContextExecRequestV1) -> InvestigationProbePlan:
    """Derive which evidence probes are warranted for this turn."""
    contract = _contract_for_request(request)
    inv = investigation_state_for_contract(contract)
    status = str(inv.get("status") or "not_needed")
    perms = request.permissions

    run_repo = bool(perms.read_repo and status in {"repo_required", "mixed_required"})
    run_traces = bool(perms.read_redis_traces and status in {"runtime_required", "mixed_required"})
    run_runtime = bool(perms.read_runtime_logs and status in {"runtime_required", "mixed_required"})
    run_recall = bool(
        perms.read_recall
        and (
            status == "not_needed"
            or contract.requires_user_artifact_grounding
            or status in {"repo_required", "runtime_required", "mixed_required"}
        )
    )
    run_memory = bool(perms.read_memory)
    run_health = True

    skip_repo = None if run_repo else "not warranted by answer contract"
    skip_traces = None if run_traces else "not warranted by answer contract"
    skip_runtime = None if run_runtime else "not warranted by answer contract"

    return InvestigationProbePlan(
        investigation_status=status,
        run_repo=run_repo,
        run_traces=run_traces,
        run_recall=run_recall,
        run_memory=run_memory,
        run_runtime=run_runtime,
        run_health=run_health,
        skip_repo_reason=skip_repo,
        skip_traces_reason=skip_traces,
        skip_runtime_reason=skip_runtime,
    )
