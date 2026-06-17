"""Tests for contract-gated investigation probe plan."""

from __future__ import annotations

import sys
from pathlib import Path

CTX_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]


def _ctx_modules():
    for key in list(sys.modules):
        if key == "app" or key.startswith("app."):
            del sys.modules[key]
    for p in (str(REPO_ROOT), str(CTX_ROOT)):
        try:
            sys.path.remove(p)
        except ValueError:
            pass
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(CTX_ROOT))


def test_conceptual_complaint_skips_repo_probe() -> None:
    _ctx_modules()
    from app.investigation_probe_plan import probe_plan_for_request
    from orion.cognition.answer_contract_normalize import heuristic_answer_contract
    from orion.schemas.context_exec import ContextExecRequestV1, context_exec_permissions_for_llm_profile

    text = "hey buddy, why do you give shallow responses like this?"
    req = ContextExecRequestV1(
        text=text,
        mode="investigation_v2",
        permissions=context_exec_permissions_for_llm_profile("agent"),
        answer_contract=heuristic_answer_contract(text),
    )
    plan = probe_plan_for_request(req)
    assert plan.investigation_status == "not_needed"
    assert plan.run_repo is False
    assert plan.run_traces is False
    assert plan.run_recall is True


def test_repo_technical_question_warrants_repo() -> None:
    _ctx_modules()
    from app.investigation_probe_plan import probe_plan_for_request
    from orion.schemas.cognition.answer_contract import AnswerContract
    from orion.schemas.context_exec import ContextExecRequestV1, context_exec_permissions_for_llm_profile

    req = ContextExecRequestV1(
        text="grep services/orion-context-exec for repo impact",
        mode="investigation_v2",
        permissions=context_exec_permissions_for_llm_profile("agent"),
        answer_contract=AnswerContract(
            request_kind="repo_technical",
            asks_for_explanation=True,
            requires_repo_grounding=True,
            preferred_render_style="steps",
        ),
    )
    plan = probe_plan_for_request(req)
    assert plan.investigation_status == "repo_required"
    assert plan.run_repo is True
