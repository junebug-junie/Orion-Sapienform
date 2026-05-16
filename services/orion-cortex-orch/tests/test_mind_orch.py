from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from uuid import uuid4

_guard = Path(__file__).resolve().parent / "_orch_import_guard.py"
_spec = importlib.util.spec_from_file_location("_orch_guard_boot", _guard)
assert _spec and _spec.loader
_guard_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_guard_mod)

ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


def _orch_prep() -> None:
    _guard_mod.ensure_orion_cortex_orch_app()


from orion.mind.v1 import MindHandoffBriefV1, MindRunResultV1, MindStancePatchV1, MindStanceTrajectoryV1, MindProvenanceV1
from orion.schemas.cortex.contracts import CortexClientContext, CortexClientRequest, LLMMessage
from orion.schemas.cortex.schemas import ExecutionPlan, ExecutionStep, PlanExecutionRequest


_VALID_STANCE = {
    "conversation_frame": "technical",
    "user_intent": "u",
    "self_relevance": "s",
    "juniper_relevance": "j",
    "answer_strategy": "a",
    "stance_summary": "st",
}


def _plan_request() -> PlanExecutionRequest:
    return PlanExecutionRequest(
        plan=ExecutionPlan(
            verb_name="chat_general",
            steps=[ExecutionStep(verb_name="chat_general", step_name="noop", order=0, services=[])],
        ),
        context={"metadata": {}},
    )


def test_meaningful_mind_synthesis_may_skip_stance_synthesis() -> None:
    _orch_prep()
    from app.mind_runtime import merge_mind_brief_into_plan_metadata

    pr = _plan_request()
    result = MindRunResultV1(
        mind_run_id=uuid4(),
        ok=True,
        mind_quality="meaningful_synthesis",
        brief=MindHandoffBriefV1(
            mind_quality="meaningful_synthesis",
            machine_contract={"mind.route_kind": "brain"},
            stance_payload=dict(_VALID_STANCE),
        ),
    )

    merge_mind_brief_into_plan_metadata(pr, result)
    meta = pr.context["metadata"]
    assert meta.get("mind_skip_stance_synthesis") is True
    assert meta.get("mind_quality") == "meaningful_synthesis"
    assert meta.get("mind.route_kind") == "brain"


def test_fallback_contract_only_mind_never_skips_stance_synthesis() -> None:
    _orch_prep()
    from app.mind_runtime import merge_mind_brief_into_plan_metadata

    pr = _plan_request()
    result = MindRunResultV1(
        mind_run_id=uuid4(),
        ok=True,
        mind_quality="fallback_contract_only",
        trajectory=MindStanceTrajectoryV1(
            patches=[
                MindStancePatchV1(
                    loop_index=0,
                    structured=dict(_VALID_STANCE),
                    provenance=MindProvenanceV1(model_id="deterministic"),
                )
            ],
            merged_stance_brief=dict(_VALID_STANCE),
        ),
        brief=MindHandoffBriefV1(
            mind_quality="fallback_contract_only",
            summary_one_paragraph="Fallback contract only — no meaningful Mind synthesis produced.",
            machine_contract={"mind.route_kind": "brain"},
            stance_payload=dict(_VALID_STANCE),
        ),
    )

    merge_mind_brief_into_plan_metadata(pr, result)
    meta = pr.context["metadata"]
    assert meta.get("mind_skip_stance_synthesis") is False
    assert meta.get("mind_contract_only") is True
    assert meta.get("mind_quality") == "fallback_contract_only"


def test_legacy_deterministic_summary_is_treated_as_contract_only() -> None:
    _orch_prep()
    from app.mind_runtime import merge_mind_brief_into_plan_metadata

    pr = _plan_request()
    result = MindRunResultV1(
        mind_run_id=uuid4(),
        ok=True,
        brief=MindHandoffBriefV1(
            summary_one_paragraph="Deterministic mind run (v1).",
            machine_contract={"mind.route_kind": "brain"},
            stance_payload=dict(_VALID_STANCE),
        ),
    )

    merge_mind_brief_into_plan_metadata(pr, result)
    meta = pr.context["metadata"]
    assert meta.get("mind_skip_stance_synthesis") is False
    assert meta.get("mind_contract_only") is True
    assert meta.get("mind_quality") == "fallback_contract_only"


def test_meaningful_mind_with_invalid_payload_does_not_skip_stance_synthesis() -> None:
    _orch_prep()
    from app.mind_runtime import merge_mind_brief_into_plan_metadata

    pr = _plan_request()
    result = MindRunResultV1(
        mind_run_id=uuid4(),
        ok=True,
        mind_quality="meaningful_synthesis",
        brief=MindHandoffBriefV1(
            mind_quality="meaningful_synthesis",
            machine_contract={"mind.route_kind": "brain"},
            stance_payload={"conversation_frame": "___invalid___"},
        ),
    )

    merge_mind_brief_into_plan_metadata(pr, result)
    meta = pr.context["metadata"]
    assert meta.get("mind_stance_payload_invalid") is True
    assert meta.get("mind_skip_stance_synthesis") is False


def test_build_mind_run_request_merges_substrate_telemetry_facet() -> None:
    _orch_prep()
    from app.mind_runtime import build_mind_run_request

    cr = CortexClientRequest(
        verb="chat_general",
        mode="brain",
        context=CortexClientContext(
            messages=[LLMMessage(role="user", content="hi")],
            session_id="s",
            trace_id="t",
            user_message="hi",
            metadata={"mind_enabled": True},
        ),
    )
    pr = _plan_request()
    req = build_mind_run_request(
        cr,
        pr,
        "550e8400-e29b-41d4-a716-446655440000",
        substrate_telemetry_facet={"status": "absent"},
    )
    facets = (req.snapshot_inputs or {}).get("facets") or {}
    assert facets.get("substrate_telemetry") == {"status": "absent"}
