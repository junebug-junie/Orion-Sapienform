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


from orion.mind.v1 import (
    MindHandoffBriefV1,
    MindProvenanceV1,
    MindRunResultV1,
    MindShadowSynthesisV1,
    MindStancePatchV1,
    MindStanceTrajectoryV1,
)
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
        context={"metadata": {"orion_state": {"ok": True, "status": "present"}}},
    )


def _client_request(metadata: dict | None = None) -> CortexClientRequest:
    return CortexClientRequest(
        verb="chat_general",
        mode="brain",
        context=CortexClientContext(
            messages=[LLMMessage(role="user", content="hi")],
            session_id="s",
            trace_id="t",
            user_message="hi",
            metadata=metadata or {"mind_enabled": True},
        ),
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
            mind_authorized_for_stance_skip=True,
            machine_contract={"mind.route_kind": "brain"},
            stance_payload=dict(_VALID_STANCE),
        ),
    )

    merge_mind_brief_into_plan_metadata(pr, result)
    meta = pr.context["metadata"]
    assert meta.get("mind_skip_stance_synthesis") is True
    assert meta.get("mind_quality") == "meaningful_synthesis"
    assert meta.get("mind.route_kind") == "brain"


def test_meaningful_mind_synthesis_without_authorization_does_not_skip_stance_synthesis() -> None:
    _orch_prep()
    from app.mind_runtime import merge_mind_brief_into_plan_metadata

    pr = _plan_request()
    result = MindRunResultV1(
        mind_run_id=uuid4(),
        ok=True,
        mind_quality="meaningful_synthesis",
        brief=MindHandoffBriefV1(
            mind_quality="meaningful_synthesis",
            mind_authorized_for_stance_skip=False,
            machine_contract={"mind.route_kind": "brain"},
            stance_payload=dict(_VALID_STANCE),
        ),
    )

    merge_mind_brief_into_plan_metadata(pr, result)
    meta = pr.context["metadata"]
    assert meta.get("mind_skip_stance_synthesis") is False


def test_shadow_synthesis_is_preserved_but_never_skips_stance_synthesis() -> None:
    _orch_prep()
    from app.mind_runtime import merge_mind_brief_into_plan_metadata

    pr = _plan_request()
    shadow = MindShadowSynthesisV1(
        present=True,
        authorized_for_stance_skip=False,
        stance_candidate=dict(_VALID_STANCE),
        attention_focus=["projection item"],
        projection_refs_used=["node-1"],
        confidence=0.5,
        rationale="test shadow",
    )
    result = MindRunResultV1(
        mind_run_id=uuid4(),
        ok=True,
        mind_quality="shadow_synthesis",
        brief=MindHandoffBriefV1(
            mind_quality="shadow_synthesis",
            shadow_synthesis=shadow,
            mind_authorized_for_stance_skip=False,
            machine_contract={"mind.route_kind": "brain", "mind.shadow_synthesis_present": True},
            stance_payload=dict(_VALID_STANCE),
        ),
    )

    merge_mind_brief_into_plan_metadata(pr, result)
    meta = pr.context["metadata"]
    assert meta.get("mind_quality") == "shadow_synthesis"
    assert meta.get("mind_shadow_synthesis_present") is True
    assert meta.get("mind_shadow_synthesis", {}).get("schema_version") == "mind.shadow_synthesis.v1"
    assert meta.get("mind_shadow_synthesis", {}).get("projection_refs_used") == ["node-1"]
    assert meta.get("mind_authorized_for_stance_skip") is False
    assert meta.get("mind_skip_stance_synthesis") is False
    assert meta.get("mind_contract_only") is True


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
            mind_authorized_for_stance_skip=True,
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

    req = build_mind_run_request(
        _client_request(),
        _plan_request(),
        "550e8400-e29b-41d4-a716-446655440000",
        substrate_telemetry_facet={"status": "absent"},
        cognitive_projection_facet={"schema_version": "cognitive.projection.v1", "projection_id": "explicit"},
    )
    facets = (req.snapshot_inputs or {}).get("facets") or {}
    assert facets.get("substrate_telemetry") == {"status": "absent"}


def test_build_mind_run_request_accepts_explicit_cognitive_projection_facet() -> None:
    _orch_prep()
    from app.mind_runtime import build_mind_run_request

    projection = {
        "schema_version": "cognitive.projection.v1",
        "projection_id": "cog-proj-test",
        "generated_at": "2026-05-16T04:00:00+00:00",
        "source": "cognitive_unification_layer",
        "anchors": {},
        "item_count": 0,
    }
    req = build_mind_run_request(
        _client_request(),
        _plan_request(),
        "550e8400-e29b-41d4-a716-446655440000",
        cognitive_projection_facet=projection,
    )
    facets = (req.snapshot_inputs or {}).get("facets") or {}
    assert facets.get("cognitive_projection") == projection


def test_build_mind_run_request_accepts_inline_metadata_cognitive_projection_facet() -> None:
    _orch_prep()
    from app.mind_runtime import build_mind_run_request

    projection = {
        "schema_version": "cognitive.projection.v1",
        "projection_id": "cog-proj-inline",
        "generated_at": "2026-05-16T04:00:00+00:00",
        "source": "cognitive_unification_layer",
        "anchors": {
            "orion": {
                "anchor": "orion",
                "items": [
                    {
                        "item_id": "item-inline",
                        "node_id": "node-inline",
                        "bucket": "concept",
                        "node_kind": "concept",
                        "label": "inline",
                        "summary": "inline projection",
                        "salience": 0.8,
                        "confidence": 0.7,
                    }
                ],
            }
        },
        "item_count": 1,
    }
    req = build_mind_run_request(
        _client_request(metadata={"mind_enabled": True, "cognitive_projection_facet": projection}),
        _plan_request(),
        "550e8400-e29b-41d4-a716-446655440000",
    )
    facets = (req.snapshot_inputs or {}).get("facets") or {}
    assert facets.get("cognitive_projection", {}).get("projection_id") == "cog-proj-inline"
    assert facets.get("cognitive_projection", {}).get("item_count") == 1


def test_build_mind_run_request_auto_builds_cognitive_projection(monkeypatch) -> None:
    _orch_prep()
    import app.mind_runtime as mind_runtime
    from app.mind_runtime import build_mind_run_request

    calls: list[dict] = []

    class _Projection:
        def model_dump(self, mode: str = "json") -> dict:
            return {
                "schema_version": "cognitive.projection.v1",
                "projection_id": "cog-proj-auto",
                "generated_at": "2026-05-16T04:00:00+00:00",
                "source": "cognitive_unification_layer",
                "anchors": {},
                "item_count": 0,
            }

    def fake_build_projection(ctx, **kwargs):
        calls.append({"ctx": ctx, "kwargs": kwargs})
        return _Projection(), {"item_count": 0, "projection_sources_requested": ["identity_yaml"]}

    monkeypatch.setattr(mind_runtime, "build_cognitive_projection_for_mind_with_diagnostics", fake_build_projection)
    req = build_mind_run_request(
        _client_request(),
        _plan_request(),
        "550e8400-e29b-41d4-a716-446655440000",
    )

    facets = (req.snapshot_inputs or {}).get("facets") or {}
    assert facets.get("cognitive_projection", {}).get("projection_id") == "cog-proj-auto"
    assert facets.get("cognitive_projection", {}).get("projection_build_diagnostics", {}).get("item_count") == 0
    assert calls
    assert calls[0]["ctx"].get("correlation_id") == "550e8400-e29b-41d4-a716-446655440000"
    assert calls[0]["ctx"].get("user_message") == "hi"
    assert calls[0]["ctx"].get("metadata", {}).get("orion_state") == {"ok": True, "status": "present"}
    assert calls[0]["kwargs"].get("publish_tier_outcomes") is True


def test_build_mind_run_request_ignores_empty_inline_projection_shell(monkeypatch) -> None:
    _orch_prep()
    import app.mind_runtime as mind_runtime
    from app.mind_runtime import build_mind_run_request

    built = {
        "schema_version": "cognitive.projection.v1",
        "projection_id": "cog-proj-built",
        "generated_at": "2026-05-16T04:00:00+00:00",
        "source": "cognitive_unification_layer",
        "anchors": {
            "orion": {
                "anchor": "orion",
                "items": [
                    {
                        "item_id": "item-1",
                        "node_id": "node-1",
                        "bucket": "concept",
                        "node_kind": "concept",
                        "label": "x",
                        "summary": "y",
                        "salience": 0.5,
                        "confidence": 0.5,
                    }
                ],
            }
        },
        "item_count": 1,
    }

    class _Projection:
        def model_dump(self, mode: str = "json") -> dict:
            return dict(built)

    def fake_build_projection(ctx, **kwargs):
        return _Projection(), {"item_count": 1}

    monkeypatch.setattr(mind_runtime, "build_cognitive_projection_for_mind_with_diagnostics", fake_build_projection)
    empty_inline = {
        "schema_version": "cognitive.projection.v1",
        "projection_id": "cog-proj-inline-empty",
        "generated_at": "2026-05-16T04:00:00+00:00",
        "source": "cognitive_unification_layer",
        "anchors": {},
        "item_count": 0,
    }
    req = build_mind_run_request(
        _client_request(metadata={"mind_enabled": True, "cognitive_projection_facet": empty_inline}),
        _plan_request(),
        "550e8400-e29b-41d4-a716-446655440000",
    )
    facets = (req.snapshot_inputs or {}).get("facets") or {}
    assert facets.get("cognitive_projection", {}).get("projection_id") == "cog-proj-built"
    assert facets.get("cognitive_projection", {}).get("item_count") == 1
