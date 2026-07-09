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


def test_orch_mind_timeout_defaults_exceed_mind_wall() -> None:
    _orch_prep()
    from app.settings import get_settings

    settings = get_settings()
    assert settings.orion_mind_timeout_sec == 210.0
    assert settings.mind_wall_ms_default == 180_000
    assert settings.orion_mind_timeout_sec * 1000 > settings.mind_wall_ms_default


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
        "anchors": {
            "orion": {
                "anchor": "orion",
                "items": [
                    {
                        "item_id": "item-explicit",
                        "node_id": "node-explicit",
                        "bucket": "concept",
                        "node_kind": "concept",
                        "label": "explicit",
                        "summary": "explicit projection",
                        "salience": 0.8,
                        "confidence": 0.7,
                    }
                ],
            }
        },
        "item_count": 1,
    }
    req = build_mind_run_request(
        _client_request(),
        _plan_request(),
        "550e8400-e29b-41d4-a716-446655440000",
        cognitive_projection_facet=projection,
    )
    facets = (req.snapshot_inputs or {}).get("facets") or {}
    assert facets.get("cognitive_projection", {}).get("projection_id") == "cog-proj-test"
    assert facets.get("cognitive_projection", {}).get("item_count") == 1


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

    def fake_cold(ctx, correlation_id):
        return {
            "schema_version": "cognitive.projection.v1",
            "projection_id": "cog-proj-auto",
            "generated_at": "2026-05-16T04:00:00+00:00",
            "source": "cognitive_unification_layer",
            "anchors": {},
            "item_count": 0,
        }, {"item_count": 0}

    monkeypatch.setattr(mind_runtime, "build_cognitive_projection_for_mind_with_diagnostics", fake_build_projection)
    monkeypatch.setattr(mind_runtime, "_build_cold_cognitive_projection_facet", fake_cold)
    req = build_mind_run_request(
        _client_request(),
        _plan_request(),
        "550e8400-e29b-41d4-a716-446655440000",
    )

    facets = (req.snapshot_inputs or {}).get("facets") or {}
    degraded = facets.get("cognitive_projection_degraded") or {}
    resolution = facets.get("mind_projection_resolution") or {}
    assert degraded.get("projection_id") == "cog-proj-auto" or resolution.get("cold_rebuild_failed") is True
    assert resolution.get("warm_projection_empty") is True or resolution.get("inline_projection_missing") is True
    assert calls
    assert calls[0]["ctx"].get("correlation_id") == "550e8400-e29b-41d4-a716-446655440000"
    assert calls[0]["ctx"].get("user_message") == "hi"
    assert calls[0]["ctx"].get("metadata", {}).get("orion_state") == {"ok": True, "status": "present"}
    assert calls[0]["kwargs"].get("publish_tier_outcomes") is True


def test_build_mind_run_request_shares_populated_projection_in_plan_metadata(monkeypatch) -> None:
    _orch_prep()
    import app.mind_runtime as mind_runtime
    from app.mind_runtime import build_mind_run_request

    class _Projection:
        def model_dump(self, mode: str = "json") -> dict:
            return {
                "schema_version": "cognitive.projection.v1",
                "projection_id": "cog-proj-shared",
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
                                "label": "shared",
                                "summary": "shared projection",
                                "salience": 0.8,
                                "confidence": 0.7,
                            }
                        ],
                    }
                },
                "item_count": 1,
            }

    monkeypatch.setattr(
        mind_runtime,
        "build_cognitive_projection_for_mind_with_diagnostics",
        lambda ctx, **kwargs: (_Projection(), {"item_count": 1}),
    )
    plan = _plan_request()
    build_mind_run_request(_client_request(), plan, "550e8400-e29b-41d4-a716-446655440000")
    meta = plan.context["metadata"]
    assert meta.get("cognitive_projection_facet", {}).get("projection_id") == "cog-proj-shared"
    assert meta.get("cognitive_projection_source") == "orion_cortex_orch_mind_runtime"


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


def test_publish_mind_run_artifact_includes_request_session_id(monkeypatch) -> None:
    _orch_prep()
    from datetime import datetime, timezone

    from app.mind_runtime import publish_mind_run_artifact
    from orion.mind.v1 import MindRunRequestV1, MindRunResultV1
    from orion.core.bus.bus_schemas import ServiceRef

    published: list[dict] = []

    class _Bus:
        async def publish(self, channel: str, env) -> None:
            published.append({"channel": channel, "payload": env.payload})

    mind_res = MindRunResultV1(mind_run_id=uuid4(), ok=True)
    corr = "550e8400-e29b-41d4-a716-446655440099"
    mind_req = MindRunRequestV1.model_validate(
        {
            "schema_id": "mind.run.request.v1",
            "correlation_id": corr,
            "trigger": "user_turn",
            "policy": {"router_profile_id": "default"},
            "snapshot_inputs": {},
        }
    )
    client = _client_request()
    client.context.session_id = "sess-from-request"

    import asyncio

    asyncio.run(
        publish_mind_run_artifact(
            _Bus(),
            source=ServiceRef(service="orion-cortex-orch", instance="test"),
            correlation_id=corr,
            causality_chain=[],
            trace={},
            client_request=client,
            mind_req=mind_req,
            mind_res=mind_res,
        )
    )
    assert published
    artifact = published[0]["payload"]
    assert artifact["session_id"] == "sess-from-request"
    assert artifact["correlation_id"] == corr
    assert datetime.fromisoformat(artifact["created_at_utc"].replace("Z", "+00:00")).tzinfo is not None


def test_mind_http_base_url_from_settings(monkeypatch) -> None:
    _orch_prep()
    from app.mind_runtime import mind_http_base_url
    from app.settings import get_settings

    monkeypatch.setenv("ORION_MIND_BASE_URL", "http://orion-mind:6611")
    get_settings.cache_clear()
    assert mind_http_base_url() == "http://orion-mind:6611"


def test_log_mind_http_failure_includes_endpoint_and_session(caplog) -> None:
    _orch_prep()
    import logging

    from app.mind_runtime import log_mind_http_failure

    caplog.set_level(logging.WARNING, logger="orion.cortex.orch.mind")
    log_mind_http_failure(
        correlation_id="550e8400-e29b-41d4-a716-446655440001",
        session_id="sess-abc",
        trigger="user_turn",
        exc=ConnectionError("Temporary failure in name resolution"),
        base_url="http://orion-athena-mind:6611",
    )
    assert any("mind_http_failed" in r.message for r in caplog.records)
    assert any("orion-athena-mind:6611" in r.message for r in caplog.records)
    assert any("sess-abc" in r.message for r in caplog.records)
    assert any("user_turn" in r.message for r in caplog.records)


def test_readtimeout_publishes_synthetic_mind_artifact(monkeypatch) -> None:
    _orch_prep()
    import asyncio

    import httpx

    from app.mind_runtime import publish_synthetic_mind_http_failure_artifact
    from app.settings import get_settings
    from orion.core.bus.bus_schemas import ServiceRef
    from orion.mind.v1 import MindRunPolicyV1, MindRunRequestV1
    published: list[dict] = []

    class _Bus:
        async def publish(self, channel: str, env) -> None:
            published.append({"channel": channel, "payload": env.payload})

    monkeypatch.setenv("ORION_MIND_TIMEOUT_SEC", "45")
    get_settings.cache_clear()

    req = MindRunRequestV1(
        correlation_id="550e8400-e29b-41d4-a716-446655440099",
        session_id="sess-synth",
        trigger="user_turn",
        policy=MindRunPolicyV1(router_profile_id="default"),
        snapshot_inputs={},
    )
    client_request = _client_request(metadata={"mind_enabled": True})
    client_request.context.session_id = "sess-synth"
    asyncio.run(
        publish_synthetic_mind_http_failure_artifact(
            _Bus(),
            source=ServiceRef(name="orion-cortex-orch", node="test", version="0"),
            correlation_id=req.correlation_id,
            causality_chain=[],
            trace={},
            client_request=client_request,
            mind_req=req,
            exc=httpx.ReadTimeout("timed out"),
            elapsed_ms=45010.0,
        )
    )
    assert len(published) == 1
    assert published[0]["channel"] == "orion:mind:artifact"
    artifact = published[0]["payload"]
    assert artifact["correlation_id"] == req.correlation_id
    assert artifact["session_id"] == "sess-synth"
    assert artifact["ok"] is False
    assert artifact["error_code"] == "mind_http_timeout"
    result = artifact["result_jsonb"]
    assert result["brief"]["machine_contract"]["mind.orch_http_failed"] is True
    assert result["brief"]["machine_contract"]["mind.orch_http_error_type"] == "ReadTimeout"


def test_call_orion_mind_http_uses_configured_base_url(monkeypatch) -> None:
    _orch_prep()
    import asyncio

    import httpx

    from app.mind_runtime import call_orion_mind_http
    from app.settings import get_settings
    from orion.mind.v1 import MindRunRequestV1, MindRunResultV1

    monkeypatch.setenv("ORION_MIND_BASE_URL", "http://orion-mind:6611")
    get_settings.cache_clear()

    captured: dict[str, str] = {}

    class _Resp:
        def __init__(self) -> None:
            self.content = MindRunResultV1(mind_run_id=uuid4(), ok=True).model_dump_json().encode()

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return MindRunResultV1.model_validate_json(self.content)

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url: str, json: dict):
            captured["url"] = url
            return _Resp()

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: _Client())
    req = MindRunRequestV1.model_validate(
        {
            "schema_id": "mind.run.request.v1",
            "correlation_id": "550e8400-e29b-41d4-a716-446655440002",
            "trigger": "user_turn",
            "policy": {"router_profile_id": "default"},
            "snapshot_inputs": {},
        }
    )
    asyncio.run(call_orion_mind_http(req))
    assert captured["url"] == "http://orion-mind:6611/v1/mind/run"
