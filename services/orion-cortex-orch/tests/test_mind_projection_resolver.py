from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from uuid import uuid4

import pytest

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


from orion.schemas.cortex.contracts import CortexClientContext, CortexClientRequest, LLMMessage
from orion.schemas.cortex.schemas import ExecutionPlan, ExecutionStep, PlanExecutionRequest


def _plan_request() -> PlanExecutionRequest:
    return PlanExecutionRequest(
        plan=ExecutionPlan(
            verb_name="chat_general",
            steps=[ExecutionStep(verb_name="chat_general", step_name="noop", order=0, services=[])],
        ),
        context={"metadata": {"orion_state": {"ok": True, "status": "present"}}},
    )


def _client_request(*, metadata: dict | None = None) -> CortexClientRequest:
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


def _rich_projection(projection_id: str = "cog-proj-rich", *, item_count: int = 36) -> dict:
    return {
        "schema_version": "cognitive.projection.v1",
        "projection_id": projection_id,
        "generated_at": "2026-05-16T04:00:00+00:00",
        "source": "cognitive_unification_layer",
        "anchors": {
            "orion": {
                "anchor": "orion",
                "items": [
                    {
                        "item_id": f"item-{i}",
                        "anchor": "orion",
                        "node_id": f"node-{i}",
                        "bucket": "concept",
                        "node_kind": "concept",
                        "label": f"item {i}",
                        "summary": f"summary {i}",
                        "salience": 0.8,
                        "confidence": 0.7,
                    }
                    for i in range(item_count)
                ],
            }
        },
        "item_count": item_count,
    }


def test_a_empty_inline_shell_does_not_block_cold_rebuild(monkeypatch) -> None:
    _orch_prep()
    import app.mind_runtime as mind_runtime
    from app.mind_runtime import build_mind_run_request

    def fake_cold(ctx, correlation_id):
        return _rich_projection("cog-proj-cold-36", item_count=36), {"item_count": 36}

    def fake_warm(ctx, **kwargs):
        class _Empty:
            def model_dump(self, mode: str = "json") -> dict:
                return {
                    "schema_version": "cognitive.projection.v1",
                    "projection_id": "cog-proj-warm-empty",
                    "generated_at": "2026-05-16T04:00:00+00:00",
                    "source": "cognitive_unification_layer",
                    "anchors": {},
                    "item_count": 0,
                }

        return _Empty(), {"item_count": 0}

    monkeypatch.setattr(mind_runtime, "build_cognitive_projection_for_mind_with_diagnostics", fake_warm)
    monkeypatch.setattr(mind_runtime, "_build_cold_cognitive_projection_facet", fake_cold)

    empty_inline = _rich_projection("cog-proj-inline-empty", item_count=0)
    empty_inline["anchors"] = {}
    plan = _plan_request()
    req = build_mind_run_request(
        _client_request(metadata={"mind_enabled": True, "cognitive_projection_facet": empty_inline}),
        plan,
        "550e8400-e29b-41d4-a716-446655440000",
    )
    facets = (req.snapshot_inputs or {}).get("facets") or {}
    projection = facets.get("cognitive_projection") or {}
    resolution = facets.get("mind_projection_resolution") or plan.context["metadata"].get("mind_projection_resolution") or {}

    assert projection.get("item_count") == 36
    build_diag = projection.get("projection_build_diagnostics") or resolution.get("projection_build_diagnostics") or resolution
    assert build_diag.get("inline_projection_empty") is True
    assert build_diag.get("cold_rebuild_succeeded") is True


def test_b_nonempty_inline_projection_is_reused(monkeypatch) -> None:
    _orch_prep()
    import app.mind_runtime as mind_runtime
    from app.mind_runtime import resolve_cognitive_projection_for_mind

    inline = _rich_projection("cog-proj-inline-rich", item_count=3)
    cold_called = {"value": False}

    def fake_cold(ctx, correlation_id):
        cold_called["value"] = True
        return None, {}

    monkeypatch.setattr(mind_runtime, "_build_cold_cognitive_projection_facet", fake_cold)
    projection, resolution = resolve_cognitive_projection_for_mind(
        _client_request(metadata={"mind_enabled": True, "cognitive_projection_facet": inline}),
        _plan_request(),
        "550e8400-e29b-41d4-a716-446655440000",
    )

    assert projection is not None
    assert projection.get("projection_id") == "cog-proj-inline-rich"
    assert resolution.get("inline_projection_nonempty") is True
    assert resolution.get("cold_rebuild_attempted") is False
    assert cold_called["value"] is False


def test_c_empty_after_all_sources_becomes_degraded(monkeypatch) -> None:
    _orch_prep()
    import app.mind_runtime as mind_runtime
    from app.mind_runtime import build_mind_run_request

    def fake_warm(ctx, **kwargs):
        class _Empty:
            def model_dump(self, mode: str = "json") -> dict:
                return {
                    "schema_version": "cognitive.projection.v1",
                    "projection_id": "cog-proj-warm-empty",
                    "generated_at": "2026-05-16T04:00:00+00:00",
                    "source": "cognitive_unification_layer",
                    "anchors": {},
                    "item_count": 0,
                }

        return _Empty(), {"item_count": 0}

    monkeypatch.setattr(mind_runtime, "build_cognitive_projection_for_mind_with_diagnostics", fake_warm)
    monkeypatch.setattr(mind_runtime, "_build_cold_cognitive_projection_facet", lambda ctx, corr: (None, {"producer_errors": ["cold_failed"]}))

    req = build_mind_run_request(_client_request(), _plan_request(), str(uuid4()))
    facets = (req.snapshot_inputs or {}).get("facets") or {}
    assert facets.get("cognitive_projection") is None
    resolution = facets.get("mind_projection_resolution") or {}
    assert resolution.get("cold_rebuild_failed") is True
    assert resolution.get("inline_projection_missing") is True
    assert facets.get("cognitive_projection_degraded", {}).get("item_count") == 0


def test_d_brain_chat_general_mind_enabled_uses_nonempty_projection_when_available(monkeypatch) -> None:
    _orch_prep()
    import app.mind_runtime as mind_runtime
    from app.mind_runtime import build_mind_run_request

    monkeypatch.setattr(
        mind_runtime,
        "resolve_cognitive_projection_for_mind",
        lambda client_request, plan_request, correlation_id: (
            _rich_projection("cog-proj-preflight", item_count=12),
            {
                "inline_projection_missing": True,
                "cold_rebuild_succeeded": True,
                "cold_rebuild_attempted": True,
                "resolved_item_count": 12,
                "resolution_path": "cold_isolated_store",
                "orch_invoked_before_exec": True,
            },
        ),
    )
    plan = _plan_request()
    req = build_mind_run_request(_client_request(metadata={"mind_enabled": True}), plan, str(uuid4()))
    projection = (req.snapshot_inputs or {}).get("facets", {}).get("cognitive_projection") or {}
    assert projection.get("item_count") == 12
    assert plan.context["metadata"].get("cognitive_projection_facet", {}).get("item_count") == 12


def test_e_rich_exec_vs_zero_mind_requires_ordering_diagnostics(monkeypatch) -> None:
    _orch_prep()
    import app.mind_runtime as mind_runtime
    from app.mind_runtime import build_mind_run_request

    monkeypatch.setattr(
        mind_runtime,
        "build_cognitive_projection_for_mind_with_diagnostics",
        lambda ctx, **kwargs: (None, {"item_count": 0}),
    )
    monkeypatch.setattr(mind_runtime, "_build_cold_cognitive_projection_facet", lambda ctx, corr: (None, {"item_count": 0}))

    plan = _plan_request()
    build_mind_run_request(
        _client_request(metadata={"mind_enabled": True, "cognitive_projection_facet": _rich_projection(item_count=0)}),
        plan,
        str(uuid4()),
    )
    resolution = plan.context["metadata"].get("mind_projection_resolution") or {}
    assert resolution.get("orch_invoked_before_exec") is True
    assert resolution.get("inline_projection_empty") is True
