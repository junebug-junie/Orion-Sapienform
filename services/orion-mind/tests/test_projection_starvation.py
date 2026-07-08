from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_guard_path = Path(__file__).resolve().parent / "_mind_import_guard.py"


def _mind_prep() -> None:
    spec = importlib.util.spec_from_file_location("_mind_guard_lazy", _guard_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ensure_orion_mind_app()


@pytest.fixture()
def client() -> TestClient:
    _mind_prep()
    import sys

    root = Path(__file__).resolve().parents[3]
    app_root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(app_root) not in sys.path:
        sys.path.insert(0, str(app_root))
    import app.main as main_mod

    return TestClient(main_mod.app)


def _empty_projection_with_diagnostics() -> dict:
    return {
        "schema_version": "cognitive.projection.v1",
        "projection_id": "cog-proj-starved",
        "generated_at": "2026-05-16T04:00:00+00:00",
        "source": "cognitive_unification_layer",
        "anchors": {"orion": {"anchor": "orion", "items": []}},
        "item_count": 0,
        "notes": ["no_active_projection_items"],
        "projection_build_diagnostics": {
            "build_path": "test",
            "projection_sources_requested": ["identity_yaml", "autonomy"],
            "projection_sources_returned": ["identity_yaml"],
            "source_counts": {"orion": 0},
            "dropped_counts_by_reason": {"no_active_projection_items": 1},
            "producer_errors": ["recall"],
            "short_circuit_policy_active": False,
            "item_count": 0,
        },
    }


def test_mind_run_emits_projection_starvation_diagnostics(client: TestClient) -> None:
    empty = _empty_projection_with_diagnostics()
    r = client.post(
        "/v1/mind/run",
        json={
            "correlation_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "snapshot_inputs": {
                "user_text": "hello",
                "facets": {
                    "cognitive_projection_degraded": empty,
                    "mind_projection_resolution": empty["projection_build_diagnostics"],
                },
            },
            "policy": {"n_loops_max": 1, "wall_time_ms_max": 60000, "router_profile_id": "default"},
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body.get("mind_quality") == "fallback_contract_only"
    machine = body.get("brief", {}).get("machine_contract") or {}
    assert machine.get("mind.projection_starved") is True
    assert machine.get("mind.projection_sources_requested") == ["identity_yaml", "autonomy"]
    assert machine.get("mind.projection_producer_errors") == ["recall"]
    assert "Degraded Mind" in (body.get("brief", {}).get("summary_one_paragraph") or "")


def test_light_path_without_projection_does_not_claim_orch_starvation(client: TestClient) -> None:
    """Regression for fix/mind-enrichment-wall-budget: a light Mind run that sends NO
    cognitive projection (e.g. the orion-thought enrichment path) must not surface a
    misleading 'projection starved at Orch preflight' summary — that message
    misdirected root-causing away from the real wall-budget cause."""
    r = client.post(
        "/v1/mind/run",
        json={
            "correlation_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "snapshot_inputs": {"user_text": "just learned my stepdad died today"},
            "policy": {"n_loops_max": 1, "wall_time_ms_max": 60000, "router_profile_id": "default"},
        },
    )
    assert r.status_code == 200
    body = r.json()
    summary = body.get("brief", {}).get("summary_one_paragraph") or ""
    assert "Orch preflight" not in summary
    assert "projection starved" not in summary
    machine = body.get("brief", {}).get("machine_contract") or {}
    assert machine.get("mind.projection_starved") is not True


def test_rich_projection_cannot_become_zero_items_without_diagnostics(client: TestClient) -> None:
    """Regression: populated projection must not downgrade to starved fallback silently."""
    rich = {
        "schema_version": "cognitive.projection.v1",
        "projection_id": "cog-proj-rich",
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
                        "label": "curiosity",
                        "summary": "situated context",
                        "salience": 0.9,
                        "confidence": 0.8,
                    }
                ],
            }
        },
        "item_count": 1,
    }
    r = client.post(
        "/v1/mind/run",
        json={
            "correlation_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "snapshot_inputs": {"user_text": "hello", "facets": {"cognitive_projection": rich}},
            "policy": {"n_loops_max": 1, "wall_time_ms_max": 60000, "router_profile_id": "default"},
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body.get("mind_quality") == "shadow_synthesis"
    machine = body.get("brief", {}).get("machine_contract") or {}
    assert machine.get("mind.cognitive_projection_item_count") == 1
    assert machine.get("mind.projection_starved") is not True
