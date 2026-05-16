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
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    app_root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(app_root) not in sys.path:
        sys.path.insert(0, str(app_root))
    import app.main as main_mod

    return TestClient(main_mod.app)


def _projection_with_item() -> dict:
    return {
        "schema_version": "cognitive.projection.v1",
        "projection_id": "cog-proj-http-test",
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
                        "label": "curiosity about Juniper life moment",
                        "summary": "A candidate concept for noticing Juniper's shared life context.",
                        "salience": 0.9,
                        "confidence": 0.8,
                    }
                ],
            }
        },
        "item_count": 1,
        "notes": ["test_projection"],
    }


def test_health_ok(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body.get("ok") is True


def test_mind_run_deterministic_ok(client: TestClient) -> None:
    r = client.post(
        "/v1/mind/run",
        json={
            "correlation_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "snapshot_inputs": {"user_text": "hello"},
            "policy": {"n_loops_max": 1, "wall_time_ms_max": 60000, "router_profile_id": "default"},
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body.get("ok") is True
    assert body.get("mind_run_id")
    assert body.get("mind_quality") == "fallback_contract_only"
    assert body.get("brief", {}).get("stance_payload")
    assert body.get("brief", {}).get("shadow_synthesis") is None
    assert body.get("brief", {}).get("mind_authorized_for_stance_skip") is False


def test_mind_run_surfaces_cognitive_projection_seen_without_shadow_when_no_items(client: TestClient) -> None:
    projection = {
        "schema_version": "cognitive.projection.v1",
        "projection_id": "cog-proj-empty-test",
        "generated_at": "2026-05-16T04:00:00+00:00",
        "source": "cognitive_unification_layer",
        "anchors": {"orion": {"anchor": "orion", "items": []}},
        "item_count": 0,
        "notes": ["test_projection"],
    }
    r = client.post(
        "/v1/mind/run",
        json={
            "correlation_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "snapshot_inputs": {
                "user_text": "hello",
                "facets": {"cognitive_projection": projection},
            },
            "policy": {"n_loops_max": 1, "wall_time_ms_max": 60000, "router_profile_id": "default"},
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body.get("mind_quality") == "fallback_contract_only"
    machine = body.get("brief", {}).get("machine_contract") or {}
    assert machine.get("mind.cognitive_projection_seen") is True
    assert machine.get("mind.cognitive_projection_id") == "cog-proj-empty-test"
    assert machine.get("mind.cognitive_projection_item_count") == 0
    assert machine.get("mind.shadow_synthesis_present") is False
    assert machine.get("mind.projection_starved") is True
    assert body.get("brief", {}).get("shadow_synthesis") is None
    assert body.get("brief", {}).get("mind_authorized_for_stance_skip") is False


def test_mind_run_emits_shadow_synthesis_from_projection_items_without_authority(client: TestClient) -> None:
    r = client.post(
        "/v1/mind/run",
        json={
            "correlation_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "snapshot_inputs": {
                "user_text": "thanks, I'm going to watch a crazy show with Amanda, my wife :) ",
                "facets": {"cognitive_projection": _projection_with_item()},
            },
            "policy": {"n_loops_max": 1, "wall_time_ms_max": 60000, "router_profile_id": "default"},
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body.get("mind_quality") == "shadow_synthesis"
    brief = body.get("brief") or {}
    assert brief.get("mind_quality") == "shadow_synthesis"
    assert brief.get("mind_authorized_for_stance_skip") is False
    shadow = brief.get("shadow_synthesis") or {}
    assert shadow.get("schema_version") == "mind.shadow_synthesis.v1"
    assert shadow.get("present") is True
    assert shadow.get("authorized_for_stance_skip") is False
    assert shadow.get("confidence") > 0
    assert "node-1" in shadow.get("projection_refs_used", [])
    assert shadow.get("stance_candidate", {}).get("conversation_frame") == "playful_relational"
    assert shadow.get("attention_focus")
    assert shadow.get("hazards")
    machine = brief.get("machine_contract") or {}
    assert machine.get("mind.shadow_synthesis_present") is True
    assert machine.get("mind.authorized_for_stance_skip") is False
