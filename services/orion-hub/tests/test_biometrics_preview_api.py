"""Tests for GET /api/biometrics/preview/* (Cognitive EKG toggle + deep-inspect modal)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[3]
HUB_ROOT = Path(__file__).resolve().parents[1]

# Required Hub Settings fields (no defaults) for import without a live .env.
for _key, _val in (
    ("CHANNEL_VOICE_TRANSCRIPT", "orion:voice:transcript"),
    ("CHANNEL_VOICE_LLM", "orion:voice:llm"),
    ("CHANNEL_VOICE_TTS", "orion:voice:tts"),
    ("CHANNEL_COLLAPSE_INTAKE", "orion:collapse:intake"),
    ("CHANNEL_COLLAPSE_TRIAGE", "orion:collapse:triage"),
):
    os.environ.setdefault(_key, _val)


def _ensure_hub_scripts_import_path() -> None:
    for key in list(sys.modules):
        if key == "scripts" or key.startswith("scripts."):
            del sys.modules[key]
    for p in (str(REPO_ROOT), str(HUB_ROOT)):
        try:
            sys.path.remove(p)
        except ValueError:
            pass
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(HUB_ROOT))


_ensure_hub_scripts_import_path()

from scripts import biometrics_preview_routes  # noqa: E402
from scripts.biometrics_node_client import BiometricsNodeClientError  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost/test")
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    monkeypatch.setattr(biometrics_preview_routes, "_history_query", None)
    monkeypatch.setattr(biometrics_preview_routes, "_induction_engine_factory", None)
    app = FastAPI()
    app.include_router(biometrics_preview_routes.router)
    return TestClient(app)


# --- node validation --------------------------------------------------------


def test_atlas_returns_404_decommissioned(client):
    r = client.get("/api/biometrics/preview/snapshot?node=atlas")
    assert r.status_code == 404
    assert r.json()["detail"]["error"] == "node_decommissioned"


def test_unknown_node_returns_404_unknown(client):
    r = client.get("/api/biometrics/preview/snapshot?node=not-a-node")
    assert r.status_code == 404
    assert r.json()["detail"]["error"] == "unknown_node"


def test_node_is_case_insensitive(client, monkeypatch):
    async def fake_snapshot(node):
        assert node == "athena"
        return {"nodes": {"athena": {"status": "ok"}}}

    monkeypatch.setattr(
        biometrics_preview_routes.biometrics_node_client, "fetch_snapshot", fake_snapshot
    )
    r = client.get("/api/biometrics/preview/snapshot?node=ATHENA")
    assert r.status_code == 200
    assert r.json()["node"] == "athena"


# --- /snapshot ---------------------------------------------------------------


def test_snapshot_happy_path(client, monkeypatch):
    async def fake_snapshot(node):
        return {
            "nodes": {
                "athena": {
                    "status": "ok",
                    "reason": None,
                    "as_of": "2026-09-02T00:00:00Z",
                    "freshness_s": 3.1,
                    "summary": {"composites": {"strain": 0.2}},
                    "induction": {"gpu_util": {"level": 0.1}},
                }
            }
        }

    monkeypatch.setattr(
        biometrics_preview_routes.biometrics_node_client, "fetch_snapshot", fake_snapshot
    )
    r = client.get("/api/biometrics/preview/snapshot?node=athena")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["status"] == "ok"
    assert body["summary"]["composites"]["strain"] == pytest.approx(0.2)


def test_snapshot_node_unreachable_degrades_not_raises(client, monkeypatch):
    async def fake_snapshot(node):
        raise BiometricsNodeClientError("unreachable")

    monkeypatch.setattr(
        biometrics_preview_routes.biometrics_node_client, "fetch_snapshot", fake_snapshot
    )
    r = client.get("/api/biometrics/preview/snapshot?node=circe")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["error"] == "node_unreachable"


# --- /history ------------------------------------------------------------


def test_history_unknown_channel_returns_400(client):
    r = client.get("/api/biometrics/preview/history?node=athena&channel=not_a_channel")
    assert r.status_code == 400


def test_history_invalid_window_returns_400(client):
    r = client.get("/api/biometrics/preview/history?node=athena&channel=strain&window=1h")
    assert r.status_code == 400


def test_history_happy_path(client, monkeypatch):
    from datetime import datetime, timezone

    async def rows(*, node, channel, column, hours):
        assert node == "athena"
        assert channel == "gpu_util"
        assert column == "pressures"
        assert hours == 24
        return [
            {"t": datetime(2026, 9, 2, 0, 0, 0, tzinfo=timezone.utc), "v": 0.1},
            {"t": datetime(2026, 9, 2, 0, 5, 0, tzinfo=timezone.utc), "v": 0.9},
        ]

    monkeypatch.setattr(biometrics_preview_routes, "_history_query", rows)
    r = client.get("/api/biometrics/preview/history?node=athena&channel=gpu_util")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["n_raw"] == 2
    assert len(body["series"]) == 2
    assert body["series"][0]["v"] == pytest.approx(0.1)


def test_history_db_failure_returns_ok_false(client, monkeypatch):
    async def failing(*, node, channel, column, hours):
        raise OSError("db unavailable")

    monkeypatch.setattr(biometrics_preview_routes, "_history_query", failing)
    r = client.get("/api/biometrics/preview/history?node=athena&channel=strain")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["series"] == []
    assert body["error"] == "history_unavailable"


# --- /induction ------------------------------------------------------------


def test_induction_reuses_shared_helper(client, monkeypatch):
    captured = {}

    def fake_latest(engine, nodes, **kwargs):
        captured["nodes"] = nodes
        return {"athena": {"gpu_util": {"level": 0.4, "trend": 0.1}}}

    monkeypatch.setattr(
        biometrics_preview_routes, "latest_biometrics_induction_by_node", fake_latest
    )
    monkeypatch.setattr(biometrics_preview_routes, "_induction_engine_factory", lambda: object())

    r = client.get("/api/biometrics/preview/induction?node=athena")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["metrics"]["gpu_util"]["level"] == pytest.approx(0.4)
    assert captured["nodes"] == ["athena"]


def test_induction_no_row_returns_ok_false_not_error(client, monkeypatch):
    monkeypatch.setattr(
        biometrics_preview_routes, "latest_biometrics_induction_by_node", lambda engine, nodes, **k: {}
    )
    monkeypatch.setattr(biometrics_preview_routes, "_induction_engine_factory", lambda: object())

    r = client.get("/api/biometrics/preview/induction?node=circe")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["metrics"] == {}


# --- /gpu ------------------------------------------------------------


def test_gpu_happy_path_with_lane_map_and_processes(client, monkeypatch):
    async def fake_raw_recent(node, *, limit=10):
        return {
            "items": [
                {
                    "timestamp": "2026-09-02T00:00:30Z",
                    "raw": {
                        "gpu": {
                            "gpus": [
                                {
                                    "index": "0",
                                    "name": "Tesla P4",
                                    "utilization_gpu": "9",
                                    "memory_used_mb": "600",
                                    "memory_total_mb": "7680",
                                    "power_draw_watts": "23.0",
                                    "processes": [{"pid": "1", "process_name": "python3"}],
                                }
                            ]
                        }
                    },
                },
                {
                    "timestamp": "2026-09-02T00:00:00Z",
                    "raw": {"gpu": {"gpus": [{"index": "0", "utilization_gpu": "8"}]}},
                },
            ]
        }

    monkeypatch.setattr(
        biometrics_preview_routes.biometrics_node_client, "fetch_raw_recent", fake_raw_recent
    )
    monkeypatch.setattr(
        biometrics_preview_routes.settings, "GPU_LANE_MAP_ATHENA_JSON", '{"0": "orion-vision-host (P4)"}'
    )

    r = client.get("/api/biometrics/preview/gpu?node=athena")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    card = body["gpus"][0]
    assert card["lane"] == "orion-vision-host (P4)"
    assert card["processes"][0]["process_name"] == "python3"
    assert len(card["trend"]) == 2
    # trend is chronological (oldest first) for charting
    assert card["trend"][0]["utilization_gpu"] == "8"
    assert card["trend"][1]["utilization_gpu"] == "9"


def test_gpu_unmapped_index_renders_unassigned(client, monkeypatch):
    async def fake_raw_recent(node, *, limit=10):
        return {
            "items": [
                {
                    "timestamp": "2026-09-02T00:00:00Z",
                    "raw": {"gpu": {"gpus": [{"index": "6", "utilization_gpu": "95"}]}},
                }
            ]
        }

    monkeypatch.setattr(
        biometrics_preview_routes.biometrics_node_client, "fetch_raw_recent", fake_raw_recent
    )
    monkeypatch.setattr(biometrics_preview_routes.settings, "GPU_LANE_MAP_CIRCE_JSON", "{}")

    r = client.get("/api/biometrics/preview/gpu?node=circe")
    assert r.status_code == 200
    assert r.json()["gpus"][0]["lane"] == "unassigned"


def test_router_registered_on_api_routes():
    from scripts import api_routes

    paths = {getattr(route, "path", None) for route in api_routes.router.routes}
    assert "/api/biometrics/preview/snapshot" in paths
    assert "/api/biometrics/preview/gpu" in paths
