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
    monkeypatch.setattr(biometrics_preview_routes, "_history_multi_query", None)
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


def test_snapshot_forwards_cluster_measurements_by_node(client, monkeypatch):
    """Athena's own /snapshot carries circe's proxied wattage in cluster.measurements_by_node
    (per-node, pre-aggregation) -- the route must pass it through under
    cluster_measurements_by_node so the frontend can read circe's watts without a fleet sum
    losing which machine drew how much."""

    async def fake_snapshot(node):
        return {
            "nodes": {"athena": {"status": "ok", "summary": {"composites": {"strain": 0.1}}}},
            "cluster": {
                "composite": {"strain": 0.1},
                "trend": {},
                "measurements_by_node": {
                    "athena": {"chassis_watts": 390.0},
                    "circe": {"chassis_watts": 512.0, "pdu_watts": 512.0},
                },
            },
        }

    monkeypatch.setattr(
        biometrics_preview_routes.biometrics_node_client, "fetch_snapshot", fake_snapshot
    )
    r = client.get("/api/biometrics/preview/snapshot?node=athena")
    assert r.status_code == 200
    body = r.json()
    assert body["cluster_measurements_by_node"]["athena"]["chassis_watts"] == pytest.approx(390.0)
    assert body["cluster_measurements_by_node"]["circe"]["chassis_watts"] == pytest.approx(512.0)


def test_snapshot_cluster_measurements_by_node_absent_is_none_not_zero(client, monkeypatch):
    """Circe's own /snapshot has no cluster aggregation (its PDU-proxy poller is empty) --
    absence must read as None, never as an empty dict that could be mistaken for 'measured
    nothing'."""

    async def fake_snapshot(node):
        return {"nodes": {"circe": {"status": "ok", "summary": {}}}}

    monkeypatch.setattr(
        biometrics_preview_routes.biometrics_node_client, "fetch_snapshot", fake_snapshot
    )
    r = client.get("/api/biometrics/preview/snapshot?node=circe")
    assert r.status_code == 200
    assert r.json()["cluster_measurements_by_node"] is None


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


def test_history_chassis_watts_reads_from_measurements_column(client, monkeypatch):
    """chassis_watts is a raw-units channel living in the `measurements` JSONB column, not
    `pressures`/`composites` like every other known channel."""
    from datetime import datetime, timezone

    async def rows(*, node, channel, column, hours):
        assert channel == "chassis_watts"
        assert column == "measurements"
        return [{"t": datetime(2026, 9, 2, 0, 0, 0, tzinfo=timezone.utc), "v": 390.0}]

    monkeypatch.setattr(biometrics_preview_routes, "_history_query", rows)
    r = client.get("/api/biometrics/preview/history?node=athena&channel=chassis_watts")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["series"][0]["v"] == pytest.approx(390.0)


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


# --- /history_multi ---------------------------------------------------------
#
# Added because the modal's "Trended" section originally called /history
# once per channel (up to 14 concurrent asyncpg connections per node-detail
# open, no pooling -- review finding, PR #2010 connection-ceiling history).
# One request, one connection, every channel back in one response.


def test_history_multi_unknown_channel_returns_400(client):
    r = client.get("/api/biometrics/preview/history_multi?node=athena&channels=strain,not_a_channel")
    assert r.status_code == 400
    assert "not_a_channel" in r.json()["detail"]["channels"]


def test_history_multi_empty_channels_returns_400(client):
    r = client.get("/api/biometrics/preview/history_multi?node=athena&channels=")
    assert r.status_code == 400


def test_history_multi_invalid_window_returns_400(client):
    r = client.get("/api/biometrics/preview/history_multi?node=athena&channels=strain&window=1h")
    assert r.status_code == 400


def test_history_multi_happy_path_one_connection_many_channels(client, monkeypatch):
    from datetime import datetime, timezone

    call_count = {"n": 0}

    async def rows(*, node, columns_by_channel, hours):
        call_count["n"] += 1
        assert node == "athena"
        assert columns_by_channel == {"strain": "composites", "gpu_util": "pressures"}
        assert hours == 24
        return [
            {"t": datetime(2026, 9, 2, 0, 0, 0, tzinfo=timezone.utc), "strain": 0.1, "gpu_util": None},
            {"t": datetime(2026, 9, 2, 0, 5, 0, tzinfo=timezone.utc), "strain": 0.2, "gpu_util": 0.9},
        ]

    monkeypatch.setattr(biometrics_preview_routes, "_history_multi_query", rows)
    r = client.get("/api/biometrics/preview/history_multi?node=athena&channels=strain,gpu_util")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    # exactly one query call regardless of channel count -- the whole point
    assert call_count["n"] == 1
    assert len(body["series"]["strain"]) == 2
    # gpu_util's first point was None -- absent, not zero-filled
    assert len(body["series"]["gpu_util"]) == 1
    assert body["series"]["gpu_util"][0]["v"] == pytest.approx(0.9)
    assert body["n_raw"]["strain"] == 2
    assert body["n_raw"]["gpu_util"] == 1


def test_history_multi_db_failure_returns_ok_false_for_every_channel(client, monkeypatch):
    async def failing(*, node, columns_by_channel, hours):
        raise OSError("db unavailable")

    monkeypatch.setattr(biometrics_preview_routes, "_history_multi_query", failing)
    r = client.get("/api/biometrics/preview/history_multi?node=athena&channels=strain,gpu_util")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["series"] == {"strain": [], "gpu_util": []}
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


# --------------------------------------------------------------------------
# Live-DB integration tests
# --------------------------------------------------------------------------
#
# Confirmed live 2026-09-02: query_channel_history_rows/
# query_multi_channel_history_rows originally cast the bound cutoff
# parameter as `$2::timestamptz`, which asyncpg rejects for a plain ISO
# string ("expected a datetime.date or datetime.datetime instance, got
# 'str'") -- every other test in this file mocks the DB layer (_history_query/
# _history_multi_query injected directly), so this asyncpg-level parameter-
# binding error was invisible to the suite and only surfaced against the
# real deployed database. These two tests hit the actual local Postgres
# (same one CLAUDE.md documents as directly queryable) so this class of bug
# fails the suite, not just a live curl after deploy.

import asyncio

_LOCAL_DATABASE_URL = "postgresql://postgres:postgres@127.0.0.1:55432/conjourney"


def _local_postgres_reachable() -> bool:
    try:
        import asyncpg
    except ImportError:
        return False

    async def _try():
        conn = await asyncpg.connect(dsn=_LOCAL_DATABASE_URL, timeout=2)
        await conn.close()

    try:
        asyncio.run(_try())
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _local_postgres_reachable(), reason="local Postgres not reachable")
@pytest.mark.asyncio
async def test_query_channel_history_rows_against_real_postgres_does_not_raise(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", _LOCAL_DATABASE_URL)
    # Real asyncpg parameter binding, not the mocked _history_query seam --
    # this is the whole point: prove the query itself is valid SQL with
    # valid parameter types, whether or not any rows come back.
    rows = await biometrics_preview_routes.query_channel_history_rows(
        node="athena", channel="strain", column="composites", hours=24
    )
    assert isinstance(rows, list)


@pytest.mark.skipif(not _local_postgres_reachable(), reason="local Postgres not reachable")
@pytest.mark.asyncio
async def test_query_multi_channel_history_rows_against_real_postgres_does_not_raise(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", _LOCAL_DATABASE_URL)
    rows = await biometrics_preview_routes.query_multi_channel_history_rows(
        node="athena",
        columns_by_channel={"strain": "composites", "gpu_util": "pressures"},
        hours=24,
    )
    assert isinstance(rows, list)


@pytest.mark.skipif(not _local_postgres_reachable(), reason="local Postgres not reachable")
def test_history_endpoint_against_real_postgres_returns_ok_true(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", _LOCAL_DATABASE_URL)
    app = FastAPI()
    app.include_router(biometrics_preview_routes.router)
    tc = TestClient(app)
    r = tc.get("/api/biometrics/preview/history?node=athena&channel=strain&window=24h")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True, body  # was False with "history_unavailable" under the bug


@pytest.mark.skipif(not _local_postgres_reachable(), reason="local Postgres not reachable")
def test_history_multi_endpoint_against_real_postgres_returns_ok_true(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", _LOCAL_DATABASE_URL)
    app = FastAPI()
    app.include_router(biometrics_preview_routes.router)
    tc = TestClient(app)
    r = tc.get("/api/biometrics/preview/history_multi?node=athena&channels=strain,gpu_util&window=24h")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True, body  # was False with "history_unavailable" under the bug
