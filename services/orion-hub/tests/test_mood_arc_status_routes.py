from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[3]
HUB_ROOT = Path(__file__).resolve().parents[1]


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

from scripts import mood_arc_status_routes  # noqa: E402
from scripts.field_digester_client import FieldDigesterClientError  # noqa: E402


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(mood_arc_status_routes.router)
    return TestClient(app)


# --------------------------------------------------------------- /live


def test_live_relays_field_channel_anomaly_block(monkeypatch) -> None:
    async def _fake_fetch_health():
        return {
            "status": "ok",
            "service": "orion-field-digester",
            "field_channel_anomaly": {
                "enabled": True,
                "encoder_version": "v4",
                "last_live_enrichment_fields": ["action_warrant"],
            },
        }

    monkeypatch.setattr(mood_arc_status_routes, "fetch_health", _fake_fetch_health)
    resp = _client().get("/api/mood-arc-status/live")
    assert resp.status_code == 200
    body = resp.json()
    assert body["reachable"] is True
    assert body["encoder_version"] == "v4"
    assert body["last_live_enrichment_fields"] == ["action_warrant"]


def test_live_reports_unreachable_without_a_500(monkeypatch) -> None:
    """Absence must read as absence -- never a 500, never a silently-empty
    success (CLAUDE.md's 'no empty-shell cognition')."""

    async def _fake_fetch_health():
        raise FieldDigesterClientError("http://x/health unreachable")

    monkeypatch.setattr(mood_arc_status_routes, "fetch_health", _fake_fetch_health)
    resp = _client().get("/api/mood-arc-status/live")
    assert resp.status_code == 200
    body = resp.json()
    assert body["reachable"] is False
    assert "error" in body


def test_live_handles_a_null_field_channel_anomaly_block(monkeypatch) -> None:
    """Review finding (2026-09-04): `.get(key, default)` only substitutes on
    a MISSING key, not a present-but-null one -- `**None` raises TypeError."""

    async def _fake_fetch_health():
        return {"status": "ok", "service": "orion-field-digester", "field_channel_anomaly": None}

    monkeypatch.setattr(mood_arc_status_routes, "fetch_health", _fake_fetch_health)
    resp = _client().get("/api/mood-arc-status/live")
    assert resp.status_code == 200
    body = resp.json()
    assert body["reachable"] is True
    assert body["enabled"] is False


def test_live_handles_scorer_disabled(monkeypatch) -> None:
    async def _fake_fetch_health():
        return {"status": "ok", "service": "orion-field-digester", "field_channel_anomaly": {"enabled": False}}

    monkeypatch.setattr(mood_arc_status_routes, "fetch_health", _fake_fetch_health)
    resp = _client().get("/api/mood-arc-status/live")
    body = resp.json()
    assert body["reachable"] is True
    assert body["enabled"] is False


# --------------------------------------------------------- /phi-v2-inventory


def test_phi_v2_inventory_reports_both_legacy_signals() -> None:
    resp = _client().get("/api/mood-arc-status/phi-v2-inventory")
    assert resp.status_code == 200
    body = resp.json()
    ids = {s["signal_id"] for s in body["legacy_signals"]}
    assert ids == {"phi_heuristic.valence", "phi_intrinsic_reward.v1"}


def test_phi_v2_inventory_confirms_producer_is_dead() -> None:
    """orion-spark-introspector was deleted 2026-07-28 -- this must be a
    live filesystem check, not trust the registry's composition_status
    (which this repo's own convention deliberately does NOT use to encode
    liveness -- see orion/inner_state_registry.md)."""
    resp = _client().get("/api/mood-arc-status/phi-v2-inventory")
    body = resp.json()
    for signal in body["legacy_signals"]:
        assert signal["found_in_registry"] is True
        assert signal["producer_service"] == "orion-spark-introspector"
        assert signal["producer_service_exists"] is False
        assert signal["last_note"]  # non-empty, not fabricated


def test_producer_service_exists_requires_a_docker_compose_file_not_just_a_directory(
    tmp_path, monkeypatch
) -> None:
    """Regression test: confirmed live (2026-09-04 docker smoke test) that
    services/orion-spark-introspector/ still physically exists on disk in
    the deployed checkout (app/, tests/, train/, a gitignored .env) even
    though it was fully deleted from git 2026-07-28 -- a bare
    `producer_dir.is_dir()` check reported "producer present" for a
    service this repo's own git history says is dead. Every real service
    has its own docker-compose.yml; the leftover has none."""
    fake_repo = tmp_path
    (fake_repo / "services" / "orion-spark-introspector").mkdir(parents=True)
    # Leftover debris, no docker-compose.yml -- must read as absent.
    (fake_repo / "services" / "orion-spark-introspector" / "app").mkdir()
    (fake_repo / "docs" / "superpowers" / "specs").mkdir(parents=True)

    monkeypatch.setattr(mood_arc_status_routes, "resolve_repo_root", lambda: fake_repo)
    resp = _client().get("/api/mood-arc-status/phi-v2-inventory")
    body = resp.json()
    for signal in body["legacy_signals"]:
        assert signal["producer_service_exists"] is False


def test_producer_service_exists_true_when_docker_compose_file_present(
    tmp_path, monkeypatch
) -> None:
    fake_repo = tmp_path
    svc_dir = fake_repo / "services" / "orion-spark-introspector"
    svc_dir.mkdir(parents=True)
    (svc_dir / "docker-compose.yml").write_text("services: {}\n")
    (fake_repo / "docs" / "superpowers" / "specs").mkdir(parents=True)

    monkeypatch.setattr(mood_arc_status_routes, "resolve_repo_root", lambda: fake_repo)
    resp = _client().get("/api/mood-arc-status/phi-v2-inventory")
    body = resp.json()
    for signal in body["legacy_signals"]:
        assert signal["producer_service_exists"] is True


def test_phi_v2_inventory_names_the_design_doc() -> None:
    resp = _client().get("/api/mood-arc-status/phi-v2-inventory")
    body = resp.json()["design_doc"]
    assert body["exists"] is True
    assert body["path"] == "docs/superpowers/specs/2026-08-21-phi-v2-design.md"
    assert body["status"] is not None
    assert "not implemented" in body["status"].lower()


def test_phi_v2_inventory_checks_manual_cli_exists() -> None:
    resp = _client().get("/api/mood-arc-status/phi-v2-inventory")
    assert resp.json()["manual_cli_exists"] is True


# --------------------------------------------------------- _first_sentences


@pytest.mark.parametrize(
    "text,count,expected",
    [
        ("One. Two. Three.", 2, "One. Two."),
        ("Just one sentence with no trailing period", 2, "Just one sentence with no trailing period"),
        ("Wrapped\n    across  lines. Second sentence.", 1, "Wrapped across lines."),
    ],
)
def test_first_sentences(text: str, count: int, expected: str) -> None:
    assert mood_arc_status_routes._first_sentences(text, count) == expected


# ------------------------------------------------- /inference-trace, /downstream-triggers


def _field_state_row(failure_pressure: float) -> dict:
    return {
        "schema_version": "field.state.v1",
        "generated_at": "2026-09-04T00:00:00+00:00",
        "tick_id": "tick-test",
        "node_vectors": {
            "node:athena": {
                "availability": 1.0,
                "failure_pressure": failure_pressure,
                "expected_offline_suppression": 1.0,
                "stream_backlog_health": 1.0,
                "delivery_confidence": 1.0,
            }
        },
        "capability_vectors": {},
        "edges": [],
        "recent_perturbations": [],
    }


def _fake_engine_two_queries(first_rows: list[tuple], second_rows: list[tuple]) -> MagicMock:
    """side_effect, not return_value -- _inference_trace_sync() issues two
    DIFFERENT queries (brain_frame_log, then field_state) against the same
    connection, so a single fixed return_value would serve both incorrectly."""
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    conn.execute.side_effect = [first_rows, second_rows]
    return fake_engine


def test_inference_trace_returns_recon_loss_and_correlated_channel(monkeypatch) -> None:
    frame = {
        "regions": [
            {
                "region_id": "field_anomaly:reconstruction",
                "as_of": "2026-09-04T00:00:00+00:00",
                "detail": {"recon_loss": 0.021, "threshold": 0.0143, "anomalous": 1.0},
            },
            {"region_id": "lane:biometrics", "as_of": "2026-09-04T00:00:00+00:00", "detail": {}},
        ]
    }
    import datetime as dt

    frame_rows = [(json.dumps(frame),)]
    channel_rows = [(json.dumps(_field_state_row(0.6)), dt.datetime(2026, 9, 4, tzinfo=dt.timezone.utc))]
    fake_engine = _fake_engine_two_queries(frame_rows, channel_rows)

    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    with patch.object(mood_arc_status_routes, "_trace_engine", return_value=fake_engine):
        resp = _client().get("/api/mood-arc-status/inference-trace?minutes=10")

    assert resp.status_code == 200
    body = resp.json()
    assert body["points"] == [
        {"t": "2026-09-04T00:00:00+00:00", "recon_loss": 0.021, "threshold": 0.0143, "anomalous": True}
    ]
    assert body["channel"] == "failure_pressure"
    assert len(body["channel_points"]) == 1
    assert body["channel_points"][0]["value"] == pytest.approx(0.6)


def test_inference_trace_skips_regions_that_are_not_field_anomaly(monkeypatch) -> None:
    frame = {"regions": [{"region_id": "lane:biometrics", "detail": {"backlog": 3.0}}]}
    fake_engine = _fake_engine_two_queries([(json.dumps(frame),)], [])

    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    with patch.object(mood_arc_status_routes, "_trace_engine", return_value=fake_engine):
        resp = _client().get("/api/mood-arc-status/inference-trace")

    assert resp.json()["points"] == []


def test_inference_trace_degrades_to_empty_without_postgres_uri(monkeypatch) -> None:
    monkeypatch.delenv("POSTGRES_URI", raising=False)
    resp = _client().get("/api/mood-arc-status/inference-trace")
    assert resp.status_code == 200
    body = resp.json()
    assert body["points"] == []
    assert body["channel_points"] == []


def test_downstream_triggers_returns_real_firings(monkeypatch) -> None:
    import datetime as dt

    upstream = {
        "recon_loss": 0.0214,
        "threshold": 0.0144,
        "deviation_direction": "elevated",
        "top_channels": ["failure_pressure=0.598", "gpu_pressure=0.057"],
    }
    # NAIVE datetime (no tzinfo) -- matches the real column type. Regression
    # test (2026-09-04): `metacog_trigger.timestamp` is Postgres `timestamp
    # WITHOUT time zone`, so SQLAlchemy really does hand back a naive
    # datetime here; a tz-aware fixture (as this test used before) can't
    # catch the bug where a naive `.isoformat()` produced an offset-less
    # string that the browser's `Date.parse()` then read as LOCAL time.
    rows = [(dt.datetime(2026, 9, 4, 19, 29), json.dumps(upstream))]
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    conn.execute.return_value = rows

    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    with patch.object(mood_arc_status_routes, "_trace_engine", return_value=fake_engine):
        resp = _client().get("/api/mood-arc-status/downstream-triggers?minutes=60")

    assert resp.status_code == 200
    body = resp.json()
    # Must carry a real UTC offset -- "+00:00", not a bare offset-less string.
    assert body["triggers"] == [
        {
            "t": "2026-09-04T19:29:00+00:00",
            "recon_loss": 0.0214,
            "threshold": 0.0144,
            "deviation_direction": "elevated",
            "top_channel": "failure_pressure=0.598",
        }
    ]


def test_downstream_triggers_handles_a_row_with_no_top_channels(monkeypatch) -> None:
    import datetime as dt

    rows = [(dt.datetime(2026, 9, 4), json.dumps({"recon_loss": 0.02}))]
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    conn.execute.return_value = rows

    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    with patch.object(mood_arc_status_routes, "_trace_engine", return_value=fake_engine):
        resp = _client().get("/api/mood-arc-status/downstream-triggers")

    assert resp.json()["triggers"][0]["top_channel"] is None


# --------------------------------------------------------------- _iso_utc


def test_iso_utc_stamps_utc_on_a_naive_datetime() -> None:
    """The actual regression: a naive datetime (Postgres `timestamp WITHOUT
    time zone`, e.g. metacog_trigger.timestamp) must come out with a real
    UTC offset, not a bare string a browser's Date.parse() would misread
    as local time."""
    import datetime as dt

    naive = dt.datetime(2026, 9, 4, 20, 42, 9, 843730)
    assert mood_arc_status_routes._iso_utc(naive) == "2026-09-04T20:42:09.843730+00:00"


def test_iso_utc_leaves_an_already_aware_datetime_alone() -> None:
    import datetime as dt

    aware = dt.datetime(2026, 9, 4, 20, 42, 9, tzinfo=dt.timezone.utc)
    assert mood_arc_status_routes._iso_utc(aware) == "2026-09-04T20:42:09+00:00"


def test_iso_utc_passes_through_a_non_datetime_value() -> None:
    assert mood_arc_status_routes._iso_utc(None) is None
    assert mood_arc_status_routes._iso_utc("already-a-string") == "already-a-string"
