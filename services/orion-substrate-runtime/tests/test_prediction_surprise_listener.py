"""prediction.surprise.v1 -> node:prediction.room / node:prediction.body (forecast surprise)."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from orion.schemas.prediction import PredictionSurpriseV1

NOW = datetime(2026, 11, 11, 9, 0, 30, tzinfo=timezone.utc)


def _payload(series="office.temp_c", hm=5, origin_min_ago=10, excess=1.5, fid="77") -> dict:
    origin = NOW.replace(second=0) - timedelta(minutes=origin_min_ago)
    p = 0.05 if excess > 2 else 0.2
    return PredictionSurpriseV1(
        forecast_id=fid, series_id=series, horizon_min=hm, origin_ts=origin, issued_at=origin,
        target_ts=origin + timedelta(minutes=hm), scored_at=NOW, issuing_model="direct_ar",
        interval_source="adaptive", point=22.0, actual=22.6, p_value=p, surprisal=excess + 0.95,
        calm_surprisal=0.95, excess=excess,
    ).model_dump(mode="json")


def _worker():
    from app.worker import BiometricsSubstrateWorker

    w = object.__new__(BiometricsSubstrateWorker)
    w._settings = SimpleNamespace(prediction_surprise_nodes_enabled=True,
                                  prediction_surprise_channel="orion:prediction:surprise")
    w.writes = []
    w._write_prediction_error_node = lambda **kw: w.writes.append(kw) or True
    w._store = MagicMock()
    return w


def test_an_accepted_outcome_writes_its_node_with_the_forecast_as_evidence():
    w = _worker()
    w._process_prediction_surprise(_payload(excess=1.5, fid="77"), now=NOW)
    assert len(w.writes) == 1
    kw = w.writes[0]
    assert kw["node_id"] == "node:prediction.room" and kw["reducer_key"] == "prediction_surprise"
    assert kw["error"] == pytest.approx(1.5 / 0.69793 ** 0.5 / 3, rel=1e-3)
    assert kw["evidence_event_ids"] == ["prediction_forecast:77"]
    # FalkorDB node only: no field receipt, so the digester's decay/expiry never sees it.
    w._store.save_receipt.assert_not_called()
    status = w._prediction_surprise_state().counts
    assert status["accepted"] == 1 and status["node_writes_total"] == 1


def test_the_cabinet_series_writes_the_body_node():
    w = _worker()
    w._process_prediction_surprise(_payload(series="athena.cabinet_temp_c"), now=NOW)
    assert [kw["node_id"] for kw in w.writes] == ["node:prediction.body"]


@pytest.mark.parametrize("payload,verdict", [
    (lambda: _payload(series="somewhere.else"), "unknown_series"),
    (lambda: _payload(hm=15, origin_min_ago=20), "overlapping_origin"),    # 08:40 is not a 15-min block
    (lambda: _payload(origin_min_ago=120), "older_than_window"),           # backlog after an outage
])
def test_refused_outcomes_write_nothing(payload, verdict):
    w = _worker()
    w._process_prediction_surprise(payload(), now=NOW)
    assert w.writes == [] and w._prediction_surprise_state().counts == {verdict: 1}


def test_an_invalid_payload_writes_nothing_and_is_counted():
    w = _worker()
    bad = {**_payload(), "interval_source": "backtest"}      # the gate's own rule, enforced by schema
    w._process_prediction_surprise(bad, now=NOW)
    assert w.writes == [] and w._prediction_surprise_state().counts == {"invalid_total": 1}


def test_a_write_failure_is_counted_never_raised():
    w = _worker()

    def boom(**_kw):
        raise RuntimeError("falkor down")

    w._write_prediction_error_node = boom
    w._process_prediction_surprise(_payload(), now=NOW)
    assert w._prediction_surprise_state().counts["failed_total"] == 1


def test_handler_decodes_the_envelope_and_counts_a_bad_decode():
    w = _worker()
    w._bus = MagicMock()
    w._bus.codec.decode.return_value = SimpleNamespace(ok=False, error="boom")
    asyncio.run(w._handle_prediction_surprise_message({"data": b"x"}))
    assert w._prediction_surprise_state().counts == {"invalid_total": 1}


def test_health_status_reports_each_node_window():
    w = _worker()
    st = w.prediction_surprise_status()
    assert st["enabled"] is True and set(st["nodes"]) == {"node:prediction.room", "node:prediction.body"}
    assert all(v is None for v in st["nodes"].values())      # no outcomes yet: no reading, not calm


def test_prediction_nodes_are_recorded_in_the_pe_history(monkeypatch):
    """Attention, self-readings and the episode history all rank a node against
    substrate_node_prediction_error_history: node:prediction.* must be recorded there."""
    from app.worker import BiometricsSubstrateWorker

    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    import app.settings as settings_mod

    settings_mod._settings = None
    w = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    w._settings = settings_mod.get_settings()
    w._store = MagicMock()
    w._store.fetch_prediction_error_history.return_value = []
    now = datetime.now(timezone.utc)
    node = SimpleNamespace(node_id="node:prediction.room", metadata={"prediction_error": 0.4},
                           temporal=SimpleNamespace(observed_at=now))
    other = SimpleNamespace(node_id="node:elsewhere.x", metadata={"prediction_error": 0.4},
                            temporal=SimpleNamespace(observed_at=now))
    magnitudes = w._prediction_error_magnitudes(nodes=[node, other], now=now)
    saved = w._store.save_prediction_error_history_samples.call_args.args[0]
    assert saved == [("node:prediction.room", now, 0.4)]
    assert "node:prediction.room" in magnitudes


def test_settings_default_on_and_compose_env_parity():
    from pathlib import Path

    import app.settings as settings_mod

    fields = settings_mod.Settings.model_fields
    assert fields["prediction_surprise_nodes_enabled"].default is True
    root = Path(__file__).resolve().parents[1]
    for path in (root / ".env_example", root / "docker-compose.yml"):
        text = path.read_text()
        assert "SUBSTRATE_PREDICTION_SURPRISE_NODES_ENABLED" in text, path
        assert "SUBSTRATE_PREDICTION_SURPRISE_CHANNEL" in text, path


def test_a_write_the_shared_writer_skipped_is_not_counted_as_written():
    """The writer is a no-op with SUBSTRATE_WRITE_PREDICTION_ERROR_NODES off: /health must say so."""
    w = _worker()
    w._write_prediction_error_node = lambda **kw: False
    w._process_prediction_surprise(_payload(), now=NOW)
    counts = w._prediction_surprise_state().counts
    assert counts.get("node_writes_total") is None and counts["node_write_skipped_total"] == 1
