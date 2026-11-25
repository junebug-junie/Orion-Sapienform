"""prediction.street_rhythm_surprise.v1 -> node:prediction.street_rhythm (walkway rhythm forecast surprise, shadow)."""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

NOW = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)


def _payload(eid="e1", p=0.9, happened=False, scored_at=NOW - timedelta(minutes=2), **kw) -> dict:
    s = -math.log(p if happened else 1 - p)
    h = -p * math.log(p) - (1 - p) * math.log(1 - p)
    lg = math.log(p / (1 - p))
    return {"expectation_id": eid, "stream_id": "walkway", "subject_key": "individual:abc",
            "subject_kind": "individual", "window_start": (NOW - timedelta(hours=3)).isoformat(),
            "window_end": (NOW - timedelta(minutes=20)).isoformat(), "scored_at": scored_at.isoformat(),
            "stated_p": p, "p": p, "happened": happened, "surprisal": s, "calm_surprisal": h,
            "excess": s - h, "calm_var": p * (1 - p) * lg * lg, "key_age_days": 20.0, **kw}


def _worker():
    from app.worker import BiometricsSubstrateWorker

    w = object.__new__(BiometricsSubstrateWorker)
    w._settings = SimpleNamespace(prediction_street_rhythm_node_enabled=True,
                                  prediction_street_rhythm_channel="orion:prediction:street_rhythm_surprise")
    w.writes = []
    w._write_prediction_error_node = lambda **kw: w.writes.append(kw) or True
    w._store = MagicMock()
    return w


def test_a_confident_miss_writes_the_node_with_the_expectation_as_evidence():
    w = _worker()
    w._process_prediction_street_rhythm(_payload(), now=NOW)
    (kw,) = w.writes
    assert kw["node_id"] == "node:prediction.street_rhythm" and kw["reducer_key"] == "prediction_street_rhythm"
    assert kw["error"] == 1.0                       # z = sqrt(0.9 / 0.1) = 3 -> saturates
    assert kw["evidence_event_ids"] == ["vision_percept_expectation:e1"]
    w._store.save_receipt.assert_not_called()       # FalkorDB node only: the field digester never decays it
    assert w._prediction_street_rhythm_state().counts == {"accepted": 1, "node_writes_total": 1}


def test_a_met_forecast_writes_exactly_zero_not_nothing():
    w = _worker()
    w._process_prediction_street_rhythm(_payload(happened=True), now=NOW)
    assert [kw["error"] for kw in w.writes] == [0.0]


def test_a_coin_flip_forecast_alone_carries_no_information_and_writes_nothing():
    w = _worker()
    w._process_prediction_street_rhythm(_payload(p=0.5, happened=False), now=NOW)
    assert w.writes == [] and w._prediction_street_rhythm_state().counts == {"accepted": 1, "no_information": 1}


def test_a_republish_is_a_duplicate_and_writes_nothing():
    w = _worker()
    w._process_prediction_street_rhythm(_payload(), now=NOW)
    w._process_prediction_street_rhythm(_payload(), now=NOW + timedelta(minutes=3))
    assert len(w.writes) == 1
    assert w._prediction_street_rhythm_state().counts == {"accepted": 1, "node_writes_total": 1, "duplicate": 1}


def test_a_backlog_older_than_the_window_writes_nothing():
    w = _worker()
    w._process_prediction_street_rhythm(_payload(scored_at=NOW - timedelta(minutes=31)), now=NOW)
    assert w.writes == [] and w._prediction_street_rhythm_state().counts == {"older_than_window": 1}


def test_an_invalid_payload_writes_nothing_and_is_counted():
    w = _worker()
    w._process_prediction_street_rhythm(_payload(stream_id="doorbell"), now=NOW)
    w._process_prediction_street_rhythm(_payload(subject_label="a name"), now=NOW)   # extra="forbid": no names
    assert w.writes == [] and w._prediction_street_rhythm_state().counts == {"invalid_total": 2}


def test_a_write_failure_is_counted_never_raised():
    w = _worker()

    def boom(**kw):
        raise RuntimeError("falkor down")

    w._write_prediction_error_node = boom
    w._process_prediction_street_rhythm(_payload(), now=NOW)
    assert w._prediction_street_rhythm_state().counts["failed_total"] == 1


def test_health_status_reports_shadow_and_the_window():
    w = _worker()
    w._process_prediction_street_rhythm(_payload(scored_at=datetime.now(timezone.utc)),
                                        now=datetime.now(timezone.utc))
    st = w.prediction_street_rhythm_status()
    assert st["enabled"] is True and st["shadow"] is True
    assert st["node"]["n"] == 1 and st["node"]["value"] == 1.0
