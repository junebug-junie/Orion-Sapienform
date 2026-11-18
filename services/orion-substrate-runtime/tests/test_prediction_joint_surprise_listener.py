"""prediction.joint_surprise.v1 -> node:prediction.body_joint / .room_joint (successor of node:substrate.biometrics)."""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from orion.schemas.prediction import PredictionJointSurpriseV1

NOW = datetime(2026, 11, 18, 9, 7, tzinfo=timezone.utc)
ORIGIN = datetime(2026, 11, 18, 8, 55, tzinfo=timezone.utc)     # a 5-min block start


def _payload(jid="1", group="body", p=0.01, origin=ORIGIN, horizon=5, n_cal=1000, **kw) -> dict:
    calm = math.log(n_cal + 1) - math.lgamma(n_cal + 2) / (n_cal + 1)
    return {"joint_id": jid, "family": "family_a", "group": group, "run_id": "run-1", "horizon_min": horizon,
            "origin_ts": origin.isoformat(), "target_ts": (origin + timedelta(minutes=horizon)).isoformat(),
            "scored_at": NOW.isoformat(), "calibration": "adaptive", "n_observed": 30, "n_cal": n_cal,
            "d2": 120.0, "p": p, "calm_point": calm, "excess": -math.log(p) - calm,
            "top_shares": {"athena.load_1m": 0.93, "athena.cpu_watts": 0.03}, **kw}


def _worker():
    from app.worker import BiometricsSubstrateWorker

    w = object.__new__(BiometricsSubstrateWorker)
    w._settings = SimpleNamespace(prediction_joint_surprise_nodes_enabled=True,
                                  prediction_joint_surprise_channel="orion:prediction:joint_surprise")
    w.writes = []
    w._write_prediction_error_node = lambda **kw: w.writes.append(kw) or True
    w._store = MagicMock()
    return w


def test_a_body_spike_writes_the_body_node_with_the_joint_row_as_evidence():
    w = _worker()
    w._process_prediction_joint_surprise(_payload(p=0.001), now=NOW)
    assert len(w.writes) == 1
    kw = w.writes[0]
    assert kw["node_id"] == "node:prediction.body_joint" and kw["reducer_key"] == "prediction_joint_surprise"
    assert 0.9 < kw["error"] <= 1.0                     # -ln 0.001 - 1 = 5.9 nats: z ~ 6 -> saturates
    assert kw["evidence_event_ids"] == ["prediction_joint_surprise:1"]
    w._store.save_receipt.assert_not_called()           # FalkorDB node only: the digester never decays it
    assert w._prediction_joint_surprise_state().counts == {"accepted": 1, "node_writes_total": 1}


def test_room_goes_to_its_own_node_never_the_body():
    w = _worker()
    w._process_prediction_joint_surprise(_payload(group="room"), now=NOW)
    assert [kw["node_id"] for kw in w.writes] == ["node:prediction.room_joint"]


def test_a_calm_outcome_writes_exactly_zero_not_nothing():
    w = _worker()
    w._process_prediction_joint_surprise(_payload(p=0.9), now=NOW)
    assert [kw["error"] for kw in w.writes] == [0.0]


@pytest.mark.parametrize("kw,verdict", [
    ({"origin": ORIGIN - timedelta(hours=2)}, "older_than_window"),                    # a backlog after an outage
    ({"origin": ORIGIN - timedelta(minutes=5), "horizon": 15}, "overlapping_origin"),  # 08:50 starts no 15-min block
])
def test_refused_outcomes_write_nothing(kw, verdict):
    w = _worker()
    w._process_prediction_joint_surprise(_payload(**kw), now=NOW)
    assert w.writes == [] and w._prediction_joint_surprise_state().counts == {verdict: 1}


def test_an_invalid_payload_writes_nothing_and_is_counted():
    w = _worker()
    w._process_prediction_joint_surprise(_payload(calibration="static"), now=NOW)   # adaptive only, by schema
    w._process_prediction_joint_surprise(_payload(group="air"), now=NOW)
    assert w.writes == [] and w._prediction_joint_surprise_state().counts == {"invalid_total": 2}


def test_a_write_failure_is_counted_never_raised():
    w = _worker()

    def boom(**_kw):
        raise RuntimeError("falkor down")

    w._write_prediction_error_node = boom
    w._process_prediction_joint_surprise(_payload(), now=NOW)
    assert w._prediction_joint_surprise_state().counts["failed_total"] == 1


def test_health_status_reports_both_windows():
    w = _worker()
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    origin = now - timedelta(minutes=now.minute % 5 + 5)
    w._process_prediction_joint_surprise(_payload(origin=origin), now=now)
    st = w.prediction_joint_surprise_status()
    assert st["enabled"] is True
    assert set(st["nodes"]) == {"node:prediction.body_joint", "node:prediction.room_joint"}
    assert st["nodes"]["node:prediction.body_joint"]["top_reading"] == "athena.load_1m"
    assert st["nodes"]["node:prediction.room_joint"] is None


def test_schema_round_trip_and_the_blend_is_not_a_valid_group():
    assert PredictionJointSurpriseV1.model_validate(_payload()).group == "body"
    with pytest.raises(ValueError):
        PredictionJointSurpriseV1.model_validate(_payload(group="blend"))
