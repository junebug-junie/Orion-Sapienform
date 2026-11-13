"""prediction.turn_surprise.v1 -> node:prediction.chat_turn (successor of node:substrate.chat)."""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from orion.schemas.prediction import PredictionTurnSurpriseV1

NOW = datetime(2026, 11, 13, 9, 0, tzinfo=timezone.utc)


def _bet(q=0.1, happened=True):
    lg = math.log(q / (1 - q))
    s = -math.log(q if happened else 1 - q)
    h = -q * math.log(q) - (1 - q) * math.log(1 - q)
    return {"faculty": "turn.self.asks_question", "stated_p": q, "calibrated_p": q, "happened": happened,
            "surprisal": s, "expected_surprisal": h, "excess": s - h, "calm_var": q * (1 - q) * lg * lg}


def _payload(cid="c1", happened=True, settled_ago=timedelta(minutes=1), caller="chat") -> dict:
    b = _bet(happened=happened)
    return {"correlation_id": cid, "caller": caller, "placed_at": (NOW - settled_ago - timedelta(minutes=2)).isoformat(),
            "settled_at": (NOW - settled_ago).isoformat(), "bets": [b], "excess": b["excess"],
            "calm_var": b["calm_var"], "z": b["excess"] / math.sqrt(b["calm_var"])}


def _worker():
    from app.worker import BiometricsSubstrateWorker

    w = object.__new__(BiometricsSubstrateWorker)
    w._settings = SimpleNamespace(prediction_turn_surprise_node_enabled=True,
                                  prediction_turn_surprise_channel="orion:prediction:turn_surprise")
    w.writes = []
    w._write_prediction_error_node = lambda **kw: w.writes.append(kw) or True
    w._store = MagicMock()
    return w


def test_a_surprising_turn_writes_the_node_with_the_turn_as_evidence():
    w = _worker()
    w._process_prediction_turn_surprise(_payload(happened=True), now=NOW)
    assert len(w.writes) == 1
    kw = w.writes[0]
    assert kw["node_id"] == "node:prediction.chat_turn" and kw["reducer_key"] == "prediction_turn_surprise"
    assert kw["error"] == 1.0                       # a question at p=0.1: z = 3, saturates
    assert kw["evidence_event_ids"] == ["prediction_turn_self_bet:c1"]
    # FalkorDB node only: no field receipt, so the digester's decay never touches it.
    w._store.save_receipt.assert_not_called()
    assert w._prediction_turn_surprise_state().counts == {"accepted": 1, "node_writes_total": 1}


def test_a_calm_turn_writes_exactly_zero_not_nothing():
    w = _worker()
    w._process_prediction_turn_surprise(_payload(happened=False), now=NOW)
    assert [kw["error"] for kw in w.writes] == [0.0]


def test_turns_in_the_window_fold_together_and_old_ones_drop_out():
    w = _worker()
    w._process_prediction_turn_surprise(_payload("a", happened=True), now=NOW)
    for i in range(8):
        w._process_prediction_turn_surprise(_payload(f"calm{i}", happened=False), now=NOW)
    # One surprise among eight calm turns: the window z drops below saturation but stays above 0.
    assert 0.0 < w.writes[-1]["error"] < 1.0
    later = NOW + timedelta(minutes=45)
    w._process_prediction_turn_surprise(_payload("b", happened=False, settled_ago=timedelta(0)) | {
        "settled_at": later.isoformat(), "placed_at": (later - timedelta(minutes=2)).isoformat()}, now=later)
    assert w.writes[-1]["error"] == 0.0 and w.writes[-1]["evidence_event_ids"] == ["prediction_turn_self_bet:b"]


@pytest.mark.parametrize("payload,verdict", [
    (lambda: _payload(settled_ago=timedelta(hours=2)), "older_than_window"),   # backlog after an outage
])
def test_refused_turns_write_nothing(payload, verdict):
    w = _worker()
    w._process_prediction_turn_surprise(payload(), now=NOW)
    assert w.writes == [] and w._prediction_turn_surprise_state().counts == {verdict: 1}


def test_an_invalid_payload_writes_nothing_and_is_counted():
    w = _worker()
    w._process_prediction_turn_surprise(_payload(caller="orion_curiosity"), now=NOW)   # chat only, by schema
    assert w.writes == [] and w._prediction_turn_surprise_state().counts == {"invalid_total": 1}


def test_a_write_failure_is_counted_never_raised():
    w = _worker()

    def boom(**_kw):
        raise RuntimeError("falkor down")

    w._write_prediction_error_node = boom
    w._process_prediction_turn_surprise(_payload(), now=NOW)
    assert w._prediction_turn_surprise_state().counts["failed_total"] == 1


def test_health_status_reports_the_window():
    w = _worker()
    w._process_prediction_turn_surprise(_payload(settled_ago=timedelta(0)), now=datetime.now(timezone.utc))
    st = w.prediction_turn_surprise_status()
    assert st["enabled"] is True and "node:prediction.chat_turn" in st["nodes"]


def test_schema_round_trip():
    assert PredictionTurnSurpriseV1.model_validate(_payload()).caller == "chat"
