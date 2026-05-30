from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from orion.schemas.reduction_receipt import ReductionReceiptV1
from orion.schemas.state_delta import StateDeltaV1
from orion.substrate.receipts.retention import (
    ReceiptRetentionSettings,
    classify_receipt,
    compact_receipt_json,
    payload_fingerprint,
    retention_expires_at,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services/orion-substrate-runtime"))

from app.store import _build_receipt_insert_params


FIXED = datetime(2026, 5, 29, 12, 0, tzinfo=timezone.utc)


def _delta(delta_id: str = "delta-1") -> StateDeltaV1:
    return StateDeltaV1(
        delta_id=delta_id,
        target_projection="substrate.node_biometrics",
        target_kind="node_biometrics",
        target_id="node:athena",
        operation="update",
        before={"x": 1},
        after={"x": 2},
        caused_by_event_ids=["evt-1"],
        reducer_id="biometrics_node_reducer",
    )


def _receipt(**kwargs) -> ReductionReceiptV1:
    base = dict(
        receipt_id="rcpt-1",
        accepted_event_ids=["e1", "e2", "e3"],
        rejected_event_ids=[],
        state_deltas=[_delta()],
        created_at=FIXED,
    )
    base.update(kwargs)
    return ReductionReceiptV1(**base)


def test_classify_success_when_deltas_present():
    settings = ReceiptRetentionSettings()
    c = classify_receipt(_receipt(), settings=settings, rng_value=0.99)
    assert c.receipt_kind == "success"
    assert c.receipt_status == "ok"
    assert c.is_full_payload is False


def test_classify_error_when_warnings():
    settings = ReceiptRetentionSettings()
    c = classify_receipt(_receipt(warnings=["reducer_failed"]), settings=settings, rng_value=0.0)
    assert c.receipt_kind == "error"
    assert c.receipt_status == "error"
    assert c.is_full_payload is True


def test_classify_debug_sample_on_rng():
    settings = ReceiptRetentionSettings(full_payload_sample_rate=0.5)
    c = classify_receipt(_receipt(), settings=settings, rng_value=0.1)
    assert c.receipt_kind == "debug_sample"
    assert c.is_full_payload is True


def test_compact_success_keeps_state_deltas_drops_event_lists():
    receipt = _receipt()
    slim = compact_receipt_json(receipt, is_full_payload=False)
    assert "state_deltas" in slim
    assert len(slim["state_deltas"]) == 1
    assert "accepted_event_ids" not in slim
    assert "projection_updates" not in slim


def test_compact_full_payload_is_full_dump():
    receipt = _receipt()
    full = compact_receipt_json(receipt, is_full_payload=True)
    assert full["accepted_event_ids"] == ["e1", "e2", "e3"]


def test_payload_fingerprint_stable():
    receipt = _receipt()
    a = payload_fingerprint(receipt)
    b = payload_fingerprint(receipt)
    assert a == b
    assert len(a) == 64


def test_retention_expires_at_success_hours():
    settings = ReceiptRetentionSettings(success_hours=48)
    c = classify_receipt(_receipt(), settings=settings, rng_value=0.99)
    exp = retention_expires_at(c, settings=settings, now=FIXED)
    assert exp == FIXED + timedelta(hours=48)


def test_build_receipt_insert_params_success_compact():
    settings = ReceiptRetentionSettings(success_hours=48)
    params = _build_receipt_insert_params(
        _receipt(),
        retention_settings=settings,
        rng_value=0.99,
        force_metadata=False,
        now=FIXED,
    )
    assert params["is_full_payload"] is False
    assert params["receipt_kind"] == "success"
    assert "state_deltas" in params["receipt_json"]
    assert len(params["receipt_json"]["state_deltas"]) == 1
    assert params["expires_at"] == FIXED + timedelta(hours=48)


def test_build_receipt_insert_params_warning_full_error():
    settings = ReceiptRetentionSettings()
    params = _build_receipt_insert_params(
        _receipt(warnings=["reducer_failed"]),
        retention_settings=settings,
        rng_value=0.0,
        force_metadata=False,
        now=FIXED,
    )
    assert params["is_full_payload"] is True
    assert params["receipt_kind"] == "error"
    assert params["receipt_status"] == "error"
    assert params["receipt_json"]["accepted_event_ids"] == ["e1", "e2", "e3"]
