from __future__ import annotations

import json
from datetime import date, datetime, timezone

from app.bills import load_processed_bills, scan_bills
from app.portal_status import read_portal_status
from orion.schemas.energy import EnergyBillActualV1, EnergyBillForecastV1

NOW = datetime(2026, 9, 27, 6, tzinfo=timezone.utc)

ACTUAL = {
    "kind": "energy.bill.actual.v1",
    "billing_period_start": "2026-08-12",
    "billing_period_end": "2026-09-11",
    "kwh_billed": 712,
    "current_charges": 101.23,
    "taxes": 4.10,
}


def _drop(inbox, name, obj) -> None:
    inbox.mkdir(parents=True, exist_ok=True)
    (inbox / name).write_text(obj if isinstance(obj, str) else json.dumps(obj))


def test_hand_entered_bill_is_parsed_stamped_and_replayed(tmp_path) -> None:
    inbox, processed = tmp_path / "inbox", tmp_path / "processed"
    _drop(inbox, "aug.json", ACTUAL)
    [bill] = scan_bills(inbox, processed, now=NOW)
    assert isinstance(bill, EnergyBillActualV1)
    assert bill.source == "file_drop"
    assert bill.retrieved_at == NOW
    assert bill.billing_period_start == date(2026, 8, 12)
    assert bill.energy_charge is None
    assert not list(inbox.glob("*.json"))
    [again] = load_processed_bills(processed)
    assert again == bill


def test_portal_bill_keeps_its_own_source_and_time(tmp_path) -> None:
    inbox, processed = tmp_path / "inbox", tmp_path / "processed"
    got = "2026-09-26T05:00:00+00:00"
    _drop(inbox, "rmp-portal-x-00.json", {
        "kind": "energy.bill.forecast.v1", "source": "rockymountain_power",
        "billing_period_start": "2026-09-11", "as_of": got, "projected_total_usd": 96.0, "retrieved_at": got,
    })
    [fc] = scan_bills(inbox, processed, now=NOW)
    assert isinstance(fc, EnergyBillForecastV1)
    assert fc.source == "rockymountain_power"
    assert fc.retrieved_at == datetime(2026, 9, 26, 5, tzinfo=timezone.utc)


def test_bad_bills_go_to_failed_and_publish_nothing(tmp_path) -> None:
    inbox, processed = tmp_path / "inbox", tmp_path / "processed"
    _drop(inbox, "nokind.json", {k: v for k, v in ACTUAL.items() if k != "kind"})
    _drop(inbox, "empty_forecast.json", {"kind": "energy.bill.forecast.v1", "billing_period_start": "2026-09-11", "as_of": "2026-09-26T05:00:00Z"})
    _drop(inbox, "garbage.json", "{not json")
    _drop(inbox, "listkind.json", {**ACTUAL, "kind": []})
    _drop(inbox, "dictkind.json", {**ACTUAL, "kind": {"a": 1}})
    assert scan_bills(inbox, processed, now=NOW) == []
    assert sorted(p.name.split("__", 1)[1] for p in (inbox / "failed").iterdir()) == [
        "dictkind.json", "empty_forecast.json", "garbage.json", "listkind.json", "nokind.json",
    ]


def test_portal_status_missing_or_invalid_reads_none(tmp_path) -> None:
    assert read_portal_status(tmp_path / "missing.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert read_portal_status(bad) is None
    unknown = tmp_path / "unknown.json"
    unknown.write_text(json.dumps({"state": "fine", "last_attempt_at": "2026-09-27T06:00:00Z"}))
    assert read_portal_status(unknown) is None
    not_object = tmp_path / "list.json"
    not_object.write_text("[]")
    assert read_portal_status(not_object) is None


def test_portal_status_valid_file_parses(tmp_path) -> None:
    path = tmp_path / "status.json"
    path.write_text(json.dumps({
        "state": "reauth_required", "reason": "session_expired",
        "last_attempt_at": "2026-09-27T06:00:00Z", "last_success_at": None,
    }))
    status = read_portal_status(path)
    assert status is not None
    assert status.state == "reauth_required" and status.reason == "session_expired"
    assert status.last_attempt_at == NOW and status.last_success_at is None
