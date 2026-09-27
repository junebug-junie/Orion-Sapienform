from __future__ import annotations

import asyncio
import json
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from app import main as energy_main
from app.settings import Settings
from orion.energy.testing import hourly
from orion.schemas.energy import ENERGY_IMPORTER_STATUS_KIND, ENERGY_RECONCILE_KIND, ENERGY_STAKES_KIND

REPO = Path(__file__).resolve().parents[3]
TARIFF = REPO / "config/energy/tariff.rmp_ut_sch1.2026-08-10.yaml"
DENVER = ZoneInfo("America/Denver")
NOW = datetime(2026, 9, 27, 6, tzinfo=timezone.utc)


class _Stop(Exception):
    pass


class _FakeBus:
    def __init__(self) -> None:
        self.published: list[tuple[str, object]] = []

    async def publish(self, channel: str, envelope: object) -> None:
        self.published.append((channel, envelope))


def _settings(tmp_path: Path, **kw) -> Settings:
    return Settings(
        ORION_BUS_ENABLED=False,
        ENERGY_TARIFF_PATH=str(TARIFF),
        ENERGY_PROCESSED_DIR=str(tmp_path / "processed"),
        ENERGY_BILL_INBOX_DIR=str(tmp_path / "bills/inbox"),
        ENERGY_BILL_PROCESSED_DIR=str(tmp_path / "bills/processed"),
        ENERGY_PORTAL_STATUS_PATH=str(tmp_path / "portal/status.json"),
        ENERGY_PUBLISH_MAX_PER_SEC=0,
        ENERGY_USAGE_POINT_ID="UP1",
        **kw,
    )


def test_build_pipeline_replays_processed_bills(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    processed = Path(s.ENERGY_BILL_PROCESSED_DIR)
    processed.mkdir(parents=True)
    (processed / "20260912T000000Z__aug.json").write_text(json.dumps({
        "kind": "energy.bill.actual.v1", "billing_period_start": "2026-08-12",
        "billing_period_end": "2026-09-11", "kwh_billed": 712, "current_charges": 101.23,
    }))
    pipeline = energy_main.build_pipeline(s)
    # Replay publishes nothing itself; the replayed bill reconciles once its usage lands.
    out = pipeline.ingest_intervals(hourly(datetime(2026, 8, 12, tzinfo=DENVER), 30 * 24), now=NOW)
    [rec] = [o.payload for o in out if o.kind == ENERGY_RECONCILE_KIND]
    assert rec.billing_period_start == date(2026, 8, 12)
    assert rec.reconcile_gap is None and rec.orion_total_usd is not None


def test_status_loop_publishes_importer_then_stakes_on_configured_channels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = _settings(
        tmp_path, ENERGY_PORTAL_ENABLED=True,
        ENERGY_STAKES_CHANNEL="test:stakes", ENERGY_IMPORTER_STATUS_CHANNEL="test:importer",
    )
    status = Path(s.ENERGY_PORTAL_STATUS_PATH)
    status.parent.mkdir(parents=True)
    status.write_text(json.dumps({
        "state": "reauth_required", "reason": "session_expired", "last_attempt_at": "2026-09-27T06:00:00Z",
    }))
    pipeline = energy_main.build_pipeline(s)
    bus = _FakeBus()

    async def _stop_sleep(_seconds: float) -> None:
        raise _Stop

    monkeypatch.setattr(energy_main.asyncio, "sleep", _stop_sleep)
    with pytest.raises(_Stop):
        asyncio.run(energy_main.status_loop(bus, s, pipeline, asyncio.Lock()))
    assert [(ch, env.kind) for ch, env in bus.published] == [
        ("test:importer", ENERGY_IMPORTER_STATUS_KIND),
        ("test:stakes", ENERGY_STAKES_KIND),
    ]
    importer, stakes = bus.published[0][1].payload, bus.published[1][1].payload
    assert importer["state"] == "reauth_required" and importer["reason"] == "session_expired"
    assert stakes["pressure"] == "unknown" and stakes["pressure_reason"] == "importer_reauth_required"
    assert ENERGY_RECONCILE_KIND not in {env.kind for _, env in bus.published}
