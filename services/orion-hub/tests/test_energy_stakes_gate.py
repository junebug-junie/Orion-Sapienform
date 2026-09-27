"""The energy-stakes hold is a pure read of one structured snapshot row.

It holds only on fresh, healthy, compared pressure; every other shape of the
row (missing, stale, unknown, unhealthy importer, garbage) is "no hold".
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from uuid import NAMESPACE_URL, uuid5

import pytest

from scripts.energy_stakes_gate import HOLD_REASON, energy_stakes_hold, hold_attention_row

NOW = datetime(2026, 9, 27, 18, tzinfo=timezone.utc)


def _snap(pressure="over_forecast", age_sec=60.0, **over) -> dict:
    base = {
        "as_of": NOW - timedelta(seconds=age_sec), "pressure": pressure, "pressure_reason": "ratio=1.105",
        "projected_to_forecast_ratio": 1.105, "orion_projected_total_usd": 88.4, "forecast_total_usd": 80.0,
        "marginal_usd_per_kwh": 0.12, "importer_state": "healthy", "cycle_start": date(2026, 9, 11),
    }
    base.update(over)
    return base


@pytest.mark.parametrize("pressure", ["over_forecast", "near_forecast"])
def test_fresh_pressure_holds(pressure) -> None:
    hold = energy_stakes_hold(_snap(pressure), now=NOW, max_age_sec=1800)
    assert hold is not None and hold.pressure == pressure


@pytest.mark.parametrize(
    "snapshot",
    [None, {}, _snap("normal"), _snap("unknown"), _snap(age_sec=7200.0), _snap(as_of=None)],
)
def test_never_holds_without_fresh_evidence(snapshot) -> None:
    assert energy_stakes_hold(snapshot, now=NOW, max_age_sec=1800) is None


@pytest.mark.parametrize("state", ["stale", "reauth_required", "degraded", None])
def test_never_holds_without_a_healthy_importer(state) -> None:
    """The schema forbids compared pressure off a non-healthy importer; the gate
    does not trust that alone -- a hand-written or migrated row could violate it."""
    assert energy_stakes_hold(_snap(importer_state=state), now=NOW, max_age_sec=1800) is None


@pytest.mark.parametrize("as_of", ["not-a-date", 12345, (NOW + timedelta(hours=2)).isoformat()])
def test_garbage_or_future_as_of_never_holds(as_of) -> None:
    assert energy_stakes_hold(_snap(as_of=as_of), now=NOW, max_age_sec=1800) is None


def test_iso_string_as_of_is_accepted() -> None:
    snap = _snap(as_of=(NOW - timedelta(seconds=30)).isoformat())
    assert energy_stakes_hold(snap, now=NOW, max_age_sec=1800) is not None


def test_hold_row_is_a_curiosity_attention_row() -> None:
    hold = energy_stakes_hold(_snap(), now=NOW, max_age_sec=1800)
    row = hold_attention_row(hold, now=NOW)
    assert row.process == "curiosity"
    assert row.attention_reason == HOLD_REASON == "held_off:energy_stakes"
    assert row.attended_id is None
    assert row.narrative_kind == "computed"
    assert "$88.40" in row.reason_narrative and "$80.00" in row.reason_narrative
    assert row.entry_id == "curiosity:held_off:energy_stakes:2026-09-11:over_forecast"
    assert row.correlation_id == str(uuid5(NAMESPACE_URL, row.entry_id))


def _row(**over):
    snap = _snap(**over)
    return hold_attention_row(energy_stakes_hold(snap, now=NOW, max_age_sec=1800), now=NOW)


def test_the_same_hold_episode_is_one_row_across_ticks() -> None:
    """A 5-minute snapshot tick must not mint a new attention row per tick: the id is the
    episode (cycle, pressure), so repeats collapse on the attention PK's ON CONFLICT no-op."""
    first, later = _row(age_sec=600.0), _row(age_sec=60.0)
    assert first.entry_id == later.entry_id
    assert first.correlation_id == later.correlation_id


def test_a_new_pressure_or_cycle_is_a_new_episode() -> None:
    base = _row()
    assert _row(pressure="near_forecast").entry_id != base.entry_id
    assert _row(cycle_start=date(2026, 10, 11)).entry_id != base.entry_id


def test_unknown_cycle_falls_back_to_the_snapshot_day_not_one_row_forever() -> None:
    row = _row(cycle_start=None)
    assert row.entry_id == f"curiosity:held_off:energy_stakes:{(NOW - timedelta(seconds=60)).date().isoformat()}:over_forecast"


def test_unknown_numbers_render_as_unknown_not_zero() -> None:
    hold = energy_stakes_hold(_snap(marginal_usd_per_kwh=None), now=NOW, max_age_sec=1800)
    assert "unknown" in hold_attention_row(hold, now=NOW).reason_narrative


def test_hub_settings_default_off_and_read_their_aliases(monkeypatch) -> None:
    from app.settings import Settings

    monkeypatch.delenv("ORION_ENERGY_STAKES_ENABLED", raising=False)
    monkeypatch.delenv("ORION_ENERGY_STAKES_MAX_AGE_SEC", raising=False)
    off = Settings(_env_file=None)
    assert off.ORION_ENERGY_STAKES_ENABLED is False
    assert off.ORION_ENERGY_STAKES_MAX_AGE_SEC == 1800.0

    monkeypatch.setenv("ORION_ENERGY_STAKES_ENABLED", "true")
    monkeypatch.setenv("ORION_ENERGY_STAKES_MAX_AGE_SEC", "900")
    on = Settings(_env_file=None)
    assert on.ORION_ENERGY_STAKES_ENABLED is True
    assert on.ORION_ENERGY_STAKES_MAX_AGE_SEC == 900.0
