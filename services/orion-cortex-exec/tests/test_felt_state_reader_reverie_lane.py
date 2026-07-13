"""Tests for the latest_reverie_thought lane in felt-state reader."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.substrate_felt_state_reader import _LANES, SubstrateFeltStateReader


def _lane(ctx_key):
    return next(lane for lane in _LANES if lane.ctx_key == ctx_key)


def _reader() -> SubstrateFeltStateReader:
    return SubstrateFeltStateReader(
        enabled=True,
        database_url="postgresql://unused/unused",
        max_age_sec=120,
    )


def test_reverie_lane_registered():
    lane = _lane("latest_reverie_thought")
    assert lane.table == "substrate_reverie_thought"
    assert lane.payload_col == "thought_json"
    assert lane.ts_col == "created_at"
    assert lane.projection_id is None
    # 2x the ~90s reverie tick interval, same convention as curiosity_signals.
    assert lane.max_age_sec == 180


def test_hydrate_injects_fresh_reverie_thought(monkeypatch):
    reader = _reader()
    thought = {
        "interpretation": "The coalition is fixated on unresolved transport anomalies.",
        "hollow": False,
        "evidence_refs": ["evt:1"],
        "coalition": ["node:a"],
    }

    def fake_fetch(lane):
        if lane.ctx_key == "latest_reverie_thought":
            return (thought, datetime.now(timezone.utc) - timedelta(seconds=10))
        return None

    monkeypatch.setattr(reader, "_fetch_lane", fake_fetch)
    ctx: dict = {}
    reader.hydrate(ctx)
    assert ctx["latest_reverie_thought"] == thought


def test_hydrate_rejects_stale_reverie_thought(monkeypatch):
    reader = _reader()
    thought = {"interpretation": "old thought", "hollow": False}

    def fake_fetch(lane):
        if lane.ctx_key == "latest_reverie_thought":
            return (thought, datetime.now(timezone.utc) - timedelta(seconds=200))
        return None

    monkeypatch.setattr(reader, "_fetch_lane", fake_fetch)
    ctx: dict = {}
    reader.hydrate(ctx)
    assert "latest_reverie_thought" not in ctx


def test_hydrate_absent_row_leaves_ctx_key_unset(monkeypatch):
    reader = _reader()

    def fake_fetch(lane):
        return None

    monkeypatch.setattr(reader, "_fetch_lane", fake_fetch)
    ctx: dict = {}
    reader.hydrate(ctx)
    assert "latest_reverie_thought" not in ctx
