"""Tests for curiosity_signals lane in felt-state reader."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add app dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.substrate_felt_state_reader import (
    LaneSpec,
    SubstrateFeltStateReader,
)


def test_curiosity_lane_registered():
    """Verify curiosity_signals lane is registered in _LANES."""
    from app.substrate_felt_state_reader import _LANES

    lane_keys = {lane.ctx_key for lane in _LANES}
    assert "curiosity_signals" in lane_keys

    # Find the lane and verify properties
    curiosity_lane = next(lane for lane in _LANES if lane.ctx_key == "curiosity_signals")
    assert curiosity_lane.table == "substrate_endogenous_curiosity_candidates"
    assert curiosity_lane.payload_col == "candidates_json"
    assert curiosity_lane.ts_col == "generated_at"
    assert curiosity_lane.projection_id is None
    assert curiosity_lane.max_age_sec == 30


def test_curiosity_lane_hydrates_fresh():
    """Fresh curiosity signals hydrate into ctx."""
    reader = SubstrateFeltStateReader(enabled=False, database_url="postgresql://invalid", max_age_sec=120)
    reader._enabled = True  # Enable reader after engine init to avoid URL parsing

    # Mock the _fetch_lane to return fresh candidates
    now = datetime.now(timezone.utc)
    candidates = [
        {"signal_type": "curiosity_candidate", "signal_strength": 0.8, "focal_node_refs": ["node:a"]},
        {"signal_type": "curiosity_candidate", "signal_strength": 0.6, "focal_node_refs": ["node:b"]},
    ]

    def mock_fetch(lane: LaneSpec):
        if lane.ctx_key == "curiosity_signals":
            return (candidates, now)
        return None

    reader._fetch_lane = mock_fetch
    ctx = {}
    reader.hydrate(ctx)

    assert "curiosity_signals" in ctx
    assert ctx["curiosity_signals"] == candidates


def test_curiosity_lane_staleness_check():
    """Curiosity signals older than 30s are rejected."""
    reader = SubstrateFeltStateReader(enabled=False, database_url="postgresql://invalid", max_age_sec=120)
    reader._enabled = True

    # Mock _fetch_lane to return stale data
    old_time = datetime.now(timezone.utc) - timedelta(seconds=40)
    candidates = [{"signal_type": "curiosity_candidate", "signal_strength": 0.8}]

    def mock_fetch(lane: LaneSpec):
        if lane.ctx_key == "curiosity_signals":
            return (candidates, old_time)
        return None

    reader._fetch_lane = mock_fetch
    ctx = {}
    reader.hydrate(ctx)

    # Stale data should not hydrate
    assert ctx.get("curiosity_signals") is None


def test_curiosity_lane_absent_table():
    """Gracefully degrade when table is absent."""
    reader = SubstrateFeltStateReader(enabled=False, database_url="postgresql://invalid", max_age_sec=120)
    reader._enabled = True

    # Mock _fetch_lane to return None (table missing)
    def mock_fetch(lane: LaneSpec):
        if lane.ctx_key == "curiosity_signals":
            return None
        return None

    reader._fetch_lane = mock_fetch
    ctx = {}
    reader.hydrate(ctx)

    # Should not raise; ctx key should be absent or empty
    assert ctx.get("curiosity_signals") is None


def test_curiosity_lane_empty_candidates():
    """Empty candidates list handled correctly."""
    reader = SubstrateFeltStateReader(enabled=False, database_url="postgresql://invalid", max_age_sec=120)
    reader._enabled = True

    now = datetime.now(timezone.utc)

    def mock_fetch(lane: LaneSpec):
        if lane.ctx_key == "curiosity_signals":
            return ([], now)
        return None

    reader._fetch_lane = mock_fetch
    ctx = {}
    reader.hydrate(ctx)

    assert "curiosity_signals" in ctx
    assert ctx["curiosity_signals"] == []
