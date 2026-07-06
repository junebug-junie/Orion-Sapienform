from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HUB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for candidate in (str(REPO_ROOT), str(HUB_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from scripts.substrate_observability_routes import (
    _compaction_delta_section,
    _compaction_queue_section,
    _resonance_alert_section,
    _reverie_section,
)


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self

    def all(self):
        return self._rows


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, *a, **k):
        return _FakeResult(self._rows)


class _FakeEngine:
    def __init__(self, rows):
        self._rows = rows

    def connect(self):
        return _FakeConn(self._rows)


def _row(tid="t-1"):
    return {
        "thought_id": tid,
        "correlation_id": "c-1",
        "created_at": datetime(2026, 7, 6, tzinfo=timezone.utc),
        "salience": 0.7,
        "interpretation": "conflict ol-1 keeps re-surfacing",
        "thought_json": json.dumps(
            {"coalition": {"attended_node_ids": ["n-1"], "selected_open_loop_id": "ol-1"}}
        ),
    }


def test_reverie_section_shapes_rows():
    section = _reverie_section(_FakeEngine([_row("t-1"), _row("t-2")]))
    assert section is not None
    assert section["count"] == 2
    first = section["recent"][0]
    assert first["thought_id"] == "t-1"
    assert first["salience"] == 0.7
    assert first["attended_node_ids"] == ["n-1"]
    assert first["selected_open_loop_id"] == "ol-1"
    assert first["created_at"].startswith("2026-07-06")


def test_reverie_section_none_when_empty():
    assert _reverie_section(_FakeEngine([])) is None


def _queue_row(rid="r-1"):
    return {
        "request_id": rid,
        "theme": "loop:ol-1",
        "op_hint": "consolidate",
        "reason": "reverie_chain_max_steps",
        "origin_chain_id": "ch-1",
        "created_at": datetime(2026, 7, 6, tzinfo=timezone.utc),
        "consumed_at": None,
    }


def test_compaction_queue_section_shapes_rows():
    section = _compaction_queue_section(_FakeEngine([_queue_row("r-1"), _queue_row("r-2")]))
    assert section is not None
    assert section["pending_count"] == 2
    assert section["pending"][0]["op_hint"] == "consolidate"
    assert section["pending"][0]["theme"] == "loop:ol-1"


def test_compaction_queue_section_none_when_empty():
    assert _compaction_queue_section(_FakeEngine([])) is None


def _delta_row(did="d-1", applied_at=None):
    return {
        "delta_id": did,
        "dream_id": "dream-1",
        "cards_out": 2,
        "edges_downscaled": 0,
        "rows_pruned": 0,
        "bytes_reclaimed_est": 0,
        "proposal_marked": True,
        "applied_at": applied_at,
        "created_at": datetime(2026, 7, 6, tzinfo=timezone.utc),
        "delta_json": json.dumps(
            {
                "consolidate": [
                    {"gist_card": "Consolidate theme 'loop:ol-1' from 2 settled episode(s)."},
                    {"gist_card": "Consolidate theme 'loop:ol-2' from 1 settled episode(s)."},
                ]
            }
        ),
    }


def test_compaction_delta_section_shapes_rows_as_proposals():
    section = _compaction_delta_section(_FakeEngine([_delta_row("d-1"), _delta_row("d-2")]))
    assert section is not None
    assert section["staged_count"] == 2
    assert section["applied_any"] is False
    first = section["deltas"][0]
    assert first["delta_id"] == "d-1"
    assert first["proposal_marked"] is True
    assert first["applied"] is False
    assert first["cards_out"] == 2
    assert len(first["gist_cards"]) == 2
    assert first["created_at"].startswith("2026-07-06")


def test_compaction_delta_section_none_when_empty():
    assert _compaction_delta_section(_FakeEngine([])) is None


def _alert_row(aid="a-1"):
    return {
        "alert_id": aid,
        "theme_key": "loop:ol-1",
        "violation_count": 3,
        "refractory_sec": 900.0,
        "min_gap_sec": 60.0,
        "occurrences": 4,
        "created_at": datetime(2026, 7, 6, tzinfo=timezone.utc),
    }


def test_resonance_alert_section_shapes_rows():
    section = _resonance_alert_section(_FakeEngine([_alert_row("a-1"), _alert_row("a-2")]))
    assert section is not None
    assert section["alert_count"] == 2
    first = section["alerts"][0]
    assert first["theme_key"] == "loop:ol-1"
    assert first["violation_count"] == 3
    assert first["min_gap_sec"] == 60.0


def test_resonance_alert_section_none_when_empty():
    assert _resonance_alert_section(_FakeEngine([])) is None
