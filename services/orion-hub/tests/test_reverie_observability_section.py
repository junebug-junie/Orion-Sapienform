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

from scripts.substrate_observability_routes import _reverie_section


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
