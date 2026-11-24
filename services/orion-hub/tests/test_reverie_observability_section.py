from __future__ import annotations

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


def _row(qid="q-1", **kw):
    base = {"question_id": qid, "seed_kind": "absence", "question": "Was that the neighbor, unrecognised?",
            "none_reason": None, "refused_reason": None, "routed_to": "focus:f1",
            "created_at": datetime(2026, 11, 24, tzinfo=timezone.utc)}
    base.update(kw)
    return base


def test_reverie_section_shows_questions_and_never_shows_a_refusal_as_a_question():
    section = _reverie_section(_FakeEngine([_row("q-1"), _row("q-2", refused_reason="near_duplicate")]))
    assert section is not None and section["count"] == 2
    first, second = section["recent"]
    assert first["question"].startswith("Was that") and first["routed_to"] == "focus:f1"
    assert first["created_at"].startswith("2026-11-24")
    assert second["question"] is None and second["refused_reason"] == "near_duplicate"


def test_reverie_section_none_when_empty():
    assert _reverie_section(_FakeEngine([])) is None
