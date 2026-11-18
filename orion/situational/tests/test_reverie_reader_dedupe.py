"""Reverie reader returns distinct lines (2026-10-06, turn-latency L3).

A fixated reverie writes the same interpretation many times in a row; newest-N
without a dedupe put the same line in the reply-writer prompt twice."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import orion.situational.reverie_reader as reader

NOW = datetime(2026, 10, 6, 12, 0, tzinfo=timezone.utc)


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _Conn:
    def __init__(self, rows, seen):
        self._rows, self._seen = rows, seen

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query, params):
        self._seen.append(params)
        return _Result(self._rows[: params["limit"]])


class _Engine:
    def __init__(self, rows):
        self.rows, self.params = rows, []

    def connect(self):
        return _Conn(self.rows, self.params)


def _rows(*texts):
    return [(t, NOW - timedelta(minutes=i), 0.5) for i, t in enumerate(texts)]


def test_repeated_reverie_lines_collapse_to_one(monkeypatch) -> None:
    engine = _Engine(
        _rows(
            "Fixated on bus synaptic prediction error.",
            "fixated on  bus synaptic prediction error",
            "Fixated on bus synaptic prediction error!",
            "The walkway is quiet.",
            "Fixated on bus synaptic prediction error.",
            "Juniper asked about the decay loop.",
        )
    )
    monkeypatch.setattr(reader, "_get_engine", lambda: engine)
    out = reader.fetch_recent_reverie_snippets(3)
    assert [r.text for r in out] == [
        "Fixated on bus synaptic prediction error.",  # newest occurrence kept
        "The walkway is quiet.",
        "Juniper asked about the decay loop.",
    ]
    assert engine.params[0]["limit"] > 3  # over-fetched so dupes don't starve the slots


def test_limit_zero_and_no_engine_return_empty(monkeypatch) -> None:
    monkeypatch.setattr(reader, "_get_engine", lambda: _Engine(_rows("a")))
    assert reader.fetch_recent_reverie_snippets(0) == []
    monkeypatch.setattr(reader, "_get_engine", lambda: None)
    assert reader.fetch_recent_reverie_snippets(3) == []


def test_distinct_lines_are_unchanged_and_capped(monkeypatch) -> None:
    monkeypatch.setattr(reader, "_get_engine", lambda: _Engine(_rows("a", "b", "c", "d")))
    assert [r.text for r in reader.fetch_recent_reverie_snippets(2)] == ["a", "b"]
