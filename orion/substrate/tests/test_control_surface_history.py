"""Every control-surface write records what it replaced.

The current-value table holds one row per surface, so before this a write
destroyed the only evidence of the previous setting. Orion's first
self-modification (2026-09-02, routing threshold) hit exactly that: afterwards
nothing in the system could say what the value had been, and the answer had to
be inferred from a pytest fixture that had been leaking writes onto the live row.

These drive the real store against a temp SQLite file and against the in-memory
fallback -- not a stub -- because the property under test is that no write path
can skip the record.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from orion.substrate.mutation_control_surface import RuntimeControlSurfaceStore

_KEY = "routing.chat_reflective_lane_threshold"


@pytest.fixture(params=["sqlite", "memory"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> RuntimeControlSurfaceStore:
    """Both durable and fallback backends must record; neither may be a hole."""
    if request.param == "sqlite":
        return RuntimeControlSurfaceStore(sql_db_path=str(tmp_path / "control.sqlite3"))
    return RuntimeControlSurfaceStore(sql_db_path=None, postgres_url=None)


def _set(store: RuntimeControlSurfaceStore, value: float, actor: str) -> None:
    store.upsert(key=_KEY, value={"surface": _KEY, "value": value, "actor": actor})


def test_first_write_records_that_there_was_no_previous_value(
    store: RuntimeControlSurfaceStore,
) -> None:
    _set(store, 0.5, "seed")

    entries = store.history(_KEY)
    assert len(entries) == 1
    assert entries[0]["previous_value"] is None
    assert entries[0]["new_value"]["value"] == 0.5
    assert entries[0]["actor"] == "seed"


def test_write_records_the_value_it_replaced(store: RuntimeControlSurfaceStore) -> None:
    """The question that was unanswerable on 2026-09-02: what did it move from?"""
    _set(store, 0.5, "seed")
    _set(store, 0.58, "mutation_apply")

    newest = store.history(_KEY)[0]
    assert newest["previous_value"]["value"] == 0.5
    assert newest["new_value"]["value"] == 0.58
    assert newest["actor"] == "mutation_apply"


def test_history_is_newest_first_and_keeps_every_change(
    store: RuntimeControlSurfaceStore,
) -> None:
    for value in (0.5, 0.58, 0.62):
        _set(store, value, "actor")

    entries = store.history(_KEY)
    assert [e["new_value"]["value"] for e in entries] == [0.62, 0.58, 0.5]
    assert store.get(_KEY)["value"] == 0.62  # current-value row still correct


def test_history_is_scoped_to_one_surface(store: RuntimeControlSurfaceStore) -> None:
    _set(store, 0.5, "seed")
    store.upsert(key="other.surface", value={"value": 1.0, "actor": "seed"})

    assert len(store.history(_KEY)) == 1
    assert len(store.history("other.surface")) == 1


def test_history_limit_is_bounded(store: RuntimeControlSurfaceStore) -> None:
    for value in (0.1, 0.2, 0.3, 0.4):
        _set(store, value, "actor")

    assert len(store.history(_KEY, limit=2)) == 2
    assert len(store.history(_KEY, limit=0)) == 1  # clamped up to 1, never unbounded


def test_history_for_an_untouched_surface_is_empty_not_an_error(
    store: RuntimeControlSurfaceStore,
) -> None:
    assert store.history("never.written") == []
