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


def test_a_failed_durable_write_raises_instead_of_silently_dropping(tmp_path) -> None:
    """A configured backend that refuses must not look like success.

    The in-memory branch is not a recovery path: ``_source_kind`` stays
    "sqlite"/"postgres" after a failure, so ``get()`` never reads the memory
    copy again. Swallowing the error meant the caller was told the write landed
    when the value had not moved.
    """
    import sqlite3

    import pytest as _pytest

    from orion.substrate.mutation_control_surface import ControlSurfaceWriteError

    db = tmp_path / "control.sqlite3"
    store = RuntimeControlSurfaceStore(sql_db_path=str(db))
    _set(store, 0.5, "seed")

    with sqlite3.connect(db) as conn:  # break only the history table
        conn.execute("DROP TABLE substrate_runtime_control_surface_history")
        conn.commit()

    with _pytest.raises(ControlSurfaceWriteError):
        _set(store, 0.99, "mutation_apply")

    assert store.get(_KEY)["value"] == 0.5  # value write rolled back with it


def test_a_failed_surface_write_does_not_mint_an_adoption(tmp_path, monkeypatch) -> None:
    """The consequence that made the silent drop dangerous.

    ``PatchApplier.apply`` would otherwise build a MutationAdoptionV1, take the
    one-live-mutation-per-surface lock, and record an applied_patch for a change
    that never reached the live surface.
    """
    import sqlite3

    from orion.core.schemas.substrate_mutation import (
        MutationDecisionV1,
        MutationPatchV1,
        MutationProposalV1,
    )
    from orion.substrate import mutation_control_surface
    from orion.substrate.mutation_apply import PatchApplier

    db = tmp_path / "control.sqlite3"
    store = RuntimeControlSurfaceStore(sql_db_path=str(db))
    monkeypatch.setattr(mutation_control_surface, "_CONTROL_SURFACE_STORE", store)
    _set(store, 0.5, "seed")
    with sqlite3.connect(db) as conn:
        conn.execute("DROP TABLE substrate_runtime_control_surface_history")
        conn.commit()

    proposal = MutationProposalV1(
        mutation_class="routing_threshold_patch",
        target_surface="routing",
        lane="operational",
        risk_tier="low",
        rationale="test",
        anchor_scope="orion",
        subject_ref="orion",
        expected_effect="reduce_runtime_executed",
        evidence_refs=["e-1"],
        source_signal_ids=["s-1"],
        source_pressure_id="pr-1",
        patch=MutationPatchV1(
            mutation_class="routing_threshold_patch",
            target_surface="routing",
            target_ref="routing",
            patch={"chat_reflective_lane_threshold": 0.58},
            rollback_payload={"chat_reflective_lane_threshold": 0.5},
        ),
    )
    decision = MutationDecisionV1(proposal_id=proposal.proposal_id, action="auto_promote", reason="t")

    assert PatchApplier(surfaces={}).apply(proposal=proposal, decision=decision) is None
    assert store.get(_KEY)["value"] == 0.5


def test_history_is_bounded_per_surface(monkeypatch, store) -> None:
    """Append-only must still be finite; a busy surface must not crowd out a quiet one."""
    monkeypatch.setenv("SUBSTRATE_CONTROL_SURFACE_HISTORY_MAX_ROWS", "10")
    for i in range(25):
        _set(store, 0.5 + i / 100, "actor")
    store.upsert(key="quiet.surface", value={"value": 1.0, "actor": "seed"})

    entries = store.history(_KEY, limit=500)
    assert len(entries) == 10
    assert entries[0]["new_value"]["value"] == 0.5 + 24 / 100  # newest kept
    assert len(store.history("quiet.surface")) == 1  # untouched by the busy one
