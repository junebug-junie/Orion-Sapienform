"""The mutation store's in-memory dicts are read on the loop, written on a thread.

Since 2026-09-03 the substrate mutation cycle runs via `asyncio.to_thread`
(services/orion-hub/scripts/main.py) instead of inline on the event loop. That
was necessary -- inline it blocked the hub ~70% of the time -- but it means a
writer on the worker thread can now overlap a reader on the loop, where before
the single event loop serialized them.

The reader that matters is `build_mutation_cognition_context`, on Orion's chat
path, and its call site is NOT inside a try. An unguarded
`sorted(store._proposals.values())` racing an insert raises
`RuntimeError: dictionary changed size during iteration` -- a 500 on the main
chat endpoint.

This currently cannot fire in production, because live cycles report
`proposals_created: 0` -- nothing writes those dicts yet. That is a dormant
landmine, not a fix: it arms itself the moment proposal generation starts
producing, which is the explicit goal of the autonomy work. So it gets a test
now rather than an incident later.
"""
from __future__ import annotations

import threading
import time

import pytest

from orion.substrate.mutation_queue import SubstrateMutationStore, cognition_view_snapshot


class _Row:
    """Minimal stand-in with the attributes the snapshot's callers sort on."""

    def __init__(self, ident: str) -> None:
        self.proposal_id = ident
        self.trial_id = ident
        self.created_at = ident
        self.updated_at = ident
        self.mutation_class = "routing_threshold_patch"


def _hammer_writer(store, stop, errors):
    def writer() -> None:
        i = 0
        try:
            while not stop.is_set():
                i += 1
                with store._lock:
                    store._proposals[f"p{i}"] = _Row(f"p{i}")
                    store._trials[f"t{i}"] = _Row(f"t{i}")
                if i % 8 == 0:
                    time.sleep(0)
        except BaseException as exc:  # noqa: BLE001 -- surfaced by the caller
            errors.append(exc)

    return writer


def test_a_python_level_iteration_of_the_live_dict_really_does_race() -> None:
    """Establish that the hazard is real before asserting the fix closes it.

    This matters because the obvious version of this test is VACUOUS:
    `list(d.values())` is a C-level operation that does not release the GIL
    mid-iteration, so it never raises no matter how hard another thread writes.
    An earlier draft of this file asserted exactly that and passed happily with
    the lock removed.

    The shape that actually raises is a Python-level comprehension over a live
    `.values()` view -- which is precisely what
    `build_mutation_cognition_context` did at line 56 before the snapshot:

        [row for row in mutation_store._trials.values() if ...]

    Bytecode between items means the interpreter can switch threads mid-loop.
    """
    store = SubstrateMutationStore()
    stop = threading.Event()
    errors: list[BaseException] = []
    t = threading.Thread(target=_hammer_writer(store, stop, errors), daemon=True)
    t.start()
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                # Deliberately unguarded -- the pre-fix reader.
                [row for row in store._trials.values() if row.mutation_class == "x"]
            except RuntimeError as exc:
                assert "changed size during iteration" in str(exc)
                break
        else:
            pytest.skip(
                "could not provoke the dict-mutation race in 5s; the fix below "
                "is still correct but this run did not demonstrate the hazard"
            )
    finally:
        stop.set()
        t.join(timeout=5)
    assert not errors, f"writer raised: {errors[0]!r}"


def test_snapshot_survives_concurrent_inserts() -> None:
    """The same hammering, through the locked snapshot, must never raise.

    Non-vacuous by construction: the test above proves the unguarded shape
    raises under identical load.
    """
    store = SubstrateMutationStore()
    stop = threading.Event()
    errors: list[BaseException] = []
    t = threading.Thread(target=_hammer_writer(store, stop, errors), daemon=True)
    t.start()
    try:
        deadline = time.monotonic() + 2.0
        reads = 0
        while time.monotonic() < deadline:
            view = cognition_view_snapshot(store)
            # Python-level iteration of the snapshot -- the shape that races
            # on a live view -- must be safe here.
            [row for row in view["trials"] if row.mutation_class == "routing_threshold_patch"]
            sorted(view["proposals"], key=lambda r: r.created_at, reverse=True)
            reads += 1
        assert reads > 50, f"only {reads} snapshots taken; the race was not exercised"
    finally:
        stop.set()
        t.join(timeout=5)
    assert not errors, f"writer raised: {errors[0]!r}"


def test_snapshot_is_a_point_in_time_copy_not_a_live_view() -> None:
    """Later writes must not appear in an already-taken snapshot.

    A `.values()` view would reflect them, which is what makes the reader able
    to see a trial whose proposal is not yet visible.
    """
    store = SubstrateMutationStore()
    with store._lock:
        store._proposals["a"] = _Row("a")
    view = cognition_view_snapshot(store)
    assert len(view["proposals"]) == 1
    with store._lock:
        store._proposals["b"] = _Row("b")
    assert len(view["proposals"]) == 1, "snapshot tracked a later write -- it is a live view"


def test_every_snapshot_key_is_a_list_not_a_dict_view() -> None:
    store = SubstrateMutationStore()
    for key, value in cognition_view_snapshot(store).items():
        assert isinstance(value, list), f"{key} is {type(value).__name__}, not list"


def test_store_exposes_a_reentrant_lock() -> None:
    """Reentrant matters: the store's mutators call one another."""
    store = SubstrateMutationStore()
    with store._lock:
        with store._lock:  # would deadlock on a plain threading.Lock
            pass
