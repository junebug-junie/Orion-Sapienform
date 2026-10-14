"""Presence transition history (Temporal Self seam S1): one append-only row per state change.

Times are plain floats (epoch seconds); grace is 120 s and the dark-camera threshold 120 s, so
every expected state below is hand-checkable: last seen at t, "recent" until t+120, "absent" after.
"""

from __future__ import annotations

import asyncio
import pathlib
import sys
from datetime import datetime, timezone

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app import presence as P  # noqa: E402
from app.presence import PresenceRegistry, transition_id_for, transition_row  # noqa: E402

PERSON = frozenset({"person"})
EMPTY: frozenset[str] = frozenset()
T = 1_760_000_000.0  # 2025-10-09T08:53:20Z
HINT = {"subject": "juniper", "state": "probable"}


def _reg(**kw) -> PresenceRegistry:
    return PresenceRegistry(grace_sec=120.0, write_min_interval_sec=5.0, transitions_enabled=True,
                            unknown_after_sec=120.0, **kw)


def _states(rows):
    return [(r["from_state"], r["to_state"]) for r in rows]


def test_rows_only_on_state_change() -> None:
    reg = _reg()
    reg.record("cam0", EMPTY, now=T)  # boot: absent
    reg.record("cam0", EMPTY, now=T + 5)  # unchanged
    reg.record("cam0", PERSON, now=T + 10)  # present
    reg.record("cam0", PERSON, now=T + 15)  # unchanged
    reg.record("cam0", EMPTY, now=T + 20)  # last seen T+15: 5 s <= 120, recent
    reg.record("cam0", EMPTY, now=T + 135)  # 120 s: still recent (<=)
    reg.record("cam0", EMPTY, now=T + 136)  # 121 s > 120: absent
    rows = reg.drain_transitions()
    assert _states(rows) == [(None, "absent"), ("absent", "present"), ("present", "recent"), ("recent", "absent")]
    # occurred_at is the state machine's own state_since: the observation that changed state.
    assert [r["occurred_at"].timestamp() for r in rows] == [T, T + 10, T + 20, T + 136]
    assert reg.drain_transitions() == []  # drained


def test_recent_back_to_present_is_one_row_each_way() -> None:
    reg = _reg()
    reg.record("cam0", PERSON, now=T)
    reg.record("cam0", EMPTY, now=T + 5)
    reg.record("cam0", PERSON, now=T + 30)  # back inside the grace window
    assert _states(reg.drain_transitions()) == [(None, "present"), ("present", "recent"), ("recent", "present")]


def test_unknown_when_frames_stop_and_occurred_at_is_last_observation() -> None:
    reg = _reg()
    reg.record("cam0", PERSON, now=T)
    reg.record("cam0", PERSON, now=T + 50)
    reg.drain_transitions()
    assert reg.sweep_stale(T + 50 + 119) == 0  # 119 s of silence: not yet dark
    assert reg.sweep_stale(T + 50 + 121) == 1  # 121 s > 120 s
    rows = reg.drain_transitions()
    assert _states(rows) == [("present", "unknown")]
    assert rows[0]["occurred_at"].timestamp() == T + 50  # the last moment anything was known
    assert rows[0]["identity_confirmed"] is False and rows[0]["label"] is None
    assert reg.sweep_stale(T + 1000) == 0  # once only


def test_resume_after_unknown_opens_a_fresh_stretch() -> None:
    reg = _reg()
    reg.record("cam0", PERSON, now=T)
    reg.sweep_stale(T + 200)
    reg.drain_transitions()
    # Same person still in frame when the camera returns: a new present row with a fresh time,
    # not a silent continuation of the stretch from before the gap.
    reg.record("cam0", PERSON, now=T + 300)
    rows = reg.drain_transitions()
    assert _states(rows) == [("unknown", "present")]
    assert rows[0]["occurred_at"].timestamp() == T + 300


def test_shutdown_force_closes_every_stream() -> None:
    reg = _reg()
    reg.record("cam0", PERSON, now=T)
    reg.record("cam1", EMPTY, now=T)
    reg.drain_transitions()
    assert reg.sweep_stale(T + 1, force=True) == 2
    assert sorted(r["to_state"] for r in reg.drain_transitions()) == ["unknown", "unknown"]


def test_unknown_never_ties_with_the_change_it_follows() -> None:
    reg = _reg()
    reg.record("cam0", PERSON, now=T)  # the change and the last observation are the same instant
    reg.sweep_stale(T + 500)
    a, b = reg.drain_transitions()
    assert b["occurred_at"] > a["occurred_at"]
    assert b["transition_id"] != a["transition_id"]


def test_label_only_when_confirmed() -> None:
    reg = _reg()
    # Probable hint but no confirmed reading: subject narrows in the snapshot, the row stays unlabeled.
    reg.record("cam0", PERSON, now=T, identity_hint=HINT, identity_confidence=None)
    reg.record("cam0", PERSON, now=T + 5, identity_hint=None, identity_confidence="uncertain")
    rows = reg.drain_transitions()
    assert _states(rows) == [(None, "present")]
    assert rows[0]["identity_confirmed"] is False and rows[0]["label"] is None


def test_confirmed_at_open_carries_the_label() -> None:
    reg = _reg()
    reg.record("cam0", PERSON, now=T, identity_hint=HINT, identity_confidence="confirmed")
    (row,) = reg.drain_transitions()
    assert row["identity_confirmed"] is True and row["label"] == "juniper"


def test_identity_confirmed_mid_stretch_writes_one_upgrade_row() -> None:
    reg = _reg()
    reg.record("cam0", PERSON, now=T)
    reg.record("cam0", PERSON, now=T + 4, identity_hint=HINT, identity_confidence="confirmed")
    reg.record("cam0", PERSON, now=T + 8, identity_hint=HINT, identity_confidence="confirmed")
    reg.record("cam0", EMPTY, now=T + 12)
    reg.record("cam0", PERSON, now=T + 20, identity_hint=HINT, identity_confidence="confirmed")  # same stretch
    rows = reg.drain_transitions()
    assert _states(rows) == [(None, "present"), ("present", "present"), ("present", "recent"), ("recent", "present")]
    assert [r["identity_confirmed"] for r in rows] == [False, True, False, True]
    assert rows[1]["occurred_at"].timestamp() == T + 4
    # After the stretch closes, a new stretch must be confirmed again on its own.
    reg.record("cam0", EMPTY, now=T + 200)  # absent
    reg.record("cam0", PERSON, now=T + 300)
    reg.record("cam0", PERSON, now=T + 305, identity_hint=HINT, identity_confidence="confirmed")
    later = reg.drain_transitions()
    assert _states(later) == [("present", "absent"), ("absent", "present"), ("present", "present")]


def test_disabled_writes_nothing() -> None:
    reg = PresenceRegistry(grace_sec=120.0, transitions_enabled=False)
    reg.record("cam0", PERSON, now=T)
    assert reg.sweep_stale(T + 999) == 0
    assert reg.drain_transitions() == []


def test_transition_id_is_deterministic() -> None:
    at = datetime.fromtimestamp(T, tz=timezone.utc)
    assert transition_id_for("cam0", at, "present") == transition_id_for("cam0", at, "present")
    assert transition_id_for("cam0", at, "present") != transition_id_for("cam0", at, "absent")
    assert transition_id_for("cam0", at, "present") != transition_id_for("cam1", at, "present")
    a = transition_row("cam0", from_state=None, to_state="present", occurred_at=T, identity_confirmed=True, label=None)
    assert a["identity_confirmed"] is False  # confirmed with no label is not a confirmed row
    b = transition_row("cam0", from_state=None, to_state="present", occurred_at=T, identity_confirmed=False, label="x")
    assert b["label"] is None


def test_write_failure_does_not_break_the_state_machine(monkeypatch) -> None:
    def boom(uri):
        raise RuntimeError("db down")

    monkeypatch.setattr(P, "_engine_for", boom)
    assert P.write_transitions_to_postgres(
        [transition_row("cam0", from_state=None, to_state="present", occurred_at=T, identity_confirmed=False, label=None)],
        postgres_uri="postgresql://x",
    ) == 0
    reg = _reg()
    reg.record("cam0", PERSON, now=T)
    assert reg.current_snapshot("cam0")["state"] == "present"


def test_record_survives_a_transition_bookkeeping_error(monkeypatch) -> None:
    reg = _reg()

    def boom(*a, **k):
        raise RuntimeError("bad")

    monkeypatch.setattr(reg, "_note_transition", boom)
    reg.record("cam0", PERSON, now=T)  # never raises
    assert reg.current_snapshot("cam0")["state"] == "present"


def test_writes_executemany_with_the_insert(monkeypatch) -> None:
    calls = []

    class Conn:
        def execute(self, stmt, params):
            calls.append((str(stmt), params))

    class Ctx:
        def __enter__(self):
            return Conn()

        def __exit__(self, *a):
            return False

    class Engine:
        def begin(self):
            return Ctx()

    monkeypatch.setattr(P, "_engine_for", lambda uri: Engine())
    rows = [transition_row("cam0", from_state=None, to_state="present", occurred_at=T, identity_confirmed=False, label=None)]
    assert P.write_transitions_to_postgres(rows, postgres_uri="postgresql://x") == 1
    sql, params = calls[0]
    assert "INSERT INTO vision_presence_transition" in sql and "ON CONFLICT (transition_id) DO NOTHING" in sql
    assert params == rows


def test_service_flushes_transitions_off_the_loop(monkeypatch) -> None:
    """WindowService drains on every presence note (even when the snapshot write is rate-limited)
    and hands the rows to a worker thread."""
    from app import main as M

    written: list = []
    monkeypatch.setattr(M, "write_transitions_to_postgres", lambda rows, postgres_uri: written.append(rows) or len(rows))
    monkeypatch.setattr(M, "write_snapshot_to_postgres", lambda *a, **k: None)
    monkeypatch.setattr(M.settings, "POSTGRES_URI", "postgresql://x")
    monkeypatch.setattr(M.settings, "VISION_PRESENCE_TRANSITIONS_ENABLED", True)
    monkeypatch.setattr(M.settings, "WINDOW_PRESENCE_ENABLED", True)

    async def go():
        svc = M.WindowService()
        svc._note_presence("cam0", PERSON)
        svc._note_presence("cam0", PERSON)  # no change, rate-limited snapshot: no new rows
        await asyncio.gather(*list(svc._presence_write_tasks))

    asyncio.run(go())
    assert [r["to_state"] for batch in written for r in batch] == ["present"]


def test_possible_match_is_not_a_confirmed_name() -> None:
    reg = _reg()
    reg.record("cam0", PERSON, now=T, identity_hint={"subject": "juniper", "state": "possible"},
               identity_confidence="confirmed")
    (row,) = reg.drain_transitions()
    assert row["identity_confirmed"] is False and row["label"] is None


def test_close_writes_unknown_then_refuses_later_rows() -> None:
    reg = _reg()
    reg.record("cam0", PERSON, now=T)
    reg.drain_transitions()
    assert reg.close(T + 1) == 1
    assert [r["to_state"] for r in reg.drain_transitions()] == ["unknown"]
    reg.record("cam0", EMPTY, now=T + 2)  # a late flush after shutdown began
    assert reg.drain_transitions() == []


def test_failed_rows_are_requeued_in_order_and_bounded() -> None:
    reg = _reg()
    reg.record("cam0", PERSON, now=T)
    first = reg.drain_transitions()
    reg.record("cam0", EMPTY, now=T + 5)
    reg.requeue(first)
    assert [r["to_state"] for r in reg.drain_transitions()] == ["present", "recent"]
    reg.requeue([first[0]] * (P.MAX_PENDING_TRANSITIONS + 10))
    assert len(reg.drain_transitions()) == P.MAX_PENDING_TRANSITIONS


def test_service_requeues_on_write_failure(monkeypatch) -> None:
    from app import main as M

    monkeypatch.setattr(M, "write_transitions_to_postgres", lambda rows, postgres_uri: 0)
    monkeypatch.setattr(M, "write_snapshot_to_postgres", lambda *a, **k: None)
    monkeypatch.setattr(M.settings, "POSTGRES_URI", "postgresql://x")
    monkeypatch.setattr(M.settings, "VISION_PRESENCE_TRANSITIONS_ENABLED", True)
    monkeypatch.setattr(M.settings, "WINDOW_PRESENCE_ENABLED", True)

    async def go():
        svc = M.WindowService()
        svc._note_presence("cam0", PERSON)
        await asyncio.gather(*list(svc._presence_write_tasks))
        return svc._presence_registry.has_pending()

    assert asyncio.run(go()) is True
