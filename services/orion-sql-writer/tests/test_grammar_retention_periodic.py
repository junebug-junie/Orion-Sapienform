"""The periodic retention loop, the cursor floor, and the non-fatal debt count.

All three exist because of failures that produced NO error at the time:

  * retention ran only at process start, so it deleted ~365,000 rows per restart against
    1,117,440 rows/day of arrival and logged a growing `remaining_debt` that nobody read;
  * shortening the window to 3 days put the retention cutoff within reach of the reduction
    cursors that consume grammar_events forward in time -- a stalled reducer would have had
    its unconsumed backlog deleted silently;
  * substrate_organ_emissions' own debt COUNT exceeded the 10s grammar statement timeout and
    logged the entire run as `retention_failed` even though all 100,000 deletes had committed.

Each test below pins one of those, in the direction that fails silently.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from app import grammar_truth
from app.grammar_truth import GrammarRetentionState


class _Result:
    def __init__(self, rowcount=0, scalar=None):
        self.rowcount = rowcount
        self._scalar = scalar

    def scalar_one(self):
        if isinstance(self._scalar, Exception):
            raise self._scalar
        return self._scalar

    def scalar(self):
        return self._scalar


class _Conn:
    def __init__(self, owner):
        self.owner = owner

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, stmt, params=None):
        return self.owner.respond(" ".join(str(stmt).split()), params)


class _Engine:
    def __init__(self, *, cursor_floor=None, debt=0, delete_rowcounts=None):
        self.cursor_floor = cursor_floor
        self.debt = debt
        self.delete_rowcounts = list(delete_rowcounts or [])
        self.deletes: list[dict] = []

    def connect(self):
        return _Conn(self)

    def begin(self):
        return _Conn(self)

    def respond(self, sql, params):
        if "pg_constraint" in sql:
            return _Result(scalar=0)
        if "substrate_reduction_cursor" in sql:
            return _Result(scalar=self.cursor_floor)
        if sql.startswith("DELETE"):
            self.deletes.append(dict(params or {}))
            n = self.delete_rowcounts.pop(0) if self.delete_rowcounts else 0
            return _Result(rowcount=n)
        if "COUNT(*)" in sql:
            return _Result(scalar=self.debt)
        return _Result()


def _run(engine, **kw):
    kw.setdefault("table", "grammar_events")
    kw.setdefault("id_column", "event_id")
    kw.setdefault("retention_days", 3)
    kw.setdefault("batch_size", 1000)
    kw.setdefault("max_batches", 3)
    kw.setdefault("max_elapsed_sec", 20.0)
    return grammar_truth._apply_bounded_table_retention(engine=engine, **kw)


class TestTheCursorFloor:
    def test_a_lagging_reducer_stops_the_delete_at_its_cursor(self):
        """The whole point. A reducer 10 days behind must not have its backlog deleted."""
        behind = datetime.now(timezone.utc) - timedelta(days=10)
        eng = _Engine(cursor_floor=behind, delete_rowcounts=[0])
        state = _run(eng, respect_cursor_floor=True)
        assert state.cursor_floor_applied is True
        assert state.cutoff_at == behind, "cutoff must be pulled back to the cursor"
        assert eng.deletes and eng.deletes[0]["cutoff"] == behind

    def test_a_caught_up_reducer_does_not_hold_retention_back(self):
        """The floor must only ever RESTRICT deletion, never extend it."""
        eng = _Engine(cursor_floor=datetime.now(timezone.utc), delete_rowcounts=[0])
        state = _run(eng, respect_cursor_floor=True)
        assert state.cursor_floor_applied is False
        assert state.cutoff_at < datetime.now(timezone.utc) - timedelta(days=2)

    def test_no_cursor_row_falls_back_to_the_time_cutoff(self):
        eng = _Engine(cursor_floor=None, delete_rowcounts=[0])
        state = _run(eng, respect_cursor_floor=True)
        assert state.cursor_floor_applied is False
        assert state.cutoff_at is not None

    def test_the_floor_is_off_for_tables_no_cursor_consumes(self):
        """grammar_edges/atoms/organ_emissions are not cursor-consumed; querying the cursor
        table for them would tie their retention to an unrelated reducer's health."""
        behind = datetime.now(timezone.utc) - timedelta(days=10)
        eng = _Engine(cursor_floor=behind, delete_rowcounts=[0])
        state = _run(eng, table="grammar_edges", id_column="edge_id")
        assert state.cursor_floor_applied is False
        assert state.cutoff_at > behind

    def test_only_grammar_events_opts_in(self):
        import inspect

        src = inspect.getsource(grammar_truth.apply_grammar_events_retention)
        assert "respect_cursor_floor=True" in src
        for fn in (
            grammar_truth.apply_grammar_edges_retention,
            grammar_truth.apply_grammar_atoms_retention,
            grammar_truth.apply_substrate_organ_emissions_retention,
        ):
            assert "respect_cursor_floor" not in inspect.getsource(fn)


class TestTheDebtCountCannotMaskASuccessfulPrune:
    def test_a_timeout_counting_debt_does_not_fail_the_run(self):
        eng = _Engine(debt=RuntimeError("canceling statement due to statement timeout"),
                      delete_rowcounts=[1000, 1000, 1000])
        state = _run(eng)
        assert state.rows_pruned_last_run == 3000
        assert state.failure_reason is None, "a reporting query must not mark the prune failed"
        assert state.remaining_debt is None
        assert "timeout" in (state.debt_count_failed_reason or "")


class TestTheCycleRunner:
    def test_one_table_failing_does_not_stop_the_others(self, monkeypatch):
        calls = []

        def boom(days, **kw):
            calls.append("grammar_events")
            raise RuntimeError("nope")

        def ok(days, **kw):
            calls.append("other")
            return GrammarRetentionState(enabled=True)

        monkeypatch.setattr(
            grammar_truth,
            "GRAMMAR_RETENTION_TABLES",
            (("grammar_events", boom), ("grammar_edges", ok), ("grammar_atoms", ok)),
        )
        out = grammar_truth.run_one_retention_cycle(
            days_for={"grammar_events": 3, "grammar_edges": 3, "grammar_atoms": 3},
            max_batches=3,
            max_elapsed_sec=20.0,
        )
        assert calls == ["grammar_events", "other", "other"]
        assert set(out) == {"grammar_edges", "grammar_atoms"}

    def test_a_zero_day_window_skips_the_table_entirely(self, monkeypatch):
        seen = []
        monkeypatch.setattr(
            grammar_truth,
            "GRAMMAR_RETENTION_TABLES",
            (("grammar_events", lambda d, **kw: seen.append(d) or GrammarRetentionState()),),
        )
        grammar_truth.run_one_retention_cycle(
            days_for={"grammar_events": 0}, max_batches=3, max_elapsed_sec=20.0
        )
        assert seen == [], "0 means disabled, not 'delete everything'"


class TestTheLoop:
    def test_it_sleeps_before_the_first_cycle(self, monkeypatch):
        """Startup retention has just run with the much larger startup caps; a cycle firing
        immediately would stack disk load while the bus backlog is also replaying."""
        from app import grammar_retention_loop as mod

        order = []

        async def fake_sleep(_):
            order.append("sleep")
            raise asyncio.CancelledError

        monkeypatch.setattr(mod.asyncio, "sleep", fake_sleep)
        monkeypatch.setattr(
            mod, "run_one_retention_cycle", lambda **kw: order.append("cycle") or {}
        )

        class S:
            grammar_retention_interval_sec = 60.0
            grammar_retention_periodic_max_batches = 3
            grammar_retention_periodic_max_elapsed_sec = 20.0
            grammar_events_retention_days = 3
            grammar_edges_retention_days = 3
            grammar_atoms_retention_days = 3
            substrate_organ_emissions_retention_days = 3

        with pytest.raises(asyncio.CancelledError):
            asyncio.run(mod.grammar_retention_loop(S()))
        assert order == ["sleep"]

    def test_interval_zero_disables_instead_of_busy_looping(self):
        from app import grammar_retention_loop as mod

        class S:
            grammar_retention_interval_sec = 0.0

        asyncio.run(mod.grammar_retention_loop(S()))  # returns, does not hang

    def test_a_failing_cycle_does_not_end_the_loop(self, monkeypatch):
        """A loop that dies on one bad cycle silently restores startup-only retention --
        the exact failure this module exists to remove."""
        from app import grammar_retention_loop as mod

        cycles = {"n": 0}

        async def fake_sleep(_):
            if cycles["n"] >= 2:
                raise asyncio.CancelledError

        def boom(**kw):
            cycles["n"] += 1
            raise RuntimeError("db down")

        monkeypatch.setattr(mod.asyncio, "sleep", fake_sleep)
        monkeypatch.setattr(mod, "run_one_retention_cycle", boom)

        class S:
            grammar_retention_interval_sec = 1.0
            grammar_retention_periodic_max_batches = 3
            grammar_retention_periodic_max_elapsed_sec = 20.0
            grammar_events_retention_days = 3
            grammar_edges_retention_days = 3
            grammar_atoms_retention_days = 3
            substrate_organ_emissions_retention_days = 3

        with pytest.raises(asyncio.CancelledError):
            asyncio.run(mod.grammar_retention_loop(S()))
        assert cycles["n"] == 2, "loop must survive failures, not exit on the first"


class TestTheWindowIsActuallyThree:
    def test_defaults_are_three_days(self):
        from app.settings import Settings
        import inspect

        src = inspect.getsource(Settings)
        for key in (
            "GRAMMAR_EVENTS_RETENTION_DAYS",
            "GRAMMAR_EDGES_RETENTION_DAYS",
            "GRAMMAR_ATOMS_RETENTION_DAYS",
            "SUBSTRATE_ORGAN_EMISSIONS_RETENTION_DAYS",
        ):
            assert f'Field(3, alias="{key}")' in src or f'3, alias="{key}"' in src, key
