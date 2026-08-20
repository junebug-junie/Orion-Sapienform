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
    def __init__(self, rowcount=0, scalar=None, rows=None):
        self.rowcount = rowcount
        self._scalar = scalar
        self._rows = rows if rows is not None else []

    def mappings(self):
        return self

    def all(self):
        return self._rows

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
    """Models the real shapes: a cursor-row listing, a per-lane unconsumed probe, the FK
    check, the batched DELETE, and the debt COUNT. Any of them can be made to raise."""

    def __init__(
        self,
        *,
        cursor_rows=None,
        lane_oldest=None,
        debt=0,
        delete_rowcounts=None,
        cursor_rows_raises=None,
        lane_probe_raises=None,
    ):
        # cursor_rows: list of {"cursor_name":..., "last_event_created_at":...}
        self.cursor_rows = cursor_rows if cursor_rows is not None else []
        # lane_oldest: dict cursor_name -> datetime|None, the oldest UNCONSUMED row
        self.lane_oldest = lane_oldest or {}
        self.debt = debt
        self.delete_rowcounts = list(delete_rowcounts or [])
        self.cursor_rows_raises = cursor_rows_raises
        self.lane_probe_raises = lane_probe_raises
        self.deletes: list[dict] = []
        self.debt_params: list[dict] = []
        self.lane_probes: list[dict] = []

    def connect(self):
        return _Conn(self)

    def begin(self):
        return _Conn(self)

    def respond(self, sql, params):
        if "pg_constraint" in sql:
            return _Result(scalar=0)
        if "FROM substrate_reduction_cursor" in sql:
            if self.cursor_rows_raises:
                raise self.cursor_rows_raises
            return _Result(rows=self.cursor_rows)
        if "MIN(created_at)" in sql:
            if self.lane_probe_raises:
                raise self.lane_probe_raises
            self.lane_probes.append(dict(params or {}))
            sources = (params or {}).get("sources") or []
            key = _LANE_BY_SOURCE.get(sources[0] if sources else None)
            return _Result(scalar=self.lane_oldest.get(key))
        if sql.startswith("DELETE"):
            self.deletes.append(dict(params or {}))
            n = self.delete_rowcounts.pop(0) if self.delete_rowcounts else 0
            return _Result(rowcount=n)
        if "COUNT(*)" in sql:
            self.debt_params.append(dict(params or {}))
            return _Result(scalar=self.debt)
        return _Result()


_LANE_BY_SOURCE = {
    srcs[0]: name for name, srcs, _ in grammar_truth.GRAMMAR_LANES
}


def _cursor_rows(**by_lane):
    """Every lane present with a live cursor unless overridden."""
    now = datetime.now(timezone.utc)
    return [
        {"cursor_name": name, "last_event_created_at": by_lane.get(name, now)}
        for name, _, _ in grammar_truth.GRAMMAR_LANES
    ]


def _run(engine, **kw):
    kw.setdefault("table", "grammar_events")
    kw.setdefault("id_column", "event_id")
    kw.setdefault("retention_days", 3)
    kw.setdefault("batch_size", 1000)
    kw.setdefault("max_batches", 3)
    kw.setdefault("max_elapsed_sec", 20.0)
    return grammar_truth._apply_bounded_table_retention(engine=engine, **kw)


class TestTheCursorFloorAsksTheRightQuestion:
    """The bug code review caught before deploy: a floor built on cursor POSITION pins on
    lane silence, not reducer stall. chat_grammar_consumer tracks Juniper talking to Orion;
    it went quiet for 1 day 19.6 hours on a healthy system ingesting ~500k events/day. A
    position-based floor would have stopped retention for an ordinary quiet weekend."""

    def test_a_silent_lane_with_nothing_unconsumed_imposes_no_floor(self):
        """THE regression. Cursor far behind, but zero unconsumed rows -> no floor."""
        quiet = datetime.now(timezone.utc) - timedelta(days=9)
        eng = _Engine(
            cursor_rows=_cursor_rows(chat_grammar_consumer=quiet),
            lane_oldest={},  # nothing owed anywhere
            delete_rowcounts=[0],
        )
        state = _run(eng, respect_cursor_floor=True)
        assert state.cursor_floor_applied is False, "silence is not a stall"
        assert state.cutoff_at > quiet
        assert eng.deletes, "retention must still run"

    def test_a_genuinely_behind_lane_does_impose_a_floor(self):
        """Same stale cursor, but real unconsumed rows exist -> floor binds at the oldest."""
        quiet = datetime.now(timezone.utc) - timedelta(days=9)
        owed = datetime.now(timezone.utc) - timedelta(days=8)
        eng = _Engine(
            cursor_rows=_cursor_rows(chat_grammar_consumer=quiet),
            lane_oldest={"chat_grammar_consumer": owed},
            delete_rowcounts=[0],
        )
        state = _run(eng, respect_cursor_floor=True)
        assert state.cursor_floor_applied is True
        assert state.cutoff_at == owed
        assert eng.deletes[0]["cutoff"] == owed

    def test_the_floor_takes_the_oldest_across_lanes(self):
        older = datetime.now(timezone.utc) - timedelta(days=9)
        newer = datetime.now(timezone.utc) - timedelta(days=5)
        eng = _Engine(
            cursor_rows=_cursor_rows(),
            lane_oldest={"chat_grammar_consumer": newer, "route_grammar_consumer": older},
            delete_rowcounts=[0],
        )
        state = _run(eng, respect_cursor_floor=True)
        assert state.cutoff_at == older

    def test_every_lane_is_probed(self):
        eng = _Engine(cursor_rows=_cursor_rows(), delete_rowcounts=[0])
        _run(eng, respect_cursor_floor=True)
        probed = {tuple(p["sources"])[0] for p in eng.lane_probes}
        assert probed == {srcs[0] for _, srcs, _ in grammar_truth.GRAMMAR_LANES}

    def test_the_probe_is_bounded_by_the_time_cutoff(self):
        """Without the upper bound the probe would find rows the window still covers and
        floor on them forever."""
        eng = _Engine(cursor_rows=_cursor_rows(), delete_rowcounts=[0])
        _run(eng, respect_cursor_floor=True)
        assert all("time_cutoff" in p for p in eng.lane_probes)

    def test_a_missing_cursor_row_imposes_no_floor(self):
        eng = _Engine(cursor_rows=[], delete_rowcounts=[0])
        state = _run(eng, respect_cursor_floor=True)
        assert state.cursor_floor_applied is False
        assert eng.deletes

    def test_a_null_cursor_imposes_no_floor(self):
        """substrate-runtime seeds a new lane at the TAIL, so it never wants history."""
        eng = _Engine(
            cursor_rows=_cursor_rows(chat_grammar_consumer=None), delete_rowcounts=[0]
        )
        state = _run(eng, respect_cursor_floor=True)
        assert state.cursor_floor_applied is False

    def test_an_unreachable_cursor_table_SKIPS_rather_than_deleting(self):
        """Fail safe. Unknown floor must never fall back to the time cutoff."""
        eng = _Engine(cursor_rows_raises=RuntimeError("relation does not exist"))
        state = _run(eng, respect_cursor_floor=True)
        assert state.failure_reason == "cursor_floor_unresolved"
        assert eng.deletes == [], "must not delete when the floor is unknown"

    def test_a_failing_lane_probe_SKIPS_rather_than_deleting(self):
        eng = _Engine(
            cursor_rows=_cursor_rows(),
            lane_probe_raises=RuntimeError("canceling statement due to statement timeout"),
        )
        state = _run(eng, respect_cursor_floor=True)
        assert state.failure_reason == "cursor_floor_unresolved"
        assert eng.deletes == []

    def test_an_unexpected_cursor_type_SKIPS_rather_than_deleting(self):
        eng = _Engine(cursor_rows=[{"cursor_name": "chat_grammar_consumer",
                                    "last_event_created_at": "not-a-datetime"}])
        state = _run(eng, respect_cursor_floor=True)
        assert state.failure_reason == "cursor_floor_unresolved"
        assert eng.deletes == []

    def test_the_floor_is_off_for_tables_no_cursor_consumes(self):
        eng = _Engine(cursor_rows=_cursor_rows(), delete_rowcounts=[0])
        state = _run(eng, table="grammar_edges", id_column="edge_id")
        assert state.cursor_floor_applied is False
        assert eng.lane_probes == [], "non-queue tables must not probe lanes at all"

    def test_only_grammar_events_opts_in(self):
        import inspect

        assert "respect_cursor_floor=True" in inspect.getsource(
            grammar_truth.apply_grammar_events_retention
        )
        for fn in (
            grammar_truth.apply_grammar_edges_retention,
            grammar_truth.apply_grammar_atoms_retention,
            grammar_truth.apply_substrate_organ_emissions_retention,
        ):
            assert "respect_cursor_floor" not in inspect.getsource(fn)

    def test_the_delete_boundary_is_strict(self):
        """`<` not `<=`. The consumer reads `created_at > cursor_ts OR (= AND event_id >)`,
        so the row exactly at the floor and its tie-broken tail must survive."""
        sql = " ".join(str(grammar_truth._batch_delete_sql("grammar_events", "event_id")).split())
        assert "created_at < :cutoff" in sql
        assert "<=" not in sql


class TestTheLaneTableMatchesTheRealConsumers:
    """GRAMMAR_LANES is duplicated in sql-writer on purpose -- importing
    orion/substrate/*/constants.py executes orion/substrate/__init__.py, which drags the
    graph-DB store into this thin writer (that exact mistake crash-looped two services on
    2026-08-19). Duplication is only safe if drift is a failing test, so: this is it.

    A drifted lane is silent data loss -- retention would stop protecting a lane whose
    filter no longer matches, and nothing else would notice."""

    def test_every_lane_filter_matches_its_real_consumer(self):
        """Assert ALL FIVE lanes, sources included.

        An earlier version of this test only spot-checked chat and route, and a first draft
        of GRAMMAR_LANES consequently shipped `("orion-cortex-exec",)` for the execution
        lane when the real filter is a three-service frozenset. That drift is invisible at
        runtime: the floor probe simply finds fewer unconsumed rows and reports the lane
        clear, so retention deletes events a stalled reducer still owed."""
        from orion.substrate.chat_loop.constants import (
            CHAT_GRAMMAR_CURSOR_NAME,
            CHAT_SOURCE_SERVICE,
            CHAT_TRACE_PREFIX,
        )
        from orion.substrate.route_loop.constants import (
            ROUTE_GRAMMAR_CURSOR_NAME,
            ROUTE_SOURCE_SERVICE,
            ROUTE_TRACE_PREFIX,
        )
        from orion.substrate.biometrics_loop.constants import GRAMMAR_CURSOR_NAME
        from orion.substrate.execution_loop.constants import (
            EXECUTION_GRAMMAR_CURSOR_NAME,
            EXECUTION_SOURCE_SERVICES,
            EXECUTION_TRACE_PREFIX,
        )
        from orion.substrate.transport_loop.constants import TRANSPORT_GRAMMAR_CURSOR_NAME

        lanes = {name: (set(srcs), pfx) for name, srcs, pfx in grammar_truth.GRAMMAR_LANES}

        assert lanes[CHAT_GRAMMAR_CURSOR_NAME] == ({CHAT_SOURCE_SERVICE}, CHAT_TRACE_PREFIX)
        assert lanes[ROUTE_GRAMMAR_CURSOR_NAME] == ({ROUTE_SOURCE_SERVICE}, ROUTE_TRACE_PREFIX)
        assert lanes[EXECUTION_GRAMMAR_CURSOR_NAME] == (
            set(EXECUTION_SOURCE_SERVICES),
            EXECUTION_TRACE_PREFIX,
        )
        assert set(lanes) == {
            CHAT_GRAMMAR_CURSOR_NAME,
            ROUTE_GRAMMAR_CURSOR_NAME,
            GRAMMAR_CURSOR_NAME,
            EXECUTION_GRAMMAR_CURSOR_NAME,
            TRANSPORT_GRAMMAR_CURSOR_NAME,
        }, "a lane was added or renamed in substrate-runtime and not mirrored here"

    def test_the_two_inline_lanes_match_substrate_runtime_source(self):
        """biometrics and transport pass their filters as inline literals in
        substrate-runtime's fetch_* methods rather than via a shared constant, so there is
        nothing to import -- assert against the source text instead of trusting a copy."""
        import pathlib

        repo_root = pathlib.Path(__file__).resolve().parents[3]
        store = repo_root / "services" / "orion-substrate-runtime" / "app" / "store.py"
        assert store.is_file(), f"substrate-runtime store not found at {store}"
        src = store.read_text()
        lanes = {name: (set(srcs), pfx) for name, srcs, pfx in grammar_truth.GRAMMAR_LANES}

        assert 'source_services=("orion-biometrics",)' in src
        assert 'trace_prefix="biometrics.node:"' in src
        assert lanes["biometrics_grammar_consumer"] == ({"orion-biometrics"}, "biometrics.node:")

        assert 'source_services=("orion-bus",)' in src
        assert 'trace_prefix="bus.transport:"' in src
        assert lanes["transport_grammar_reducer"] == ({"orion-bus"}, "bus.transport:")


class TestTheDebtNumberCannotReadCalmWhileTheFloorPins:
    def test_debt_is_measured_against_the_window_not_the_clamped_cutoff(self):
        """The instrument has to mean "rows past the retention window", not "rows I was
        allowed to touch". Measuring against the clamped cutoff makes /health report 0 in
        exactly the state that needs to scream."""
        owed = datetime.now(timezone.utc) - timedelta(days=8)
        eng = _Engine(
            cursor_rows=_cursor_rows(),
            lane_oldest={"chat_grammar_consumer": owed},
            debt=999,
            delete_rowcounts=[0],
        )
        state = _run(eng, respect_cursor_floor=True)
        assert state.cursor_floor_applied is True
        assert eng.debt_params and eng.debt_params[0]["cutoff"] != owed, (
            "debt must not be counted against the floor"
        )
        assert state.remaining_debt == 999

    def test_a_timeout_counting_debt_does_not_fail_the_run(self):
        eng = _Engine(cursor_rows=_cursor_rows(),
                      debt=RuntimeError("canceling statement due to statement timeout"),
                      delete_rowcounts=[1000, 1000, 1000])
        state = _run(eng, respect_cursor_floor=True)
        assert state.rows_pruned_last_run == 3000
        assert state.failure_reason is None
        assert state.remaining_debt is None
        assert "timeout" in (state.debt_count_failed_reason or "")


class TestTheHealthSurfaceShowsTheFloor:
    def test_the_new_fields_are_actually_surfaced(self):
        import inspect

        src = inspect.getsource(grammar_truth._retention_block)
        for field in ("cursor_floor_at", "cursor_floor_applied", "debt_count_failed_reason"):
            assert field in src, f"{field} added to the state but never reported"


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
