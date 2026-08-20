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
from types import SimpleNamespace

import pytest

from app import grammar_truth
from app.grammar_truth import GrammarRetentionState
from app.settings import Settings, get_settings


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

    @pytest.mark.parametrize(
        "fn_name,expected",
        [
            ("apply_grammar_events_retention", True),
            ("apply_grammar_traces_retention", True),
            ("apply_grammar_edges_retention", False),
            ("apply_grammar_atoms_retention", False),
            ("apply_substrate_organ_emissions_retention", False),
        ],
    )
    def test_exactly_the_cursor_coupled_tables_opt_in(self, monkeypatch, fn_name, expected):
        """grammar_events opts in because reducers consume it directly; grammar_traces
        because a stalled lane must hold its parent rows back too. The leaf tables no
        cursor touches must not pay for a probe they cannot use.

        ASSERTS THE VALUE ACTUALLY PASSED, not the function's source text. The source-text
        version of this test was defeated by the patch that added it: the new function's
        own DOCSTRING contained the literal `respect_cursor_floor=True`, so flipping the
        real keyword argument to False left all tests green (mutation-tested, confirmed
        by review). A gate that a comment can satisfy is not a gate.
        """
        seen = {}

        def capture(**kw):
            seen.update(kw)
            return GrammarRetentionState()

        monkeypatch.setattr(grammar_truth, "_apply_bounded_table_retention", capture)
        getattr(grammar_truth, fn_name)(3)
        assert seen.get("respect_cursor_floor", False) is expected, seen.get("table")

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
            grammar_traces_retention_days = 3

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
            grammar_traces_retention_days = 3

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
            "GRAMMAR_TRACES_RETENTION_DAYS",
        ):
            assert f'Field(3, alias="{key}")' in src or f'3, alias="{key}"' in src, key


class TestTheParentTraceRowIsPrunedToo:
    """grammar_traces had NO retention while its children were pruned at 3 days. Measured
    live 2026-08-20: 205,465 of 487,970 traces (42%) already had zero atoms, so the Grammar
    Atlas was listing traces that expanded into empty graphs -- CLAUDE.md's "UI panel
    rendered with no real backing artifact", not merely wasted disk."""

    def test_grammar_traces_is_in_the_retention_cycle(self):
        assert "grammar_traces" in dict(grammar_truth.GRAMMAR_RETENTION_TABLES)

    def test_the_parent_is_pruned_after_its_children(self):
        """No FK enforces this -- the live DB has zero foreign keys on grammar_* (the
        migration file declares them but SQLAlchemy created the tables first). So order is
        a correctness argument this code has to make for itself."""
        order = [t for t, _ in grammar_truth.GRAMMAR_RETENTION_TABLES]
        # RELATIVE order, not "last in the tuple". The first version of this asserted
        # order[-1] == "grammar_traces", which is a different and weaker claim that happened
        # to hold when grammar_traces was the newest entry. It broke the moment an unrelated
        # table (substrate_proposal_frames, different engine, no ordering relationship with
        # the grammar lane at all) was appended -- a false failure that says nothing about
        # the invariant this test exists to protect.
        for child in ("grammar_events", "grammar_atoms", "grammar_edges"):
            assert order.index(child) < order.index("grammar_traces"), child

    def test_it_deletes_by_trace_id_not_some_other_column(self):
        """The PK is trace_id, not an id/event_id column. A copy-paste of a sibling's
        id_column would raise at runtime on the first batch, per cycle, forever."""
        import inspect

        src = inspect.getsource(grammar_truth.apply_grammar_traces_retention)
        assert 'table="grammar_traces"' in src
        assert 'id_column="trace_id"' in src

    def test_the_loop_actually_passes_a_window_for_it(self, monkeypatch):
        """A table registered in GRAMMAR_RETENTION_TABLES but missing from days_for is
        silently skipped by run_one_retention_cycle (days <= 0 means disabled). Wiring one
        half without the other looks exactly like working retention in the logs."""
        from app import grammar_retention_loop as mod

        class S:
            grammar_events_retention_days = 3
            grammar_edges_retention_days = 3
            grammar_atoms_retention_days = 3
            substrate_organ_emissions_retention_days = 3
            grammar_traces_retention_days = 3

        days = mod.retention_days_for(S())
        registered = {t for t, _ in grammar_truth.GRAMMAR_RETENTION_TABLES}
        assert registered <= set(days), f"registered but never given a window: {registered - set(days)}"
        assert days["grammar_traces"] == 3

    def test_a_stalled_lane_holds_the_parent_back_too(self):
        """The direction that matters: if a reducer still owes events, their trace rows
        must survive with them. Deleting the parent while holding the children would be
        worse than deleting neither."""
        owed = datetime.now(timezone.utc) - timedelta(days=8)
        eng = _Engine(
            cursor_rows=_cursor_rows(),
            lane_oldest={"chat_grammar_consumer": owed},
            delete_rowcounts=[0],
        )
        state = _run(eng, table="grammar_traces", id_column="trace_id", respect_cursor_floor=True)
        assert state.cursor_floor_applied is True
        assert state.cutoff_at == owed

    def test_every_managed_table_has_a_real_settings_window(self):
        """THE gate behind _other_retention_truth_blocks. That function reads
        `<table>_retention_days` off Settings for every _EXTRA_RETENTION_TABLES entry; a
        table added to one and not the other used to KeyError and take the ENTIRE /health
        endpoint down with it. It now degrades instead of raising, which means the failure
        is quiet -- so the loud part has to live here.

        Deliberately against the real Settings CLASS, not a fixture. The MagicMock in
        test_grammar_truth.py answers hasattr() for any name and now auto-populates these
        attributes, so it cannot catch this and must not be trusted to."""
        from app.settings import Settings

        for table in grammar_truth._EXTRA_RETENTION_TABLES:
            attr = f"{table}_retention_days"
            assert attr in Settings.model_fields, f"{table} has no {attr} on Settings"

    def test_a_missing_window_degrades_health_instead_of_500ing_it(self):
        """The runtime backstop for the test above. /health has no try/except around this."""

        class _Settings:
            grammar_events_retention_batch_size = 1000
            grammar_events_retention_max_batches_per_startup = 3
            grammar_events_retention_max_elapsed_sec = 20.0

        for table in grammar_truth._EXTRA_RETENTION_TABLES:
            setattr(_Settings, f"{table}_retention_days", 3)
        delattr(_Settings, "grammar_traces_retention_days")

        blocks = grammar_truth._other_retention_truth_blocks(_Settings())
        assert set(blocks) == set(grammar_truth._EXTRA_RETENTION_TABLES)
        assert blocks["grammar_traces"]["configured_days"] == 0
        assert "grammar_traces" in grammar_truth._retention_window_config_missing
        grammar_truth._retention_window_config_missing.clear()

    def test_startup_only_covers_the_pre_periodic_tables_on_purpose(self):
        """main.py hand-lists its startup retention blocks. grammar_traces is deliberately
        NOT among them: the periodic loop covers it 60s after boot, and main.py's startup
        pass already blocks the event loop for ~260s across four tables -- adding a fifth
        copy makes a known problem worse to save one cycle.

        This test exists so that stays a DECISION. A sixth table added to
        GRAMMAR_RETENTION_TABLES will fail here until someone states which side it is on."""
        import pathlib

        main_src = (
            pathlib.Path(__file__).resolve().parents[1] / "app" / "main.py"
        ).read_text()
        registered = [t for t, _ in grammar_truth.GRAMMAR_RETENTION_TABLES]
        # Both of these are deliberately periodic-only. The startup pass already blocks the
        # event loop for ~260s; every table added to it makes that worse, and the periodic
        # loop reaches the same steady state within a minute of boot anyway.
        periodic_only = {"grammar_traces", "substrate_proposal_frames"}
        for table in registered:
            covered = f"apply_{table}_retention" in main_src
            if table in periodic_only:
                assert not covered, f"{table} gained a startup block; update this test"
            else:
                assert covered, f"{table} lost its startup block in main.py"

    def test_the_atlas_listing_and_the_delete_both_have_an_index(self):
        """Two different queries, two different indexes. Without (started_at desc) the Atlas
        seq-scans the whole table on every page load -- measured live at 11,523 blocks /
        58.8 ms against 487,970 rows, with the trace_id PK as the table's ONLY index."""
        import pathlib

        repo = pathlib.Path(__file__).resolve().parents[3]
        sql = (
            repo / "services" / "orion-sql-db"
            / "manual_migration_grammar_traces_retention.sql"
        ).read_text()
        assert "idx_grammar_traces_created_at" in sql
        assert "on grammar_traces (created_at, trace_id)" in sql
        assert "idx_grammar_traces_started_at" in sql
        assert "on grammar_traces (started_at desc, trace_id)" in sql
        assert "concurrently" in sql, "a blocking index build on a live 143 MB table"


class TestSubstrateProposalFramesIsBounded:
    """substrate_proposal_frames had no retention at all until 2026-08-20.

    Live at that point: 474,230 rows / 1,758 MB, oldest 2026-07-23, growing ~27k rows and
    ~105 MB a day with nothing to stop it.
    """

    def test_it_is_in_the_retention_cycle(self):
        tables = [name for name, _ in grammar_truth.GRAMMAR_RETENTION_TABLES]
        assert "substrate_proposal_frames" in tables

    def test_it_is_floored_on_the_pipeline_markers_not_the_grammar_cursor(self, monkeypatch):
        """The two floors are not interchangeable and must not be swapped by accident.

        The grammar cursor floor probes substrate_reduction_cursor and grammar_events, which
        say nothing at all about whether a substrate pipeline stage still owes work on a
        proposal row. Wiring the wrong one here would report "nothing owed" and delete a live
        backlog.
        """
        seen = {}

        def capture(**kw):
            seen.update(kw)
            return GrammarRetentionState()

        monkeypatch.setattr(grammar_truth, "_apply_bounded_table_retention", capture)
        grammar_truth.apply_substrate_proposal_frames_retention(7)
        assert seen.get("table") == "substrate_proposal_frames"
        assert seen.get("id_column") == "frame_id"
        assert seen.get("floor_resolver") is grammar_truth._substrate_chain_floor
        assert seen.get("respect_cursor_floor", False) is False

    def test_it_uses_the_plain_engine_not_the_grammar_engine(self, monkeypatch):
        """Substrate data lives on the default engine; the grammar engine is a separate lane
        with its own 10s statement timeout."""
        seen = {}

        def capture(**kw):
            seen.update(kw)
            return GrammarRetentionState()

        monkeypatch.setattr(grammar_truth, "_apply_bounded_table_retention", capture)
        grammar_truth.apply_substrate_proposal_frames_retention(7)
        assert seen.get("engine") is grammar_truth.default_engine
        assert seen.get("engine") is not grammar_truth.grammar_engine

    def test_the_loop_actually_passes_a_window_for_it(self):
        from app import grammar_retention_loop

        days = grammar_retention_loop.retention_days_for(get_settings())
        assert days.get("substrate_proposal_frames", 0) > 0, days

    def test_it_has_a_real_settings_window(self):
        assert "substrate_proposal_frames_retention_days" in Settings.model_fields


class TestTheSubstrateChainFloor:
    """The floor asks the pipeline's pending markers, not the clock.

    manual_migration_substrate_pending_markers.sql records that a time-bounded version of
    this idea was tried and REVERTED, because the dispatch->feedback hop legitimately ran
    p50 34.6 hours and max 11.3 days behind. Retention must not put that bound back.
    """

    def test_it_covers_every_stage_that_can_reach_back_into_the_table(self):
        stages = {t for _, t, _ in grammar_truth._SUBSTRATE_CHAIN_PENDING}
        assert stages == {
            "substrate_proposal_frames",
            "substrate_policy_decision_frames",
            "substrate_execution_dispatch_frames",
        }

    def test_it_probes_created_at_not_generated_at(self):
        """Same clock as the delete predicate.

        The delete keys on created_at. generated_at is when the stage produced the frame;
        created_at is when the row was written. Measured live 2026-08-20 they diverge by up
        to 724s, and 24 policy rows have created_at strictly BEFORE generated_at. A floor
        read off one clock and compared to a cutoff on the other deletes rows it meant to
        keep.
        """
        seen = []

        class _Conn:
            def execute(self, stmt, *a, **kw):
                seen.append(str(stmt))
                return SimpleNamespace(scalar=lambda: None)

        grammar_truth._substrate_chain_floor(_Conn(), datetime.now(timezone.utc))
        assert seen, "floor made no probes at all"
        for sql in seen:
            assert "MIN(created_at)" in sql, sql
            assert "generated_at" not in sql, sql

    def test_the_oldest_pending_row_across_all_stages_wins(self):
        oldest = datetime(2026, 8, 1, tzinfo=timezone.utc)
        answers = [
            datetime(2026, 8, 10, tzinfo=timezone.utc),
            oldest,
            datetime(2026, 8, 5, tzinfo=timezone.utc),
        ]

        class _Conn:
            def __init__(self):
                self.i = 0

            def execute(self, stmt, *a, **kw):
                value = answers[self.i]
                self.i += 1
                return SimpleNamespace(scalar=lambda: value)

        floor, resolved = grammar_truth._substrate_chain_floor(
            _Conn(), datetime.now(timezone.utc)
        )
        assert resolved is True
        assert floor == oldest

    def test_all_stages_caught_up_imposes_no_floor(self):
        """(None, True) means "nothing owed", which is NOT the same as "unknown"."""

        class _Conn:
            def execute(self, stmt, *a, **kw):
                return SimpleNamespace(scalar=lambda: None)

        floor, resolved = grammar_truth._substrate_chain_floor(
            _Conn(), datetime.now(timezone.utc)
        )
        assert floor is None
        assert resolved is True

    def test_a_naive_timestamp_from_the_driver_is_treated_as_utc(self):
        """A tz-naive datetime compared against a tz-aware cutoff raises TypeError at the
        comparison, which would abort the whole retention run rather than floor it."""
        naive = datetime(2026, 8, 1, 12, 0, 0)

        class _Conn:
            def execute(self, stmt, *a, **kw):
                return SimpleNamespace(scalar=lambda: naive)

        floor, resolved = grammar_truth._substrate_chain_floor(
            _Conn(), datetime.now(timezone.utc)
        )
        assert resolved is True
        assert floor is not None and floor.tzinfo is not None
        assert floor < datetime.now(timezone.utc)  # the comparison the caller makes

    def test_a_failed_probe_reports_unresolved_and_never_no_floor(self):
        """The dangerous confusion is (None, False) collapsing into (None, True): "I could
        not ask" must not read as "nothing is owed", or a database blip deletes the backlog.
        """

        class _Conn:
            def execute(self, stmt, *a, **kw):
                raise RuntimeError("connection reset")

        floor, resolved = grammar_truth._substrate_chain_floor(
            _Conn(), datetime.now(timezone.utc)
        )
        assert floor is None
        assert resolved is False

    def test_an_unresolved_floor_refuses_to_prune(self, monkeypatch):
        """End to end through the real _apply_bounded_table_retention: an unresolved floor
        must skip the cycle, not fall through to the time cutoff."""
        deleted = []

        class _Conn:
            def execute(self, stmt, *a, **kw):
                deleted.append(str(stmt))
                return SimpleNamespace(rowcount=0, scalar_one=lambda: 0, scalar=lambda: 0)

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        class _Engine:
            def connect(self):
                return _Conn()

            def begin(self):
                return _Conn()

        monkeypatch.setattr(
            grammar_truth, "_substrate_chain_floor", lambda conn, cutoff: (None, False)
        )
        state = grammar_truth._apply_bounded_table_retention(
            engine=_Engine(),
            table="substrate_proposal_frames",
            id_column="frame_id",
            retention_days=7,
            batch_size=100,
            max_batches=5,
            max_elapsed_sec=10.0,
            floor_resolver=grammar_truth._substrate_chain_floor,
        )
        assert state.failure_reason == "cursor_floor_unresolved"
        assert not any("DELETE" in sql.upper() for sql in deleted), deleted

    def test_a_binding_floor_clamps_the_cutoff_back(self, monkeypatch):
        """A stage genuinely behind the retention window pulls the cutoff back to it."""
        behind = datetime.now(timezone.utc) - timedelta(days=30)

        class _Conn:
            def execute(self, stmt, *a, **kw):
                return SimpleNamespace(rowcount=0, scalar_one=lambda: 0, scalar=lambda: 0)

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        class _Engine:
            def connect(self):
                return _Conn()

            def begin(self):
                return _Conn()

        state = grammar_truth._apply_bounded_table_retention(
            engine=_Engine(),
            table="substrate_proposal_frames",
            id_column="frame_id",
            retention_days=7,
            batch_size=100,
            max_batches=1,
            max_elapsed_sec=10.0,
            floor_resolver=lambda conn, cutoff: (behind, True),
        )
        assert state.cursor_floor_applied is True
        assert state.cutoff_at == behind


def test_the_grammar_cursor_floor_also_normalises_a_naive_timestamp():
    """Pre-existing gap, found by mutation-testing the NEW floor and hitting this one instead.

    _grammar_events_cursor_floor has the identical `if oldest.tzinfo is None` normalisation,
    and deleting it left the entire suite green (403 passed, only the 11 unrelated
    pre-existing failures). Without it the floor returns a naive datetime, and the caller's
    `floor < cutoff` comparison against a tz-aware cutoff raises TypeError -- which aborts
    the whole retention run rather than flooring it. Same one-line bug, same blast radius,
    now pinned in both places.
    """
    naive = datetime(2026, 8, 1, 12, 0, 0)

    class _Conn:
        def execute(self, stmt, *a, **kw):
            sql = str(stmt)
            if "substrate_reduction_cursor" in sql:
                rows = [
                    {"cursor_name": name, "last_event_created_at": naive}
                    for name, _, _ in grammar_truth.GRAMMAR_LANES
                ]
                return _Result(rows=rows)
            return _Result(scalar=naive)

    floor, resolved = grammar_truth._grammar_events_cursor_floor(
        _Conn(), datetime.now(timezone.utc)
    )
    assert resolved is True
    assert floor is not None, "fixture did not reach the branch under test"
    assert floor.tzinfo is not None
    assert floor < datetime.now(timezone.utc)  # the comparison the caller makes
