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

    def test_no_table_has_a_startup_retention_block_any_more(self):
        """The periodic loop is the ONLY retention path, as of 2026-08-20.

        This replaces a test that pinned WHICH tables were startup-exempt. That framing was
        wrong: it treated a drift between two hand-maintained lists as a thing to document
        rather than a thing to delete. main.py hand-listed four tables while
        GRAMMAR_RETENTION_TABLES had six, so two were exempt purely by omission.

        The four startup blocks ran synchronously on the event loop AHEAD of the bus
        subscription -- measured at ~260s of not consuming events on every restart -- and
        could not converge against continuous arrival anyway, which is the entire reason the
        periodic loop exists. One path, and this test is what keeps it one.
        """
        import pathlib as _pathlib

        main_src = (
            _pathlib.Path(__file__).resolve().parents[1] / "app" / "main.py"
        ).read_text()
        for table, _ in grammar_truth.GRAMMAR_RETENTION_TABLES:
            assert f"apply_{table}_retention" not in main_src, (
                f"{table} has a startup retention block in main.py again. Retention belongs "
                f"in the periodic loop; a startup pass blocks the bus subscription behind it."
            )

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

    @staticmethod
    def _probes():
        seen = []

        class _Conn:
            def execute(self, stmt, *a, **kw):
                seen.append(" ".join(str(stmt).split()))
                return SimpleNamespace(scalar=lambda: None)

        grammar_truth._substrate_chain_floor(_Conn(), datetime.now(timezone.utc))
        assert seen, "floor made no probes at all"
        return seen

    def test_it_covers_every_stage_that_can_reach_back_into_the_table(self):
        stages = {name for name, _ in grammar_truth._SUBSTRATE_CHAIN_PENDING}
        assert stages == {"proposal->policy", "policy->dispatch", "dispatch->feedback"}
        sql = " ".join(self._probes())
        for table in (
            "substrate_proposal_frames",
            "substrate_policy_decision_frames",
            "substrate_execution_dispatch_frames",
        ):
            assert table in sql, table

    def test_every_probe_returns_a_proposal_timestamp_not_the_pending_rows_own(self):
        """THE bug code review caught on the first version of this floor.

        The delete removes substrate_proposal_frames rows by their own created_at. A pending
        row in a DOWNSTREAM table is always NEWER than the proposal it needs -- it is a
        child, written later -- so flooring at the child's timestamp leaves the parent below
        the floor and deletable, which is backwards from the safety being claimed. Measured
        live 2026-08-20 over 3 days: dispatch rows are created a mean 123.2s and max 920.2s
        AFTER their parent proposal; 0 of 55,573 were created before it.

        The two downstream probes must therefore JOIN back to substrate_proposal_frames
        through source_proposal_frame_id and take MIN of the PARENT's created_at.
        """
        for stage, sql in grammar_truth._SUBSTRATE_CHAIN_PENDING:
            flat = " ".join(sql.split())
            if stage == "proposal->policy":
                # The proposal row IS the parent here; no join needed.
                assert "MIN(s.created_at)" in flat, flat
                continue
            assert "MIN(p.created_at)" in flat, stage
            assert "JOIN substrate_proposal_frames p" in flat, stage
            assert "p.frame_id = d.source_proposal_frame_id" in flat, stage

    def test_it_probes_created_at_not_generated_at(self):
        """Same clock as the delete predicate.

        The delete keys on created_at. generated_at is when the stage produced the frame;
        created_at is when the row was written. Measured live 2026-08-20 they diverge by up
        to 724s, and 24 policy rows have created_at strictly BEFORE generated_at. A floor
        read off one clock and compared to a cutoff on the other deletes rows it meant to
        keep.
        """
        for sql in self._probes():
            assert "MIN(" in sql and "created_at)" in sql, sql
            assert "generated_at" not in sql, sql

    def test_every_probe_fences_the_min_aggregate_with_offset_0(self):
        """Without the fence this is a ~490ms near-full-table scan every 60 seconds.

        Postgres rewrites a bare `SELECT MIN(created_at) ... WHERE dispatch_pending` into
        `ORDER BY created_at LIMIT 1` over the FULL created_at index with the marker as a
        filter. Pending rows are always the NEWEST rows, so the ascending scan discards the
        entire table first -- measured live: 490 ms, 102,343 buffers, "Rows Removed by
        Filter: 474,708". `OFFSET 0` is an optimisation fence that forces the tiny pending
        set to be materialised through the partial index first. With it: 0.18/5.4/1.6 ms.

        The cost of getting this wrong is invisible in the obvious place: it is HIGHEST when
        the pipeline is caught up and falls as a backlog develops, so it would never look
        like a backlog problem.
        """
        for sql in self._probes():
            assert "OFFSET 0" in sql, sql

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

    def test_an_unresolved_floor_refuses_to_prune(self):
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

        state = grammar_truth._apply_bounded_table_retention(
            engine=_Engine(),
            table="substrate_proposal_frames",
            id_column="frame_id",
            retention_days=7,
            batch_size=100,
            max_batches=5,
            max_elapsed_sec=10.0,
            floor_resolver=lambda conn, cutoff: (None, False),
        )
        assert state.failure_reason == "cursor_floor_unresolved"
        # NOT `assert no DELETE was issued` -- when the floor is unresolved the function
        # returns before touching the connection at all, so that assertion is trivially true
        # and proves nothing. The load-bearing checks are the failure_reason above and the
        # fact that no cutoff was ever established.
        assert state.cutoff_at is None
        assert deleted == [], deleted

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


# Six named tables, fixed. The budget tests assert exact fair-share arithmetic
# (45.0 / 6, and the [3.0, 3.75, 5.0, 7.5, 15.0] cascade after a first-table overrun),
# so they need a stable divisor. The first entry keeps its real name because those tests
# drive an overrun through `costs={"grammar_events": 30.0}`.
BUDGET_FIXTURE_TABLES = (
    "grammar_events",
    "fixture_table_2",
    "fixture_table_3",
    "fixture_table_4",
    "fixture_table_5",
    "fixture_table_6",
)


class TestTheCycleBudget:
    """`max_elapsed_sec` bounds one table. Nothing bounded the whole cycle.

    Six tables x a 20s per-table cap is a 120s cycle on a 60s timer. The gap was invisible
    because at 3 batches every table finishes in about a second, so the per-table cap never
    binds and the cycle never approaches its worst case. That is exactly the kind of bound
    that is missing until the day it matters.
    """

    @staticmethod
    def _cycle(monkeypatch, *, costs, tables=None, **kwargs):
        """Run a real cycle with fake per-table work of known duration.

        Returns [(table, elapsed_cap_it_was_given), ...] in call order.

        `tables` defaults to the live GRAMMAR_RETENTION_TABLES. The budget-arithmetic
        tests below pass BUDGET_FIXTURE_TABLES instead, because what they assert is how
        the splitter divides a cycle -- not how many tables the service happens to
        manage today. Reading the live registry there made every one of them a tripwire
        that fired when orion_biometrics_cluster was added, which is a false failure:
        the algorithm was unchanged and correct.
        """
        calls = []
        clock = {"t": 0.0}
        # Rebind grammar_truth's module-local `time` name, NOT `grammar_truth.time.monotonic`
        # -- the latter mutates the shared `time` module object process-wide for the duration
        # of the test. monkeypatch does restore it, and no leakage was observed, but a global
        # mutation is the wrong seam when a local rebind costs the same.
        monkeypatch.setattr(
            grammar_truth, "time", SimpleNamespace(monotonic=lambda: clock["t"])
        )

        def make(table):
            def fn(days, *, max_batches, max_elapsed_sec):
                calls.append((table, max_elapsed_sec))
                clock["t"] += costs.get(table, 0.0)
                return GrammarRetentionState()
            return fn

        tables = list(tables) if tables is not None else [
            t for t, _ in grammar_truth.GRAMMAR_RETENTION_TABLES
        ]
        monkeypatch.setattr(
            grammar_truth, "GRAMMAR_RETENTION_TABLES",
            tuple((t, make(t)) for t in tables),
        )
        grammar_truth.run_one_retention_cycle(
            days_for={t: 3 for t in tables},
            max_batches=3,
            **kwargs,
        )
        return calls

    def test_without_a_cycle_budget_every_table_gets_the_full_per_table_cap(self, monkeypatch):
        """The old behaviour, kept reachable: max_cycle_elapsed_sec=None is opt-out."""
        calls = self._cycle(monkeypatch, costs={}, max_elapsed_sec=20.0,
                            max_cycle_elapsed_sec=None)
        assert calls, "no tables ran"
        assert {cap for _, cap in calls} == {20.0}

    def test_the_budget_is_split_fairly_not_first_come(self, monkeypatch):
        """The starvation bug this exists to prevent.

        grammar_events is FIRST in GRAMMAR_RETENTION_TABLES and the only table carrying real
        debt (4,134,774 rows live 2026-08-20, versus 0 for the rest). Under a first-come
        budget it takes the whole cycle every cycle and the five behind it never run -- which
        reads as healthy in the logs, because grammar_events' own numbers look fine.
        """
        calls = self._cycle(monkeypatch, costs={}, max_elapsed_sec=20.0,
                            max_cycle_elapsed_sec=45.0, tables=BUDGET_FIXTURE_TABLES)
        n = len(calls)
        assert n == 6, [t for t, _ in calls]
        # First table may claim at most its fair share, never the whole budget.
        first_cap = calls[0][1]
        assert first_cap == pytest.approx(45.0 / n), calls
        assert first_cap < 20.0, "fair share should bind before the per-table cap here"
        # Every table got a real, positive slice.
        assert all(cap > 0 for _, cap in calls), calls

    def test_the_share_is_recomputed_from_what_is_left_not_fixed_up_front(self, monkeypatch):
        """A slow early table shrinks the NEXT share; shares then grow as the divisor falls.

        The naive reading -- "every later table gets less" -- is wrong, and asserting it was
        my own first mistake here. remaining_budget/remaining_tables recovers: after a 30s
        overrun of a 45s budget, 15s is left and the shares run 3.0, 3.75, 5.0, 7.5, 15.0 as
        the divisor drops 5, 4, 3, 2, 1. That is the correct behaviour -- the budget is a
        bound on the CYCLE, so an early overrun must not permanently penalise every table
        behind it, only redistribute what remains.
        """
        calls = self._cycle(
            monkeypatch,
            costs={"grammar_events": 30.0},
            max_elapsed_sec=20.0,
            max_cycle_elapsed_sec=45.0,
            tables=BUDGET_FIXTURE_TABLES,
        )
        assert len(calls) == 6, calls
        assert calls[0][0] == "grammar_events"
        assert calls[0][1] == pytest.approx(45.0 / 6)
        later = [cap for _, cap in calls[1:]]
        assert later == pytest.approx([3.0, 3.75, 5.0, 7.5, 15.0]), calls
        # No table is ever handed a zero or negative slice, and none exceeds the per-table cap.
        assert all(0 < cap <= 20.0 for _, cap in calls), calls

    def test_an_exhausted_budget_skips_loudly_rather_than_silently(self, monkeypatch, caplog):
        """A table dropped from a cycle with no log line is indistinguishable from a table
        with no debt. That is the failure mode the whole periodic loop was built to end."""
        with caplog.at_level("WARNING"):
            calls = self._cycle(
                monkeypatch,
                costs={"grammar_events": 100.0},
                max_elapsed_sec=20.0,
                max_cycle_elapsed_sec=45.0,
                tables=BUDGET_FIXTURE_TABLES,
            )
        assert len(calls) == 1, calls
        assert caplog.text.count("grammar_retention_cycle_budget_exhausted") == 5
        # Each skipped table must be named. An earlier version of this wrote
        # `assert f"table={table}" in caplog.text or True`, which is unconditionally true --
        # so the one thing this test exists to check (that a dropped table is identifiable,
        # not just counted) was unasserted.
        skipped = [t for t, _ in grammar_truth.GRAMMAR_RETENTION_TABLES
                   if t != "grammar_events"]
        assert len(skipped) == 5, skipped
        for table in skipped:
            assert f"table={table}" in caplog.text, (table, caplog.text)

    def test_the_per_table_cap_still_wins_when_it_is_the_smaller_one(self, monkeypatch):
        """The budget adds a bound; it must not RELAX the existing one."""
        calls = self._cycle(monkeypatch, costs={}, max_elapsed_sec=2.0,
                            max_cycle_elapsed_sec=600.0)
        assert {cap for _, cap in calls} == {2.0}

    @pytest.mark.parametrize(
        "configured,expected",
        [(45.0, 45.0), (12.5, 12.5), (0.0, None), (-1.0, None)],
    )
    def test_the_loop_passes_the_cycle_budget_through(self, monkeypatch, configured, expected):
        """Asserts the VALUE the loop actually hands to run_one_retention_cycle.

        The first version of this asserted `"max_cycle_elapsed_sec=max_cycle_elapsed" in
        inspect.getsource(...)` -- the exact source-text idiom this file already documents as
        defeated once, when a new function's own docstring satisfied the string check. It also
        could not have caught the real bug here, which was the VALUE: `float(x or 45.0)` turns
        a configured 0 into 45 (so the documented opt-out silently did nothing) and lets a
        negative through untouched (so every table skipped every cycle, forever).
        """
        import asyncio

        from app import grammar_retention_loop as mod

        assert "grammar_retention_periodic_max_cycle_sec" in Settings.model_fields

        seen = {}

        def fake_cycle(**kw):
            seen.update(kw)
            raise asyncio.CancelledError  # one cycle, then unwind

        monkeypatch.setattr(mod, "run_one_retention_cycle", fake_cycle)

        settings = SimpleNamespace(
            grammar_retention_interval_sec=0.001,
            grammar_retention_periodic_max_batches=3,
            grammar_retention_periodic_max_elapsed_sec=20.0,
            grammar_retention_periodic_max_cycle_sec=configured,
            **{f"{t}_retention_days": 3 for t in
               (n for n, _ in grammar_truth.GRAMMAR_RETENTION_TABLES)},
        )
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(mod.grammar_retention_loop(settings))
        assert seen.get("max_cycle_elapsed_sec") == expected, seen

    def test_the_divisor_counts_only_tables_with_a_window(self, monkeypatch):
        """The `eligible` prefilter is what keeps a partially-configured deployment correct.

        If the divisor were len(GRAMMAR_RETENTION_TABLES) instead of the number of tables
        that actually have a window, a deployment with 2 of 6 tables configured would hand
        each of them 1/6 of the budget and quietly leave 2/3 of the cycle unused. Every other
        test here passes days_for for all six, so none of them can see this.
        """
        calls = []
        clock = {"t": 0.0}
        monkeypatch.setattr(
            grammar_truth, "time", SimpleNamespace(monotonic=lambda: clock["t"])
        )

        def make(table):
            def fn(days, *, max_batches, max_elapsed_sec):
                calls.append((table, max_elapsed_sec))
                return GrammarRetentionState()
            return fn

        names = list(BUDGET_FIXTURE_TABLES)
        monkeypatch.setattr(
            grammar_truth, "GRAMMAR_RETENTION_TABLES",
            tuple((t, make(t)) for t in names),
        )
        # Only two of the six have a window. Budget 20 with a per-table cap of 20 so the
        # SHARE is what binds -- at 45.0 the fair share would be 22.5, the per-table cap of
        # 20 would win, and a broken divisor of 6 (giving 7.5) would be indistinguishable
        # from a correct one. Picking a budget where the cap wins is how this test would
        # have passed while proving nothing; that was its first draft.
        days = {names[0]: 3, names[-1]: 3}
        grammar_truth.run_one_retention_cycle(
            days_for=days, max_batches=3, max_elapsed_sec=20.0, max_cycle_elapsed_sec=20.0
        )
        assert [t for t, _ in calls] == [names[0], names[-1]]
        # Correct divisor (2 eligible): 20/2 = 10.0. A divisor of 6 would give 3.33.
        assert calls[0][1] == pytest.approx(10.0), calls
        assert calls[1][1] == pytest.approx(20.0), calls

    @pytest.mark.parametrize("budget", [0.0, -1.0])
    def test_a_non_positive_budget_means_no_bound_not_skip_everything(self, monkeypatch, budget):
        """Getting this backwards turns a config typo into silently-disabled retention whose
        only symptom is a WARNING per table per minute that reads like a transient squeeze."""
        calls = self._cycle(monkeypatch, costs={}, max_elapsed_sec=20.0,
                            max_cycle_elapsed_sec=budget, tables=BUDGET_FIXTURE_TABLES)
        assert len(calls) == 6, calls
        assert {cap for _, cap in calls} == {20.0}

    def test_each_state_records_the_cap_it_was_actually_handed(self, monkeypatch):
        """`/grammar/truth` must report the bound that governed, not the one in settings.

        Live before this field existed, the endpoint showed `max_elapsed_sec: 120.0` next to
        `elapsed_sec: 1.08` for a run whose real cap was 7.5 -- config truth on the endpoint
        whose entire job is runtime truth. An operator raising the batch cap would see
        `capped_by_elapsed_limit: true` beside 120.0 and have no way to find the real number.
        """
        clock = {"t": 0.0}
        monkeypatch.setattr(
            grammar_truth, "time", SimpleNamespace(monotonic=lambda: clock["t"])
        )

        def make(_table):
            def fn(days, *, max_batches, max_elapsed_sec):
                return GrammarRetentionState()
            return fn

        names = list(BUDGET_FIXTURE_TABLES)
        monkeypatch.setattr(
            grammar_truth, "GRAMMAR_RETENTION_TABLES",
            tuple((t, make(t)) for t in names),
        )
        out = grammar_truth.run_one_retention_cycle(
            days_for={t: 3 for t in names},
            max_batches=3,
            max_elapsed_sec=20.0,
            max_cycle_elapsed_sec=45.0,
        )
        assert set(out) == set(names)
        # Fair share of 45s across 6 tables binds below the 20s per-table cap.
        assert out[names[0]].effective_max_elapsed_sec == pytest.approx(45.0 / 6)
        assert all(
            0 < st.effective_max_elapsed_sec <= 20.0 for st in out.values()
        ), {k: v.effective_max_elapsed_sec for k, v in out.items()}
        # And it must differ from the configured per-table cap, or it is reporting nothing.
        assert out[names[0]].effective_max_elapsed_sec != 20.0


def test_adding_a_table_cannot_silently_dilute_every_other_tables_share():
    """Replaces a coupling that was lost when the budget tests were pinned to a fixture.

    Before that change, `test_the_budget_is_split_fairly_not_first_come` ran the LIVE
    registry against the real 45s budget. It was a crude tripwire, but it was the only
    thing forcing whoever adds a table to look at the arithmetic -- and the arithmetic
    does move: going from six tables to seven cut the per-table fair share from 7.5s to
    6.43s, 14% off every existing table, while substrate_proposal_frames was measured at
    8.21s during its own backfill and was already over the old share.

    So this asserts the invariant directly instead. The next table addition should be a
    decision with a number attached, not a silent dilution."""
    from app import grammar_truth
    from app.settings import get_settings

    settings = get_settings()
    budget = float(
        getattr(settings, "grammar_retention_periodic_max_cycle_sec", 0.0) or 45.0
    )
    n = len(grammar_truth.GRAMMAR_RETENTION_TABLES)
    fair_share = budget / n

    assert n >= 1
    assert fair_share >= 5.0, (
        f"{n} retention tables against a {budget}s cycle budget leaves {fair_share:.2f}s "
        "per table. Below ~5s a real backfill cannot finish a batch, so tables start "
        "silently carrying debt forever. Raise the cycle budget or drop a table -- do "
        "not just lower this floor."
    )
