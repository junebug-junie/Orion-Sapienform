"""Tests for app/self_study_refresh.py -- the periodic Layer-1 self-fact
refresh that replaces "Orion's action-selection will pick self_repo_inspect"
(it never did; see the module docstring)."""
import asyncio
import importlib.util
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = SERVICE_DIR / "app"
PACKAGE_NAME = "orion_cortex_exec"
APP_PACKAGE_NAME = f"{PACKAGE_NAME}.app"
if PACKAGE_NAME not in sys.modules:
    pkg = types.ModuleType(PACKAGE_NAME)
    pkg.__path__ = [str(SERVICE_DIR)]
    sys.modules[PACKAGE_NAME] = pkg
if APP_PACKAGE_NAME not in sys.modules:
    pkg = types.ModuleType(APP_PACKAGE_NAME)
    pkg.__path__ = [str(APP_DIR)]
    sys.modules[APP_PACKAGE_NAME] = pkg

_key = f"{APP_PACKAGE_NAME}.self_study_refresh"
if _key in sys.modules:
    refresh = sys.modules[_key]
else:
    spec = importlib.util.spec_from_file_location(_key, APP_DIR / "self_study_refresh.py")
    refresh = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = refresh
    spec.loader.exec_module(refresh)


NOW = datetime(2026, 9, 19, 12, 0, 0, tzinfo=timezone.utc)
DAY = 86400.0


def test_due_immediately_when_no_prior_run():
    assert refresh.seconds_until_due(None, interval_sec=DAY, now=NOW) == 0.0


def test_due_immediately_when_interval_elapsed():
    stale = NOW - timedelta(days=14)
    assert refresh.seconds_until_due(stale, interval_sec=DAY, now=NOW) == 0.0


def test_waits_out_the_remainder_when_recent():
    recent = NOW - timedelta(hours=6)
    assert refresh.seconds_until_due(recent, interval_sec=DAY, now=NOW) == DAY - 6 * 3600


class _StopLoop(Exception):
    pass


class _Harness:
    """Drives the loop with fakes; `sleep` records requested durations and
    stops the loop after `max_sleeps` so tests terminate."""

    def __init__(self, *, newest_sequence, max_sleeps=1, fail_run=False):
        self._newest = list(newest_sequence)
        self.sleeps: list[float] = []
        self.runs: list[dict] = []
        self._max_sleeps = max_sleeps
        self._fail_run = fail_run
        self.bus = object()

    def newest_at(self):
        value = self._newest.pop(0) if len(self._newest) > 1 else self._newest[0]
        return value

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        if len(self.sleeps) >= self._max_sleeps:
            raise _StopLoop()

    async def run_inspect(self, *, bus, source, correlation_id):
        self.runs.append({"bus": bus, "source": source, "correlation_id": correlation_id})
        if self._fail_run:
            raise RuntimeError("scan_exploded")
        return types.SimpleNamespace(
            snapshot=types.SimpleNamespace(run_id="run-1"),
            self_knowledge_items_write=types.SimpleNamespace(status="written", detail="published=1248 failed=0"),
        )

    def drive(self, interval_sec=DAY):
        async def _go():
            with_stop = refresh.self_study_refresh_loop(
                bus_getter=lambda: self.bus,
                source="src",
                interval_sec=interval_sec,
                run_inspect=self.run_inspect,
                newest_at=self.newest_at,
                sleep=self.sleep,
                clock=lambda: NOW,
            )
            try:
                await with_stop
            except _StopLoop:
                pass

        asyncio.run(_go())


def test_stale_table_runs_inspect_then_sleeps_full_interval():
    h = _Harness(newest_sequence=[NOW - timedelta(days=14)])
    h.drive()
    assert len(h.runs) == 1
    assert h.runs[0]["bus"] is h.bus
    assert h.runs[0]["source"] == "src"
    assert h.runs[0]["correlation_id"]
    assert h.sleeps == [DAY]


def test_fresh_table_skips_run_and_sleeps_remainder():
    """Restart guard: a container restart 6h after the last run must not
    append another snapshot -- it sleeps the remaining 18h instead."""
    h = _Harness(newest_sequence=[NOW - timedelta(hours=6)])
    h.drive()
    assert h.runs == []
    assert h.sleeps == [DAY - 6 * 3600]


def test_empty_table_runs_immediately():
    h = _Harness(newest_sequence=[None])
    h.drive()
    assert len(h.runs) == 1


def test_failed_run_is_logged_and_loop_continues():
    h = _Harness(newest_sequence=[None], fail_run=True)
    h.drive()
    assert len(h.runs) == 1
    assert h.sleeps == [DAY]


def test_second_iteration_reads_newest_again():
    """After a run, the next iteration re-reads the table: the just-written
    rows make it 'fresh', so it waits instead of running twice."""
    h = _Harness(newest_sequence=[None, NOW], max_sleeps=2)
    h.drive()
    assert len(h.runs) == 1
    assert h.sleeps == [DAY, DAY]


def test_interval_floor_prevents_hot_loop():
    h = _Harness(newest_sequence=[None])
    h.drive(interval_sec=1)
    assert h.sleeps == [60.0]
