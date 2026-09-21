"""Tests for self_study_reflect_refresh_loop in app/self_study_refresh.py --
the periodic Layer-3 reflection timer that replaces "nothing ever calls
run_self_concept_reflect" (see the module docstring for the live-check that
found zero self_concept_history rows ever, under produced_by='layer3_reflect')."""
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


NOW = datetime(2026, 9, 20, 12, 0, 0, tzinfo=timezone.utc)
DAY = 86400.0


class _StopLoop(Exception):
    pass


class _Harness:
    """Same drive-with-fakes shape as test_self_study_refresh.py's harness,
    pointed at self_study_reflect_refresh_loop instead."""

    def __init__(self, *, newest_sequence, max_sleeps=1, fail_run=False, llm_call_failed=False):
        self._newest = list(newest_sequence)
        self.sleeps: list[float] = []
        self.runs: list[dict] = []
        self._max_sleeps = max_sleeps
        self._fail_run = fail_run
        self._llm_call_failed = llm_call_failed
        self.bus = object()

    def newest_at(self):
        value = self._newest.pop(0) if len(self._newest) > 1 else self._newest[0]
        return value

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        if len(self.sleeps) >= self._max_sleeps:
            raise _StopLoop()

    async def run_reflect(self, *, bus, source, correlation_id):
        self.runs.append({"bus": bus, "source": source, "correlation_id": correlation_id})
        if self._fail_run:
            raise RuntimeError("reflect_exploded")
        if self._llm_call_failed:
            return types.SimpleNamespace(
                run_id="run-1",
                findings=[],
                self_concept_history_write=types.SimpleNamespace(
                    status="skipped", detail="reflection_llm_call_failed"
                ),
            )
        return types.SimpleNamespace(
            run_id="run-1",
            findings=[object(), object()],
            self_concept_history_write=types.SimpleNamespace(status="written", detail="published=2 failed=0"),
        )

    def drive(self, interval_sec=DAY):
        async def _go():
            with_stop = refresh.self_study_reflect_refresh_loop(
                bus_getter=lambda: self.bus,
                source="src",
                interval_sec=interval_sec,
                run_reflect=self.run_reflect,
                newest_at=self.newest_at,
                sleep=self.sleep,
                clock=lambda: NOW,
            )
            try:
                await with_stop
            except _StopLoop:
                pass

        asyncio.run(_go())


def test_stale_history_runs_reflect_then_sleeps_full_interval():
    h = _Harness(newest_sequence=[NOW - timedelta(days=14)])
    h.drive()
    assert len(h.runs) == 1
    assert h.runs[0]["bus"] is h.bus
    assert h.runs[0]["source"] == "src"
    assert h.runs[0]["correlation_id"]
    assert h.sleeps == [DAY]


def test_fresh_history_skips_run_and_sleeps_remainder():
    """Restart guard: a container restart 6h after the last reflection must
    not run again -- it sleeps the remaining 18h instead."""
    h = _Harness(newest_sequence=[NOW - timedelta(hours=6)])
    h.drive()
    assert h.runs == []
    assert h.sleeps == [DAY - 6 * 3600]


def test_no_prior_reflection_runs_immediately():
    h = _Harness(newest_sequence=[None])
    h.drive()
    assert len(h.runs) == 1


def test_failed_run_is_logged_and_loop_continues():
    h = _Harness(newest_sequence=[None], fail_run=True)
    h.drive()
    assert len(h.runs) == 1
    assert h.sleeps == [DAY]


def test_llm_call_failure_does_not_crash_the_loop_and_retries_next_interval():
    """run_self_concept_reflect deliberately writes no self_concept_history
    row when the LLM call fails (no-empty-shell-cognition guard) -- the loop
    must not treat that as an exception, and must still wait a full interval
    rather than tight-looping on repeated failure."""
    h = _Harness(newest_sequence=[None], llm_call_failed=True)
    h.drive()
    assert len(h.runs) == 1
    assert h.sleeps == [DAY]


def test_second_iteration_reads_newest_again():
    """After a successful run, the next iteration re-reads the table: the
    just-written row makes it 'fresh', so it waits instead of running twice."""
    h = _Harness(newest_sequence=[None, NOW], max_sleeps=2)
    h.drive()
    assert len(h.runs) == 1
    assert h.sleeps == [DAY, DAY]


def test_interval_floor_prevents_hot_loop():
    h = _Harness(newest_sequence=[None])
    h.drive(interval_sec=1)
    assert h.sleeps == [60.0]
