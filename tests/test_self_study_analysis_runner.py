"""End-to-end shape of `run_self_study_analysis` and the verb adapter around
it, against a fake engine.

What these prove that the rule tests cannot: the refusal paths reach the right
`status`, the journal entry that DOES get written contains the real numbers,
the cooldown suppresses a repeat, a broken dedup lookup fails CLOSED, and the
verb reports a quiet run as a success rather than feeding the theater tripwire.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "services" / "orion-cortex-exec"))

from app.self_study_analysis import (  # noqa: E402
    ANALYSIS_SOURCES,
    DEFAULT_WINDOW_HOURS,
    MAX_WINDOW_HOURS,
    MIN_WINDOW_HOURS,
    SOURCE_SPECS,
    _clamp_window_hours,
    run_self_study_analysis,
    select_least_recently_analysed,
)
from orion.core.bus.bus_schemas import ServiceRef  # noqa: E402

NOW = datetime(2026, 8, 25, 12, 0, tzinfo=timezone.utc)
SOURCE_REF = ServiceRef(name="orion-cortex-exec", version="0.1.0", node="athena")


class _FakeResult:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def mappings(self):
        return iter(self._rows)

    def first(self):
        return self._rows[0] if self._rows else None


class _FakeConn:
    def __init__(self, engine: "_FakeEngine") -> None:
        self._engine = engine

    def __enter__(self) -> "_FakeConn":
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def execute(self, statement, params):
        return self._engine.execute(str(statement), params)


class _FakeEngine:
    """Answers the three query shapes the module issues, by inspecting the SQL
    text rather than by call order -- so a reordering inside the module shows up
    as a test failure instead of silently returning the wrong fixture."""

    def __init__(
        self,
        *,
        window_rows: dict[tuple[datetime, datetime], list[dict]] | None = None,
        journaled: bool = False,
        last_by_source: list[dict] | None = None,
        dedup_raises: bool = False,
        window_raises: bool = False,
    ) -> None:
        self.window_rows = window_rows or {}
        self.journaled = journaled
        self.last_by_source = last_by_source or []
        self.dedup_raises = dedup_raises
        self.window_raises = window_raises
        self.dedup_source_refs: list[str] = []

    def connect(self) -> _FakeConn:
        return _FakeConn(self)

    def execute(self, sql: str, params: dict):
        if "split_part" in sql:
            return _FakeResult(list(self.last_by_source))
        if "FROM journal_entries" in sql:
            if self.dedup_raises:
                raise RuntimeError("journal_entries_unavailable")
            self.dedup_source_refs.append(params["source_ref"])
            return _FakeResult([{"?column?": 1}] if self.journaled else [])
        if self.window_raises:
            raise RuntimeError("statement_timeout")
        key = (params["since"], params["until"])
        return _FakeResult(list(self.window_rows.get(key, [])))


def _vision_rows(n: int, *, since: datetime, until: datetime, event_type: str = "seen") -> list[dict]:
    if n <= 0:
        return []
    step = (until - since) / (n + 1)
    return [
        {
            "created_at": since + step * (i + 1),
            "confidence": 0.8,
            "salience": 0.7,
            "event_type": event_type,
        }
        for i in range(n)
    ]


def _engine_for(recent: int, baseline: int, **kwargs) -> _FakeEngine:
    hours = DEFAULT_WINDOW_HOURS
    recent_since = NOW - timedelta(hours=hours)
    baseline_since = NOW - timedelta(hours=2 * hours)
    return _FakeEngine(
        window_rows={
            (recent_since, NOW): _vision_rows(recent, since=recent_since, until=NOW),
            (baseline_since, recent_since): _vision_rows(
                baseline, since=baseline_since, until=recent_since
            ),
        },
        **kwargs,
    )


class _RecordingBus:
    def __init__(self, *, fail: bool = False) -> None:
        self.published: list[tuple[str, object]] = []
        self.fail = fail

    async def publish(self, channel: str, envelope) -> None:
        if self.fail:
            raise RuntimeError("bus_down")
        self.published.append((channel, envelope))


def _run(**kwargs):
    kwargs.setdefault("bus", None)
    kwargs.setdefault("source_ref", SOURCE_REF)
    kwargs.setdefault("source", "vision_events")
    kwargs.setdefault("correlation_id", "corr-1")
    kwargs.setdefault("now", NOW)
    return asyncio.run(run_self_study_analysis(**kwargs))


# --- refusal paths ---------------------------------------------------------


def test_a_quiet_window_writes_nothing() -> None:
    bus = _RecordingBus()
    result = _run(engine=_engine_for(50, 52), bus=bus)
    assert result.status == "skipped_not_notable"
    assert result.journal_entry is None
    assert bus.published == []


def test_an_unknown_source_is_unavailable_not_a_crash() -> None:
    result = _run(engine=_engine_for(50, 52), source="tea_leaves")
    assert result.status == "unavailable"
    assert "unknown_source" in (result.unavailable_reason or "")


def test_a_failed_query_is_unavailable_not_quiet() -> None:
    """"Cannot read" must never be reported as "nothing to report"."""
    result = _run(engine=_engine_for(50, 52, window_raises=True))
    assert result.status == "unavailable"
    assert (result.unavailable_reason or "").startswith("query_failed:")


def test_a_broken_dedup_lookup_fails_closed() -> None:
    """Skipping a real entry is recoverable; a spam loop against a broken
    journal table is not."""
    bus = _RecordingBus()
    result = _run(engine=_engine_for(20, 80, dedup_raises=True), bus=bus)
    assert result.findings, "fixture must actually fire a rule"
    assert result.status == "unavailable"
    assert (result.unavailable_reason or "").startswith("dedup_failed:")
    assert bus.published == []


def test_the_cooldown_suppresses_an_identical_finding_set() -> None:
    bus = _RecordingBus()
    result = _run(engine=_engine_for(20, 80, journaled=True), bus=bus)
    assert result.status == "skipped_recently_journaled"
    assert bus.published == []


def test_the_cooldown_is_keyed_on_the_source_and_finding_digest() -> None:
    engine = _engine_for(20, 80, journaled=True)
    result = _run(engine=engine, bus=_RecordingBus())
    assert engine.dedup_source_refs == [f"vision_events:{result.finding_digest}"]


# --- the write path --------------------------------------------------------


def test_a_notable_window_publishes_one_journal_entry() -> None:
    bus = _RecordingBus()
    result = _run(engine=_engine_for(20, 80), bus=bus)
    assert result.status == "journaled"
    assert [rule for rule in ("volume_shift",) if rule in {f.rule for f in result.findings}]
    assert len(bus.published) == 1
    channel, envelope = bus.published[0]
    assert channel == "orion:journal:write"
    assert envelope.kind == "journal.entry.write.v1"
    assert envelope.payload["source_kind"] == "self_study"
    assert envelope.payload["source_ref"] == f"vision_events:{result.finding_digest}"


def test_the_journal_body_carries_the_real_numbers_and_the_negative_space() -> None:
    result = _run(engine=_engine_for(20, 80), bus=_RecordingBus())
    body = result.journal_entry.body
    assert "20 rows recent, 80 baseline" in body
    assert "volume_shift" in body
    assert "Checked and did not fire:" in body
    assert "producer_stalled" in body  # a rule that did NOT fire, named as such
    assert SOURCE_SPECS["vision_events"].table in body
    # The entry must never overclaim its own source.
    assert "write time" in body


def test_a_bus_failure_is_reported_not_swallowed() -> None:
    result = _run(engine=_engine_for(20, 80), bus=_RecordingBus(fail=True))
    assert result.status == "journal_failed"
    assert result.journal_write is not None and result.journal_write.status == "failed"


def test_a_missing_bus_is_reported_not_counted_as_written() -> None:
    result = _run(engine=_engine_for(20, 80), bus=None)
    assert result.status == "journal_failed"
    assert result.journal_write is not None
    assert result.journal_write.detail == "missing_bus"


# --- source selection ------------------------------------------------------


def test_selection_prefers_a_source_never_analysed() -> None:
    engine = _FakeEngine(
        last_by_source=[
            {"src": name, "last_at": NOW - timedelta(hours=i + 1)}
            for i, name in enumerate(ANALYSIS_SOURCES[:-1])
        ]
    )
    assert select_least_recently_analysed(engine) == ANALYSIS_SOURCES[-1]


def test_selection_prefers_the_most_overdue_source() -> None:
    engine = _FakeEngine(
        last_by_source=[
            {"src": "concept_induction", "last_at": NOW - timedelta(hours=1)},
            {"src": "vision_events", "last_at": NOW - timedelta(hours=9)},
            {"src": "affective_state", "last_at": NOW - timedelta(hours=2)},
            {"src": "cocreation_signals", "last_at": NOW - timedelta(hours=3)},
        ]
    )
    assert select_least_recently_analysed(engine) == "vision_events"


def test_selection_cold_start_is_deterministic() -> None:
    assert select_least_recently_analysed(_FakeEngine()) == ANALYSIS_SOURCES[0]


def test_selection_ignores_source_refs_it_does_not_recognise() -> None:
    engine = _FakeEngine(
        last_by_source=[{"src": "something_else", "last_at": NOW}]
    )
    assert select_least_recently_analysed(engine) in ANALYSIS_SOURCES


# --- window clamping -------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, DEFAULT_WINDOW_HOURS),
        ("", DEFAULT_WINDOW_HOURS),
        ("nonsense", DEFAULT_WINDOW_HOURS),
        (float("nan"), DEFAULT_WINDOW_HOURS),
        (float("inf"), DEFAULT_WINDOW_HOURS),
        (-5.0, MIN_WINDOW_HOURS),
        (0.0, MIN_WINDOW_HOURS),
        (99999.0, MAX_WINDOW_HOURS),
        ("12", 12.0),
        (3.5, 3.5),
    ],
)
def test_window_hours_is_clamped(raw, expected) -> None:
    assert _clamp_window_hours(raw) == expected


# --- the verb adapter ------------------------------------------------------


def _plan_request(request_id: str, *, source: str = "vision_events"):
    from orion.schemas.cortex.schemas import ExecutionPlan, PlanExecutionArgs, PlanExecutionRequest

    return PlanExecutionRequest(
        plan=ExecutionPlan(verb_name="skills.self_study.analyze.v1", steps=[]),
        args=PlanExecutionArgs(request_id=request_id, extra={"skill_args": {"source": source}}),
        context={},
    )


def test_the_verb_reports_a_quiet_run_as_success() -> None:
    """A refusal to journal is this action working. Reporting it as an error
    would drive the execution-dispatch theater tripwire into a latch on an
    action behaving exactly as designed."""
    from app import self_study_analysis as analysis_module
    from app import verb_adapters
    from orion.core.verbs.base import VerbContext

    engine = _engine_for(50, 52)
    original = analysis_module._get_engine
    analysis_module._get_engine = lambda: engine  # type: ignore[assignment]
    try:
        payload = _plan_request("req-1")
        ctx = VerbContext(request_id="req-1", meta={"correlation_id": "corr-1", "bus": None})
        out, effects = asyncio.run(verb_adapters.SelfStudyAnalyzeVerb().execute(ctx, payload))
    finally:
        analysis_module._get_engine = original  # type: ignore[assignment]
    assert out.ok is True
    assert out.status == "success"
    assert out.error is None
    assert out.metadata["skill_result"]["status"] == "skipped_not_notable"
    assert effects == []


def test_the_verb_reports_an_unreadable_source_as_an_error() -> None:
    from app import self_study_analysis as analysis_module
    from app import verb_adapters
    from orion.core.verbs.base import VerbContext

    engine = _engine_for(50, 52, window_raises=True)
    original = analysis_module._get_engine
    analysis_module._get_engine = lambda: engine  # type: ignore[assignment]
    try:
        payload = _plan_request("req-2")
        ctx = VerbContext(request_id="req-2", meta={"correlation_id": "corr-2", "bus": None})
        out, _ = asyncio.run(verb_adapters.SelfStudyAnalyzeVerb().execute(ctx, payload))
    finally:
        analysis_module._get_engine = original  # type: ignore[assignment]
    assert out.ok is False
    assert out.status == "error"
    assert out.error is not None
