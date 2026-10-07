"""H1 verdict atoms: debounce, cap, summary, self-skip, per-producer unrouted counts."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orion.schemas.grammar import GrammarEventV1

from app.service import HeartbeatService
from app.settings import settings
from app.substrate.ensemble import EnsembleH1ResultV1
from app.substrate.routing import ORGAN_SITE_MAP, SELF_SOURCE_SERVICE, catalog_grammar_producers
from app.substrate.verdict_atoms import (
    ROLE_HOURLY_SUMMARY,
    ROLE_TRANSITION,
    SOURCE_SERVICE,
    TRACE_PREFIX,
    H1Tick,
    HourlyWindow,
    VerdictTransitionTracker,
    build_summary_event,
    build_transition_event,
)

T0 = datetime(2026, 10, 7, 0, 0, tzinfo=timezone.utc)


def _tick(i: int, verdict: str, std: float = 0.04) -> H1Tick:
    return H1Tick(
        at=T0 + timedelta(seconds=30 * i),
        verdict=verdict,
        mean_ratio=0.85,
        std_ratio=std,
        bulk_penetration_depth=0.87,
    )


def _run(tracker: VerdictTransitionTracker, verdicts: list[str]):
    return [t for i, v in enumerate(verdicts) if (t := tracker.observe(_tick(i, v))) is not None]


def test_single_tick_flicker_never_emits() -> None:
    # The live shape: mostly mixed with 1-tick flickers into the other classes.
    tracker = VerdictTransitionTracker(settle_ticks=3)
    out = _run(tracker, ["mixed"] * 3 + ["concentrated", "mixed", "redundant", "mixed", "concentrated"] + ["mixed"] * 4)
    assert [(t.from_verdict, t.to_verdict) for t in out] == [("none", "mixed")]


def test_settled_change_emits_once_with_held_window() -> None:
    tracker = VerdictTransitionTracker(settle_ticks=3)
    out = _run(tracker, ["mixed"] * 3 + ["concentrated"] * 6)
    assert [(t.from_verdict, t.to_verdict) for t in out] == [("none", "mixed"), ("mixed", "concentrated")]
    change = out[1]
    assert change.held_ticks == 3
    assert change.held_since == T0 + timedelta(seconds=90)
    assert change.confirmed_at == T0 + timedelta(seconds=150)


def test_return_to_confirmed_class_after_flicker_emits_nothing() -> None:
    tracker = VerdictTransitionTracker(settle_ticks=3)
    out = _run(tracker, ["mixed"] * 3 + ["redundant", "redundant"] + ["mixed"] * 5)
    assert len(out) == 1


def test_settle_ticks_must_be_positive() -> None:
    with pytest.raises(ValueError):
        VerdictTransitionTracker(settle_ticks=0)


def test_long_hold_keeps_constant_state_and_true_held_since() -> None:
    tracker = VerdictTransitionTracker(settle_ticks=3)
    _run(tracker, ["mixed"] * 5000)
    assert tracker._streak_len == 5000
    assert not hasattr(tracker, "_streak")
    out = [tracker.observe(_tick(5000 + i, "redundant", std=0.01 * (i + 1))) for i in range(3)]
    t = out[-1]
    assert t is not None and t.held_since == T0 + timedelta(seconds=30 * 5000)
    assert t.mean_std_ratio_while_held == pytest.approx(0.02)


def test_settings_reject_degenerate_values(monkeypatch) -> None:
    from pydantic import ValidationError

    from app.settings import Settings

    for key, bad in (
        ("HEARTBEAT_VERDICT_SETTLE_TICKS", "0"),
        ("HEARTBEAT_VERDICT_SUMMARY_INTERVAL_SEC", "0"),
        ("HEARTBEAT_VERDICT_MAX_TRANSITIONS_PER_HOUR", "0"),
    ):
        monkeypatch.setenv(key, bad)
        with pytest.raises(ValidationError):
            Settings()
        monkeypatch.delenv(key)


def test_transition_event_is_a_valid_bounded_grammar_atom() -> None:
    tracker = VerdictTransitionTracker(settle_ticks=2)
    t = _run(tracker, ["mixed", "mixed"])[0]
    event = build_transition_event(node="athena", transition=t)
    GrammarEventV1.model_validate(event.model_dump(mode="json"))
    assert event.provenance.source_service == SOURCE_SERVICE == SELF_SOURCE_SERVICE
    assert event.trace_id.startswith(f"{TRACE_PREFIX}athena:transition:")
    assert event.atom.semantic_role == ROLE_TRANSITION
    for key in ("from=none", "to=mixed", "std_ratio=", "mean_ratio=", "held_ticks=2", "held_since="):
        assert key in event.atom.summary


def test_summary_reports_class_ticks_flips_and_unrouted() -> None:
    w = HourlyWindow(start=T0)
    for i, v in enumerate(["mixed", "concentrated", "mixed", "redundant"]):
        w.record_tick(_tick(i, v, std=0.01 * (i + 1)))
    w.record_unrouted("orion-sql-writer")
    w.record_unrouted("orion-sql-writer")
    w.record_unrouted("orion-llm-gateway")
    event = build_summary_event(node="athena", window=w, end=T0 + timedelta(hours=1))
    s = event.atom.summary
    assert event.atom.semantic_role == ROLE_HOURLY_SUMMARY
    assert "h1_ticks=4" in s
    assert "verdict_ticks=concentrated:1|mixed:2|redundant:1" in s
    assert "raw_flips=3" in s
    assert "std_ratio_mean=0.0250 std_ratio_min=0.0100 std_ratio_max=0.0400" in s
    assert "unrouted_atoms=orion-llm-gateway:1|orion-sql-writer:2" in s
    assert event.atom.confidence == 1.0


def test_empty_summary_reads_as_absence_not_calm() -> None:
    w = HourlyWindow(start=T0)
    w.h1_failures = 3
    event = build_summary_event(node="athena", window=w, end=T0 + timedelta(hours=1))
    assert "h1_ticks=0" in event.atom.summary
    assert "h1_failures=3" in event.atom.summary
    assert "std_ratio_mean=none" in event.atom.summary
    assert event.atom.confidence == 0.0


def test_unrouted_keys_are_bounded() -> None:
    w = HourlyWindow(start=T0)
    for i in range(100):
        w.record_unrouted(f"svc-{i}")
    assert len(w.unrouted_by_source) == 33  # 32 named + "other"
    assert w.unrouted_by_source["other"] == 68


# --- service integration -------------------------------------------------


class _FakeBus:
    def __init__(self, fail: bool = False) -> None:
        self.published: list[tuple[str, object]] = []
        self.fail = fail

    async def publish(self, channel, envelope) -> None:
        if self.fail:
            raise ConnectionError("bus down")
        self.published.append((channel, envelope))


def _h1(verdict: str) -> EnsembleH1ResultV1:
    return EnsembleH1ResultV1(mean_ratio=0.85, std_ratio=0.05, verdict=verdict, tick_count=10)


def _svc(monkeypatch, bus: _FakeBus) -> HeartbeatService:
    svc = HeartbeatService()
    monkeypatch.setattr(svc, "bus", bus)
    return svc


async def _feed(svc: HeartbeatService, verdicts: list[str], start: datetime = T0) -> None:
    for i, v in enumerate(verdicts):
        svc.latest_h1 = _h1(v)
        await svc._observe_verdict(start + timedelta(seconds=30 * i))


@pytest.mark.asyncio
async def test_service_publishes_settled_transition_on_grammar_channel(monkeypatch) -> None:
    bus = _FakeBus()
    svc = _svc(monkeypatch, bus)
    await _feed(svc, ["mixed"] * 3 + ["concentrated"] + ["mixed"] * 3 + ["redundant"] * 3)
    roles = [env.payload["atom"]["semantic_role"] for _, env in bus.published]
    assert [ch for ch, _ in bus.published] == ["orion:grammar:event"] * 2
    assert roles == [ROLE_TRANSITION, ROLE_TRANSITION]
    assert "to=redundant" in bus.published[-1][1].payload["atom"]["summary"]
    assert svc.verdict_atoms_published == 2
    assert svc.verdict_window.raw_flips == 3
    assert svc.verdict_window.h1_ticks == 10


@pytest.mark.asyncio
async def test_transition_cap_counts_suppressed(monkeypatch) -> None:
    bus = _FakeBus()
    svc = _svc(monkeypatch, bus)
    monkeypatch.setattr(settings, "verdict_max_transitions_per_hour", 2)
    await _feed(svc, (["mixed"] * 3 + ["concentrated"] * 3) * 3)
    assert len(bus.published) == 2
    assert svc.verdict_window.transitions_emitted == 2
    assert svc.verdict_window.transitions_suppressed == 4


@pytest.mark.asyncio
async def test_cap_frees_up_after_a_rolling_hour_and_reports_suppressed(monkeypatch) -> None:
    bus = _FakeBus()
    svc = _svc(monkeypatch, bus)
    monkeypatch.setattr(settings, "verdict_max_transitions_per_hour", 1)
    await _feed(svc, ["mixed"] * 3 + ["concentrated"] * 3)  # 2nd transition capped
    assert len(bus.published) == 1
    await _feed(svc, ["redundant"] * 3, start=T0 + timedelta(hours=1, minutes=1))
    assert len(bus.published) == 2
    summary = bus.published[-1][1].payload["atom"]["summary"]
    assert "from=concentrated" in summary and "suppressed_since_last=1" in summary


@pytest.mark.asyncio
async def test_publish_failure_is_counted_not_raised(monkeypatch) -> None:
    bus = _FakeBus(fail=True)
    svc = _svc(monkeypatch, bus)
    await _feed(svc, ["mixed"] * 3)
    assert svc.verdict_atoms_publish_failed == 1
    assert svc.verdict_atoms_published == 0
    assert svc.verdict_window.transitions_publish_failed == 1
    bus.fail = False
    await _feed(svc, ["redundant"] * 3, start=T0 + timedelta(minutes=5))
    assert "suppressed_since_last=1" in bus.published[-1][1].payload["atom"]["summary"]


async def _run_h1_loop_once(svc: HeartbeatService, monkeypatch) -> None:
    import asyncio

    monkeypatch.setattr(settings, "h1_interval_sec", 0.0)
    real_sleep = asyncio.sleep

    async def one_tick_sleep(_sec):
        # Stop is checked at the top of the loop, so setting it here lets
        # exactly one loop body run after this sleep.
        svc._stop.set()
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", one_tick_sleep)
    await svc._h1_loop()


@pytest.mark.asyncio
async def test_h1_loop_failure_still_flushes_an_absence_summary(monkeypatch) -> None:
    bus = _FakeBus()
    svc = _svc(monkeypatch, bus)
    monkeypatch.setattr(settings, "verdict_summary_interval_sec", 0.001)
    svc.verdict_window = HourlyWindow(start=T0)

    def boom(*_a, **_k):
        raise RuntimeError("quimb exploded")

    # Patch the globals the loop actually resolves names in: conftest drops
    # sys.modules["app.*"] between tests, so a dotted-string target misses.
    monkeypatch.setitem(type(svc)._h1_loop.__globals__, "compute_h1_ensemble", boom)
    await _run_h1_loop_once(svc, monkeypatch)
    roles = [env.payload["atom"]["semantic_role"] for _, env in bus.published]
    assert roles == [ROLE_HOURLY_SUMMARY]
    summary = bus.published[0][1].payload["atom"]["summary"]
    assert "h1_ticks=0" in summary and "h1_failures=1" in summary
    assert bus.published[0][1].payload["atom"]["confidence"] == 0.0


@pytest.mark.asyncio
async def test_h1_loop_publishes_nothing_when_disabled(monkeypatch) -> None:
    bus = _FakeBus()
    svc = _svc(monkeypatch, bus)
    monkeypatch.setattr(settings, "verdict_atoms_enabled", False)
    monkeypatch.setattr(settings, "verdict_summary_interval_sec", 0.001)
    monkeypatch.setattr(settings, "verdict_settle_ticks", 1)
    svc.verdict_tracker = VerdictTransitionTracker(settle_ticks=1)
    svc.verdict_window = HourlyWindow(start=T0)
    await _run_h1_loop_once(svc, monkeypatch)
    assert svc.latest_h1 is not None  # H1 itself still ran
    assert bus.published == []


@pytest.mark.asyncio
async def test_summary_flushes_once_per_window_and_resets(monkeypatch) -> None:
    bus = _FakeBus()
    svc = _svc(monkeypatch, bus)
    svc.verdict_window = HourlyWindow(start=T0)
    await _feed(svc, ["mixed"])
    await svc._maybe_flush_summary(T0 + timedelta(minutes=59))
    assert len(bus.published) == 0
    await svc._maybe_flush_summary(T0 + timedelta(hours=1))
    assert [env.payload["atom"]["semantic_role"] for _, env in bus.published] == [ROLE_HOURLY_SUMMARY]
    assert svc.verdict_window.start == T0 + timedelta(hours=1)
    assert svc.verdict_window.h1_ticks == 0


@pytest.mark.asyncio
async def test_own_published_atom_echo_is_skipped_not_absorbed_or_counted_unrouted(monkeypatch) -> None:
    bus = _FakeBus()
    svc = _svc(monkeypatch, bus)
    await _feed(svc, ["mixed"] * 3)
    _, envelope = bus.published[0]
    # Feed back the exact wire payload heartbeat just published.
    await svc._handle_grammar_message(envelope.payload)
    assert svc.events_skipped_self == 1
    assert svc.events_skipped_organ == 0
    assert svc.events_queued == 0
    assert svc.events_skipped_organ_by_source == {}


@pytest.mark.asyncio
async def test_unrouted_producers_are_counted_by_source(monkeypatch) -> None:
    svc = _svc(monkeypatch, _FakeBus())
    for src in ("orion-sql-writer", "orion-sql-writer", "orion-llm-gateway"):
        await svc._handle_grammar_message(
            {
                "event_kind": "atom_emitted",
                "provenance": {"source_service": src},
                "atom": {"atom_type": "observation", "confidence": 1.0},
            }
        )
    assert svc.events_skipped_organ == 3
    assert svc.events_skipped_organ_by_source == {"orion-sql-writer": 2, "orion-llm-gateway": 1}
    assert svc.verdict_window.unrouted_by_source == {"orion-sql-writer": 2, "orion-llm-gateway": 1}
    stats = await svc.stats()
    assert "orion-sql-writer" in stats["catalog_producers_unrouted"]
    assert stats["uncatalogued_sources_seen"] == []
    assert stats["catalog_loaded"] is True
    assert stats["verdict_atoms"]["enabled"] is True


@pytest.mark.asyncio
async def test_unreadable_catalog_reads_unknown_not_empty(monkeypatch) -> None:
    svc = _svc(monkeypatch, _FakeBus())
    svc.catalog_producers = frozenset()
    stats = await svc.stats()
    assert stats["catalog_loaded"] is False
    assert stats["catalog_producers_unrouted"] is None


def test_catalog_producers_include_routed_organs_and_self() -> None:
    producers = catalog_grammar_producers()
    assert set(ORGAN_SITE_MAP) <= producers
    assert SELF_SOURCE_SERVICE in producers

