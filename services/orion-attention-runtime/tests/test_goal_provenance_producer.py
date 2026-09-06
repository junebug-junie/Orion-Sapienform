from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.worker import AttentionRuntimeWorker
from orion.attention.field_attention.goal_provenance import DominanceStreak
from orion.schemas.field_attention_frame import FieldAttentionFrameV1, FieldAttentionTargetV1
from orion.schemas.field_goal import DominanceStreakTickV1, FieldGoalProvenanceV1


def _target(target_id: str, salience: float, kind: str = "node") -> FieldAttentionTargetV1:
    return FieldAttentionTargetV1(
        target_id=target_id,
        target_kind=kind,
        salience_score=salience,
        pressure_score=salience,
        novelty_score=salience,
        urgency_score=salience,
        confidence_score=1.0,
    )


def _frame(node_targets: list[FieldAttentionTargetV1]) -> FieldAttentionFrameV1:
    return FieldAttentionFrameV1(
        frame_id="frame-1",
        generated_at="2026-07-30T00:00:00Z",
        source_field_tick_id="tick-1",
        source_field_generated_at="2026-07-30T00:00:00Z",
        overall_salience=max((t.salience_score for t in node_targets), default=0.0),
        dominant_targets=node_targets,
        node_targets=node_targets,
    )


def _make_worker(
    monkeypatch,
    *,
    producer_enabled: bool = True,
    min_streak: int = 3,
    streak_tick_telemetry_enabled: bool = True,
) -> AttentionRuntimeWorker:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    monkeypatch.setenv("ORION_GOAL_PROVENANCE_PRODUCER_ENABLED", str(producer_enabled))
    monkeypatch.setenv("ORION_GOAL_PROVENANCE_MIN_STREAK", str(min_streak))
    monkeypatch.setenv(
        "ORION_GOAL_PROVENANCE_STREAK_TICK_TELEMETRY_ENABLED", str(streak_tick_telemetry_enabled)
    )
    import app.settings as settings_mod

    settings_mod._settings = None

    worker = AttentionRuntimeWorker.__new__(AttentionRuntimeWorker)
    worker._settings = settings_mod.get_settings()
    worker._store = MagicMock()
    worker._node_streak = DominanceStreak()
    worker._bus = MagicMock()
    return worker


def test_maybe_build_goal_returns_none_when_producer_disabled(monkeypatch):
    worker = _make_worker(monkeypatch, producer_enabled=False)
    real_domain = "node:substrate.biometrics"
    frame = _frame([_target(real_domain, 0.9)])

    goal, streak_tick = worker._maybe_build_goal(frame)
    assert goal is None
    # Producer disabled short-circuits before any streak advance -- no telemetry either.
    assert streak_tick is None


def test_maybe_build_goal_returns_none_when_bus_absent(monkeypatch):
    worker = _make_worker(monkeypatch)
    worker._bus = None
    real_domain = "node:substrate.biometrics"
    frame = _frame([_target(real_domain, 0.9)])

    goal, streak_tick = worker._maybe_build_goal(frame)
    assert goal is None
    assert streak_tick is None


def test_maybe_build_goal_returns_none_before_streak_threshold(monkeypatch):
    worker = _make_worker(monkeypatch, min_streak=3)
    real_domain = "node:substrate.biometrics"
    frame = _frame([_target(real_domain, 0.9)])

    goal, streak_tick = worker._maybe_build_goal(frame)  # streak=1
    assert goal is None
    assert isinstance(streak_tick, DominanceStreakTickV1)
    assert streak_tick.target_id == real_domain
    assert streak_tick.streak_count == 1
    assert streak_tick.min_streak_at_tick == 3
    assert streak_tick.qualified is False
    assert streak_tick.source_field_tick_id == "tick-1"
    assert streak_tick.source_attention_frame_id == "frame-1"

    goal, streak_tick = worker._maybe_build_goal(frame)  # streak=2
    assert goal is None
    assert streak_tick.streak_count == 2
    assert streak_tick.qualified is False


def test_maybe_build_goal_returns_real_goal_at_streak_threshold(monkeypatch):
    worker = _make_worker(monkeypatch, min_streak=3)
    real_domain = "node:substrate.biometrics"
    frame = _frame([_target(real_domain, 0.77)])

    worker._maybe_build_goal(frame)  # streak=1
    worker._maybe_build_goal(frame)  # streak=2
    goal, streak_tick = worker._maybe_build_goal(frame)  # streak=3

    assert isinstance(goal, FieldGoalProvenanceV1)
    assert goal.field_target_id == real_domain
    assert goal.target_kind == "node"
    assert goal.salience_score == pytest.approx(0.77)
    assert goal.priority == pytest.approx(0.77)
    assert goal.source_field_tick_id == "tick-1"
    assert goal.source_attention_frame_id == "frame-1"
    assert goal.proposal_status == "proposed"

    # The qualifying tick's own telemetry row says so too -- it's the same real event,
    # not a second, independent computation.
    assert streak_tick.target_id == real_domain
    assert streak_tick.streak_count == 3
    assert streak_tick.min_streak_at_tick == 3
    assert streak_tick.qualified is True


def test_maybe_build_goal_ignores_host_only_frame(monkeypatch):
    # Candidate B host target only -- no real node:substrate.* domain present.
    worker = _make_worker(monkeypatch, min_streak=1)
    frame = _frame([_target("node:athena", 0.95)])

    goal, streak_tick = worker._maybe_build_goal(frame)
    assert goal is None
    # No node:substrate.* winner -> update_dominance_streak's None-target_id reset case.
    # Still real, uncensored telemetry: this tick genuinely had no winner, not a gap.
    assert streak_tick.target_id is None
    assert streak_tick.streak_count == 0
    assert streak_tick.qualified is False


def test_maybe_build_goal_streak_tick_telemetry_disabled_by_flag(monkeypatch):
    worker = _make_worker(monkeypatch, min_streak=3, streak_tick_telemetry_enabled=False)
    real_domain = "node:substrate.biometrics"
    frame = _frame([_target(real_domain, 0.9)])

    goal, streak_tick = worker._maybe_build_goal(frame)
    assert goal is None
    assert streak_tick is None


def test_maybe_build_goal_lazy_loads_streak_from_store_once(monkeypatch):
    """Regression (2026-07-31 fix): the streak used to always start cold
    (`DominanceStreak()`), resetting real accumulated dominance to zero on
    every restart. Now `_node_streak` starts as `None` and is lazy-loaded
    from the store on the first real tick -- and only that first tick, not
    every tick, since the in-memory value is authoritative once loaded."""
    worker = _make_worker(monkeypatch, min_streak=5)
    worker._node_streak = None
    persisted = DominanceStreak(target_id="node:substrate.biometrics", count=2)
    worker._store.load_node_dominance_streak.return_value = persisted
    real_domain = "node:substrate.biometrics"
    frame = _frame([_target(real_domain, 0.9)])

    worker._maybe_build_goal(frame)  # loads persisted count=2, advances to 3
    worker._maybe_build_goal(frame)  # advances to 4, no reload

    worker._store.load_node_dominance_streak.assert_called_once()
    assert worker._node_streak.count == 4


def test_maybe_build_goal_persists_streak_every_tick(monkeypatch):
    worker = _make_worker(monkeypatch, min_streak=3)
    real_domain = "node:substrate.biometrics"
    frame = _frame([_target(real_domain, 0.9)])

    worker._maybe_build_goal(frame)

    worker._store.save_node_dominance_streak.assert_called_once()
    saved = worker._store.save_node_dominance_streak.call_args[0][0]
    assert saved.target_id == real_domain
    assert saved.count == 1


@pytest.mark.asyncio
async def test_publish_goal_calls_publish_with_reconnect(monkeypatch):
    worker = _make_worker(monkeypatch)
    goal = FieldGoalProvenanceV1(
        subject="attention",
        model_layer="field_attention",
        entity_id="node:substrate.biometrics",
        kind="memory.field_goals.proposed.v1",
        field_target_id="node:substrate.biometrics",
        target_kind="node",
        salience_score=0.8,
        source_field_tick_id="tick-1",
        source_attention_frame_id="frame-1",
        priority=0.8,
        provenance={"intake_channel": "internal.attention_runtime"},
    )

    mock_publish = AsyncMock()
    monkeypatch.setattr("orion.core.bus.resilience.publish_with_reconnect", mock_publish)

    await worker._publish_goal(goal)

    mock_publish.assert_called_once()
    args, kwargs = mock_publish.call_args
    assert args[0] is worker._bus
    assert args[1] == "orion:memory:goals:proposed"
    assert kwargs.get("log_label") == "attention_runtime_goal_provenance"


@pytest.mark.asyncio
async def test_publish_goal_noop_when_bus_absent(monkeypatch):
    worker = _make_worker(monkeypatch)
    worker._bus = None
    goal = FieldGoalProvenanceV1(
        subject="attention",
        model_layer="field_attention",
        entity_id="node:substrate.biometrics",
        kind="memory.field_goals.proposed.v1",
        field_target_id="node:substrate.biometrics",
        target_kind="node",
        salience_score=0.8,
        source_field_tick_id="tick-1",
        source_attention_frame_id="frame-1",
        priority=0.8,
        provenance={"intake_channel": "internal.attention_runtime"},
    )

    mock_publish = AsyncMock()
    monkeypatch.setattr("orion.core.bus.resilience.publish_with_reconnect", mock_publish)

    await worker._publish_goal(goal)

    mock_publish.assert_not_called()


@pytest.mark.asyncio
async def test_publish_streak_tick_calls_publish_with_reconnect(monkeypatch):
    worker = _make_worker(monkeypatch)
    streak_tick = DominanceStreakTickV1(
        target_id="node:substrate.biometrics",
        streak_count=2,
        min_streak_at_tick=3,
        qualified=False,
        source_field_tick_id="tick-1",
        source_attention_frame_id="frame-1",
    )

    mock_publish = AsyncMock()
    monkeypatch.setattr("orion.core.bus.resilience.publish_with_reconnect", mock_publish)

    await worker._publish_streak_tick(streak_tick)

    mock_publish.assert_called_once()
    args, kwargs = mock_publish.call_args
    assert args[0] is worker._bus
    assert args[1] == "orion:debug:attention:streak_tick"
    assert kwargs.get("log_label") == "attention_runtime_streak_tick"


@pytest.mark.asyncio
async def test_publish_streak_tick_noop_when_bus_absent(monkeypatch):
    worker = _make_worker(monkeypatch)
    worker._bus = None
    streak_tick = DominanceStreakTickV1(
        target_id="node:substrate.biometrics",
        streak_count=2,
        min_streak_at_tick=3,
        qualified=False,
        source_field_tick_id="tick-1",
        source_attention_frame_id="frame-1",
    )

    mock_publish = AsyncMock()
    monkeypatch.setattr("orion.core.bus.resilience.publish_with_reconnect", mock_publish)

    await worker._publish_streak_tick(streak_tick)

    mock_publish.assert_not_called()


@pytest.mark.asyncio
async def test_publish_streak_tick_never_raises_on_publish_failure(monkeypatch):
    """Debug telemetry must never surface as attention_runtime_tick_failed -- a publish
    failure here is swallowed (logged at debug), unlike _publish_goal's real emission."""
    worker = _make_worker(monkeypatch)
    streak_tick = DominanceStreakTickV1(
        target_id="node:substrate.biometrics",
        streak_count=2,
        min_streak_at_tick=3,
        qualified=False,
        source_field_tick_id="tick-1",
        source_attention_frame_id="frame-1",
    )

    mock_publish = AsyncMock(side_effect=RuntimeError("bus down"))
    monkeypatch.setattr("orion.core.bus.resilience.publish_with_reconnect", mock_publish)

    await worker._publish_streak_tick(streak_tick)  # must not raise


@pytest.mark.asyncio
async def test_start_connects_bus_when_producer_and_bus_enabled(monkeypatch):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    monkeypatch.setenv("ORION_GOAL_PROVENANCE_PRODUCER_ENABLED", "true")
    monkeypatch.setenv("ORION_BUS_ENABLED", "true")
    import app.settings as settings_mod

    settings_mod._settings = None

    worker = AttentionRuntimeWorker.__new__(AttentionRuntimeWorker)
    worker._settings = settings_mod.get_settings()
    worker._store = MagicMock()
    worker._policy = MagicMock()
    worker._health_monitor = MagicMock()
    worker._node_streak = DominanceStreak()
    worker._bus = None
    import asyncio

    worker._stop = asyncio.Event()

    mock_bus_instance = AsyncMock()
    mock_bus_cls = MagicMock(return_value=mock_bus_instance)
    monkeypatch.setattr("orion.core.bus.async_service.OrionBusAsync", mock_bus_cls)
    monkeypatch.setattr(asyncio, "create_task", MagicMock())

    await worker.start()

    mock_bus_cls.assert_called_once()
    mock_bus_instance.connect.assert_awaited_once()
    assert worker._bus is mock_bus_instance


@pytest.mark.asyncio
async def test_start_skips_bus_when_producer_disabled(monkeypatch):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    monkeypatch.setenv("ORION_GOAL_PROVENANCE_PRODUCER_ENABLED", "false")
    monkeypatch.setenv("ORION_BUS_ENABLED", "true")
    import app.settings as settings_mod

    settings_mod._settings = None

    worker = AttentionRuntimeWorker.__new__(AttentionRuntimeWorker)
    worker._settings = settings_mod.get_settings()
    worker._store = MagicMock()
    worker._policy = MagicMock()
    worker._health_monitor = MagicMock()
    worker._node_streak = DominanceStreak()
    worker._bus = None
    import asyncio

    worker._stop = asyncio.Event()

    mock_bus_cls = MagicMock()
    monkeypatch.setattr("orion.core.bus.async_service.OrionBusAsync", mock_bus_cls)
    monkeypatch.setattr(asyncio, "create_task", MagicMock())

    await worker.start()

    mock_bus_cls.assert_not_called()
    assert worker._bus is None


@pytest.mark.asyncio
async def test_stop_awaits_poll_task_before_closing_bus(monkeypatch):
    """Regression: stop() must not tear down the bus connection while the poll
    loop could still be mid-publish (a real race a review pass on this same
    patch found -- publish_with_reconnect would silently reconnect right after
    an intentional close). Ordering, not just "both eventually happen."
    """
    worker = _make_worker(monkeypatch)
    events: list[str] = []

    async def fake_poll_task() -> None:
        await asyncio.sleep(0.01)
        events.append("poll_task_done")

    worker._poll_task = asyncio.create_task(fake_poll_task())

    async def fake_close() -> None:
        events.append("bus_closed")

    worker._bus.close = AsyncMock(side_effect=fake_close)
    worker._stop = asyncio.Event()

    await worker.stop()

    assert events == ["poll_task_done", "bus_closed"]


# --- the one bridge: the producer reads the competition (2026-09-06) ------------
# The producer's receipt is a log line (not a schema field -- see worker.py);
# tests read it back with caplog.

import logging
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

from app.store import AttentionRuntimeStore


def _bridge_worker(monkeypatch, *, competing, reads_competition: bool = True):
    monkeypatch.setenv("ORION_GOAL_PROVENANCE_READS_COMPETITION", str(reads_competition))
    worker = _make_worker(monkeypatch, min_streak=1)
    worker._store.load_competing_loop_refs.return_value = competing
    return worker


def _emit_twice(worker, frame):
    """A brand-new target's first tick never emits (the streak starts at 1 and
    the first-tick branch returns False regardless of min_streak), so tick
    once to warm the streak, then emit."""
    worker._maybe_build_goal(frame)
    return worker._maybe_build_goal(frame)


def _receipt(caplog) -> str:
    lines = [r.getMessage() for r in caplog.records if "field_goal_provenance_competition_read" in r.getMessage()]
    assert lines, "no receipt logged"
    return lines[-1].split("competition_read=")[1].split()[0]


def _two_candidates():
    return _frame([_target("node:substrate.execution", 0.9), _target("node:substrate.biometrics", 0.6)])


def test_bridge_goal_targets_the_competing_candidate_and_says_so(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    worker = _bridge_worker(monkeypatch, competing={"node:substrate.biometrics"})
    goal, _ = _emit_twice(worker, _two_candidates())
    assert goal is not None and goal.field_target_id == "node:substrate.biometrics"
    assert _receipt(caplog) == "in_competition"
    worker._store.load_competing_loop_refs.assert_called_with(
        max_age_sec=worker._settings.goal_competition_max_age_sec
    )


def test_bridge_falls_back_and_records_not_in_competition(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    worker = _bridge_worker(monkeypatch, competing={"node:substrate.chat"})
    goal, _ = _emit_twice(worker, _two_candidates())
    assert goal.field_target_id == "node:substrate.execution"
    assert _receipt(caplog) == "not_in_competition"


def test_bridge_unknown_competition_is_unavailable_not_empty(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    worker = _bridge_worker(monkeypatch, competing=None)
    goal, _ = _emit_twice(worker, _two_candidates())
    assert goal is not None and _receipt(caplog) == "unavailable"


def test_bridge_kill_switch_never_reads_the_store(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    worker = _bridge_worker(monkeypatch, competing={"node:substrate.biometrics"}, reads_competition=False)
    goal, _ = _emit_twice(worker, _two_candidates())
    assert goal.field_target_id == "node:substrate.execution"  # pre-bridge behaviour exactly
    assert _receipt(caplog) == "unavailable"
    worker._store.load_competing_loop_refs.assert_not_called()


def test_bridge_skips_the_read_when_it_cannot_matter(monkeypatch):
    """With fewer than two qualified candidates no competition set can change
    the answer, so the cross-service read is not made (review finding)."""
    worker = _bridge_worker(monkeypatch, competing={"node:substrate.biometrics"})
    goal, _ = _emit_twice(worker, _frame([_target("node:substrate.execution", 0.9)]))
    assert goal is not None and goal.field_target_id == "node:substrate.execution"
    worker._store.load_competing_loop_refs.assert_not_called()


def test_bridge_read_failure_is_logged_not_fatal(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    worker = _bridge_worker(monkeypatch, competing=None)
    worker._store.load_competing_loop_refs.side_effect = RuntimeError("db down")
    goal, _ = _emit_twice(worker, _two_candidates())
    assert goal is not None and _receipt(caplog) == "unavailable"


def test_bridge_unknown_reads_do_not_flap_the_streak(monkeypatch):
    """The competition read changes every ~30s and is None on any error; if
    that flipped the winner, the 3-tick streak would never fill and the
    producer would fire LESS than before. A None/empty read keeps the current
    streak target."""
    monkeypatch.setenv("ORION_GOAL_PROVENANCE_READS_COMPETITION", "true")
    worker = _make_worker(monkeypatch, min_streak=3)
    reads = [{"node:substrate.biometrics"}, None, set(), RuntimeError("db down"), {"node:substrate.biometrics"}]
    worker._store.load_competing_loop_refs.side_effect = reads
    goals = [worker._maybe_build_goal(_two_candidates())[0] for _ in reads]
    assert [g.field_target_id for g in goals if g is not None] == ["node:substrate.biometrics"] * 3
    assert worker._node_streak.target_id == "node:substrate.biometrics" and worker._node_streak.count == 5


# --- store reader: one fake engine, shared by the reader tests ---------------


def _store_with(row):
    class _Result:
        def mappings(self):
            return self

        def first(self):
            return row

    class _Conn:
        def execute(self, *_a, **_k):
            return _Result()

    class _Engine:
        @contextmanager
        def connect(self):
            yield _Conn()

    store = AttentionRuntimeStore.__new__(AttentionRuntimeStore)
    store._engine = _Engine()
    return store


def test_store_reader_returns_node_refs_from_a_fresh_projection():
    store = _store_with({"loops_type": "array", "refs": ["node:substrate.execution", "node:substrate.biometrics"]})
    assert store.load_competing_loop_refs(max_age_sec=120.0) == {"node:substrate.execution", "node:substrate.biometrics"}


def test_store_reader_distinguishes_unknown_from_empty():
    # No fresh row (absent or older than max_age: the WHERE clause filters it) -> unknown.
    assert _store_with(None).load_competing_loop_refs(max_age_sec=120.0) is None
    # Fresh row whose shape is not what we expect (schema drift) -> unknown, not empty.
    assert _store_with({"loops_type": None, "refs": None}).load_competing_loop_refs(max_age_sec=120.0) is None
    assert _store_with({"loops_type": "object", "refs": None}).load_competing_loop_refs(max_age_sec=120.0) is None
    # Fresh row, real array, nothing competing -> empty, which is a real state.
    assert _store_with({"loops_type": "array", "refs": None}).load_competing_loop_refs(max_age_sec=120.0) == set()


def test_bridge_env_keys_reach_the_container_via_compose():
    """This service lists env keys explicitly (no env_file); a key missing from
    that list never reaches the container, so the kill switch would be
    decorative. Delegates to the repo's own parity gate rather than hand-listing
    keys (review finding)."""
    repo = Path(__file__).resolve().parents[3]
    proc = subprocess.run(
        [sys.executable, str(repo / "scripts/check_service_env_compose_parity.py"), "orion-attention-runtime"],
        capture_output=True, text=True, cwd=str(repo),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
