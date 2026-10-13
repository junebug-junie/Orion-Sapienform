"""telemetry_anomaly episode gate: one fire per episode, turn-coupled windows
left to chat_turn, restart-safe, sustained outages still fire."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orion.metacog.anomaly_episode import (  # noqa: E402
    AnomalyEpisodeStateV1,
    TurnActivityTracker,
    mark_published,
    state_from_dict,
    state_to_dict,
    step,
)
from app.telemetry_anomaly_metacog_gate import (  # noqa: E402
    evaluate_telemetry_anomaly_episode,
    lead_channel_category,
)

T0 = 1_791_400_000.0
P95 = 0.004784
THR = P95 * 3.0


def _flag(state, t, ratio=2.0, cat="physical_substrate", turn=False, **kw):
    return step(state, window_end=t, window_end_iso=str(t), flagged=True, ratio=ratio,
                lead_category=cat, turn_active=turn, **kw)


def _run(windows, **kw):
    """windows: list of (t, flagged, ratio, cat, turn). Publishes every fire."""
    state, fires = AnomalyEpisodeStateV1(), []
    for t, flagged, ratio, cat, turn in windows:
        d = step(state, window_end=t, window_end_iso=str(t), flagged=flagged, ratio=ratio,
                 lead_category=cat, turn_active=turn, **kw)
        state = d.state
        if d.fire:
            state = mark_published(state, d.severity)
            fires.append((t, d.reason, d.severity))
    return state, fires


def test_sustained_anomaly_fires_once_not_per_window():
    windows = [(T0 + 60 * i, True, 2.0, "physical_substrate", False) for i in range(30)]
    state, fires = _run(windows)
    assert fires == [(T0, "episode_open", "degraded")]
    assert state.flagged_windows == 30


def test_single_calm_window_is_flicker_two_calm_windows_end_episode():
    flicker = [(T0, True, 2.0, None, False), (T0 + 60, False, 0.5, None, False),
               (T0 + 120, True, 2.0, None, False)]
    _, fires = _run(flicker)
    assert len(fires) == 1
    split = [(T0, True, 2.0, None, False), (T0 + 60, False, 0.5, None, False),
             (T0 + 120, False, 0.5, None, False), (T0 + 180, True, 2.0, None, False)]
    _, fires = _run(split)
    assert [f[1] for f in fires] == ["episode_open", "episode_open"]


def test_refires_only_on_higher_severity_band():
    ratios = [1.2, 1.3, 1.8, 1.9, 1.6, 3.0, 2.6, 1.7]  # nominal, degraded, critical
    windows = [(T0 + 60 * i, True, r, None, False) for i, r in enumerate(ratios)]
    _, fires = _run(windows)
    assert [(f[1], f[2]) for f in fires] == [
        ("episode_open", "nominal"),
        ("severity_escalation", "degraded"),
        ("severity_escalation", "critical"),
    ]


def test_turn_explained_task_execution_window_is_suppressed():
    d = _flag(AnomalyEpisodeStateV1(), T0, cat="task_execution", turn=True)
    assert not d.fire and d.reason == "turn_explained" and d.turn_explained
    # Same lead with no turn: fires. Turn active but non-turn channel: fires.
    assert _flag(AnomalyEpisodeStateV1(), T0, cat="task_execution", turn=False).fire
    assert _flag(AnomalyEpisodeStateV1(), T0, cat="sensor_trust_liveness", turn=True).fire
    # Unknown category (glossary miss) never suppresses.
    assert _flag(AnomalyEpisodeStateV1(), T0, cat=None, turn=True).fire


def test_anomaly_outlasting_the_turn_fires_on_first_unexplained_window():
    windows = [(T0 + 60 * i, True, 2.0, "task_execution", i < 3) for i in range(6)]
    _, fires = _run(windows)
    assert fires == [(T0 + 180, "episode_open", "degraded")]


def test_sustained_turn_explained_episode_fires_after_n_windows():
    """10-08 shape: the outage hung turns for an hour, so every window is
    turn-covered. The sustained override must still fire it."""
    windows = [(T0 + 60 * i, True, 2.0, "task_execution", True) for i in range(15)]
    _, fires = _run(windows, sustained_windows=10)
    assert fires == [(T0 + 540, "sustained_past_turn", "degraded")]
    _, fires = _run(windows, sustained_windows=0)
    assert fires == []


def test_turn_suppress_off_fires_turn_coupled_windows():
    d = _flag(AnomalyEpisodeStateV1(), T0, cat="task_execution", turn=True, turn_suppress=False)
    assert d.fire and not d.turn_explained


def test_unpublished_fire_is_retried_on_next_window():
    d = _flag(AnomalyEpisodeStateV1(), T0)
    assert d.fire
    d2 = _flag(d.state, T0 + 60)  # caller never called mark_published
    assert d2.fire and d2.reason == "episode_open"


def test_stale_or_duplicate_window_never_fires():
    d = _flag(AnomalyEpisodeStateV1(), T0)
    s = mark_published(d.state, d.severity)
    dup = _flag(s, T0, ratio=4.0)
    assert not dup.fire and dup.reason == "stale_window"


def test_restart_restores_open_episode_without_refire():
    d = _flag(AnomalyEpisodeStateV1(), T0)
    s = mark_published(d.state, d.severity)
    restored = state_from_dict(json.loads(json.dumps(state_to_dict(s))))
    assert restored == s
    assert not _flag(restored, T0 + 60).fire


def test_corrupt_state_restores_as_cold():
    for raw in (None, {}, {"episode_id": "x"}, {"episode_id": "x", "started_at": "nan?"}, "junk"):
        assert state_from_dict(raw) == AnomalyEpisodeStateV1()


def test_turn_tracker_brackets_open_and_closed_turns():
    tr = TurnActivityTracker(turn_carry_sec=150, open_max_sec=3600)
    tr.observe("a", T0)  # open turn
    assert tr.active_between(T0 + 600, T0 + 660)  # still open -> active
    tr.observe("a", T0 + 300, closed=True)
    assert tr.active_between(T0 + 400, T0 + 460)  # within carry of the close
    assert not tr.active_between(T0 + 500, T0 + 560)
    # Close-only (turn started before boot): fcc_elapsed_sec back-dates the start.
    tr.observe("b", T0 + 5000, closed=True, elapsed_sec=1000)
    assert tr.active_between(T0 + 4100, T0 + 4160)
    # An open turn that went silent past open_max_sec stops counting.
    tr.observe("c", T0 + 10_000)
    assert not tr.active_between(T0 + 10_000 + 3600 + 400, T0 + 10_000 + 3600 + 460)


def test_turn_tracker_prunes_old_turns():
    tr = TurnActivityTracker(turn_carry_sec=150, open_max_sec=3600)
    for i in range(50):
        tr.observe(f"t{i}", T0 + i, closed=True)
    tr.observe("late", T0 + 100_000)
    assert len(tr) == 1


def _score(recon_loss, *, lead="failure_pressure=0.6", we="2026-10-08T14:41:00+00:00",
           ws="2026-10-08T14:40:00+00:00"):
    return {
        "correlation_id": "c1", "encoder_id": "mood-arc-encoder:v4", "encoder_version": "v4",
        "recon_loss": recon_loss, "recon_error_p95": P95, "threshold_multiplier": 3.0,
        "threshold": THR, "anomalous": recon_loss > THR, "window_start": ws, "window_end": we,
        "window_size": 30, "top_channels": [lead, "reliability_pressure=0.1"],
        "deviation_direction": "elevated", "mean_signed_deviation": 0.1,
    }


def _eval(state, score, turns=None, sustained=10):
    return evaluate_telemetry_anomaly_episode(
        state, correlation_id="c1", score=score, turns=turns, zen_state="zen", pressure=0.1,
        recall_enabled=False, threshold_multiplier=3.0, calm_gap_sec=150.0,
        sustained_windows=sustained, now=T0,
    )


def test_glossary_category_comes_from_semantic_layer():
    assert lead_channel_category(["failure_pressure=0.6"]) == "task_execution"
    assert lead_channel_category(["availability=0.9"]) == "sensor_trust_liveness"
    assert lead_channel_category(["not_a_channel=1"]) is None
    assert lead_channel_category([]) is None


def test_adapter_fire_carries_episode_upstream_and_keeps_reason_key():
    state, decision, trig = _eval(AnomalyEpisodeStateV1(), _score(THR * 2.0))
    assert trig is not None and decision.reason == "episode_open"
    # Reason format unchanged: orion-dream's novelty key is built from it.
    assert trig.reason.startswith("telemetry_anomaly:elevated:recon_loss=")
    assert trig.reason.endswith(":top=failure_pressure=0.6")
    up = trig.upstream
    assert up["episode_windows"] == 1 and up["episode_fire_reason"] == "episode_open"
    assert up["lead_channel_category"] == "task_execution" and up["turn_active"] is False
    assert up["recon_loss"] == THR * 2.0 and up["threshold"] == pytest.approx(THR)


def test_adapter_suppresses_during_turn_and_ignores_malformed():
    turns = TurnActivityTracker(turn_carry_sec=150)
    we = 1_791_470_460.0  # 2026-10-08T14:41:00Z
    turns.observe("turn-1", we - 30)
    state, decision, trig = _eval(AnomalyEpisodeStateV1(), _score(THR * 2.0), turns=turns)
    assert trig is None and decision.reason == "turn_explained"
    state2, decision2, trig2 = _eval(state, {"recon_loss": "x"})
    assert decision2 is None and trig2 is None and state2 == state
    # Below threshold: calm, no trigger.
    _, d3, t3 = _eval(AnomalyEpisodeStateV1(), _score(THR * 0.5))
    assert d3.reason == "calm" and t3 is None


# --- service wiring -------------------------------------------------------

def _service(monkeypatch, *, publish_ok=True):
    from app.service import EquilibriumService, settings

    monkeypatch.setattr(settings, "metacog_telemetry_anomaly_episode_enable", True)
    svc = EquilibriumService()
    svc.bus = MagicMock()
    svc.bus.publish = AsyncMock(side_effect=None if publish_ok else RuntimeError("down"))
    svc.bus.redis = MagicMock()
    svc.bus.redis.set = AsyncMock()
    svc.bus.redis.get = AsyncMock(return_value=None)
    return svc, settings


@pytest.mark.asyncio
async def test_service_publishes_once_per_episode_bypassing_shared_lane(monkeypatch):
    svc, settings = _service(monkeypatch)
    monkeypatch.setattr(settings, "metacog_cooldown_sec", 10_000.0)
    svc._last_metacog_trigger_ts = 1e18  # shared lane "just fired"
    for i in range(5):
        we = f"2026-10-08T14:4{i}:00+00:00"
        await svc._handle_anomaly_score(_score(THR * 2.0, lead="availability=0.9", we=we, ws=we),
                                        zen=0.9, distress=0.1)
    assert svc.bus.publish.await_count == 1
    assert svc._anomaly_episode_state.fires == 1
    assert svc.bus.redis.set.await_count >= 1
    saved = json.loads(svc.bus.redis.set.await_args.args[1])
    assert saved["flagged_windows"] == 5


@pytest.mark.asyncio
async def test_service_failed_publish_does_not_consume_the_fire(monkeypatch):
    svc, _ = _service(monkeypatch, publish_ok=False)
    await svc._handle_anomaly_score(_score(THR * 2.0, lead="availability=0.9"), zen=0.9, distress=0.1)
    assert svc._anomaly_episode_state.fired_rank == -1


@pytest.mark.asyncio
async def test_service_restart_restores_episode_no_boot_burst(monkeypatch):
    svc, _ = _service(monkeypatch)
    await svc._handle_anomaly_score(
        _score(THR * 2.0, lead="availability=0.9", we="2026-10-08T14:40:00+00:00"), zen=0.9, distress=0.1)
    saved = svc.bus.redis.set.await_args.args[1]
    svc2, _ = _service(monkeypatch)
    svc2.bus.redis.get = AsyncMock(return_value=saved.encode())
    await svc2._load_anomaly_episode_state()
    await svc2._handle_anomaly_score(
        _score(THR * 2.0, lead="availability=0.9", we="2026-10-08T14:41:00+00:00"), zen=0.9, distress=0.1)
    assert svc2.bus.publish.await_count == 0


def test_service_turn_observation_from_grammar_and_run_artifact(monkeypatch):
    svc, settings = _service(monkeypatch)
    assert svc._anomaly_turns is not None
    svc._observe_turn_activity(settings.channel_grammar_event, {
        "correlation_id": "t1", "provenance": {"source_service": "orion-harness-governor"}})
    svc._observe_turn_activity(settings.channel_grammar_event, {
        "correlation_id": "x1", "provenance": {"source_service": "orion-biometrics"}})
    assert {cid for cid, _, _ in svc._anomaly_turns.turns()} == {"t1"}
    svc._observe_turn_activity(settings.channel_harness_run_artifact,
                               {"correlation_id": "t1", "fcc_elapsed_sec": 42.0})
    (cid, _, closed_at), = list(svc._anomaly_turns.turns())
    assert closed_at is not None
