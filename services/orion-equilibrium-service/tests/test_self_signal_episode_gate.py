"""self_signal_episode gate (equilibrium adapter): seam-judged observations,
open budget, close-row outbox, persistence, flag wiring."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.self_signal_episode_gate import (  # noqa: E402
    GateConfig,
    SelfSignalEpisodeGate,
    observe_signals,
)
from orion.metacog.evidence_map import map_trigger  # noqa: E402
from orion.schemas.attention_frame import PredictionErrorMagnitudeV1  # noqa: E402

T0 = datetime(2026, 10, 14, 0, 0, tzinfo=timezone.utc)
BUS = "node:substrate.bus_synaptic"
BIO = "node:substrate.biometrics"


def _mag(pct: float, band: str, value: float = 0.12) -> PredictionErrorMagnitudeV1:
    return PredictionErrorMagnitudeV1(
        value=value, age_sec=5.0, percentile_now=pct, n_readings_7d=5000, band=band,
        p50_7d=0.03, p90_7d=0.06,
    )


def _mags(now: datetime, **by_node):
    return {node: (mag, now - timedelta(seconds=5), None) for node, mag in by_node.items()}


HOT = _mag(0.97, "high")
CALM = _mag(0.2, "quiet", value=0.02)


def _poll(gate, now, mags=None, *, error=None):
    obs = observe_signals(mags, now=now, history_error=error, known_signal_ids=list(gate.state.episodes))
    return gate.step(obs, zen_state="zen", pressure=0.1, recall_enabled=False)


def _publish_all(gate, out, now):
    for o in out:
        gate.mark_published(o, now=now.timestamp())


def _drive(gate, start, polls, mag, *, node=BUS, publish=True):
    rows = []
    for i in range(polls):
        now = start + timedelta(seconds=30 * i)
        out = _poll(gate, now, _mags(now, **{node: mag}))
        if publish:
            _publish_all(gate, out, now)
        rows += out
    return rows


def _warm(gate):
    """A first poll with nothing hot, so the gate is past its cold start."""
    _poll(gate, T0 - timedelta(seconds=30), _mags(T0 - timedelta(seconds=30), **{BUS: CALM}))
    assert not gate.cold_start


def test_external_and_unknown_nodes_are_not_observed():
    obs = observe_signals(
        {
            "node:substrate.perception": (HOT, T0, None),
            BUS: (HOT, T0, None),
            "world:chat": (HOT, T0, None),
        },
        now=T0,
    )
    assert [o.signal_id for o in obs] == [BUS]


def test_open_then_close_rows_through_the_real_seam():
    gate = SelfSignalEpisodeGate(GateConfig())
    _warm(gate)
    rows = _drive(gate, T0, 11, HOT)
    assert [(r.phase, r.signal_id) for r in rows] == [("open", BUS)]
    t = rows[0].trigger
    assert t.trigger_kind == "self_signal_episode" and t.upstream["phase"] == "open"
    m = map_trigger(t.trigger_kind, t.reason, t.upstream)
    assert m.severity == "degraded" and not m.density_rationale.startswith("no_evidence")
    rows = _drive(gate, T0 + timedelta(seconds=330), 21, CALM)
    assert [r.phase for r in rows] == ["close"]
    up = rows[0].trigger.upstream
    assert up["close_reason"] == "calm" and up["duration_sec"] == pytest.approx(930.0)
    assert map_trigger("self_signal_episode", rows[0].trigger.reason, up).severity == "nominal"


def test_db_failure_is_absent_not_calm():
    gate = SelfSignalEpisodeGate(GateConfig())
    _warm(gate)
    _drive(gate, T0, 11, HOT)
    # 30 min of failed reads: every signal unknown -> nothing closes.
    for i in range(60):
        now = T0 + timedelta(seconds=330 + 30 * i)
        assert _poll(gate, now, None, error="history unreadable") == []
    assert gate.state.episodes[BUS].confirmed


def test_failed_open_publish_is_retried_once_published_never_again():
    gate = SelfSignalEpisodeGate(GateConfig())
    _warm(gate)
    rows = _drive(gate, T0, 11, HOT, publish=False)
    assert [r.phase for r in rows] == ["open"]
    now = T0 + timedelta(seconds=330)
    retry = _poll(gate, now, _mags(now, **{BUS: HOT}))
    assert [r.phase for r in retry] == ["open"]
    _publish_all(gate, retry, now)
    assert _drive(gate, now + timedelta(seconds=30), 20, HOT) == []


def test_close_publish_failure_is_retried():
    gate = SelfSignalEpisodeGate(GateConfig())
    _warm(gate)
    _drive(gate, T0, 11, HOT)
    rows = _drive(gate, T0 + timedelta(seconds=330), 21, CALM, publish=False)
    assert [r.phase for r in rows] == ["close"]
    now = T0 + timedelta(seconds=1000)
    again = _poll(gate, now, _mags(now, **{BUS: CALM}))
    assert [r.phase for r in again] == ["close"]
    _publish_all(gate, again, now)
    assert _poll(gate, now + timedelta(seconds=30), _mags(now, **{BUS: CALM})) == []


def test_open_budget_silences_extra_episodes():
    gate = SelfSignalEpisodeGate(GateConfig(max_opens_per_hour=1))
    _warm(gate)
    rows = []
    for i in range(11):
        now = T0 + timedelta(seconds=30 * i)
        out = _poll(gate, now, _mags(now, **{BUS: HOT, BIO: HOT}))
        _publish_all(gate, out, now)
        rows += out
    assert [r.phase for r in rows] == ["open"]
    silenced = [ep for ep in gate.state.episodes.values() if ep.silent]
    assert len(silenced) == 1 and silenced[0].silent_reason == "over_open_budget"


def test_restart_resumes_without_a_burst_and_cold_start_adopts():
    gate = SelfSignalEpisodeGate(GateConfig())
    _warm(gate)
    _drive(gate, T0, 11, HOT)
    blob = gate.dump()
    resumed = SelfSignalEpisodeGate(GateConfig())
    resumed.load(blob)
    assert not resumed.cold_start
    assert _drive(resumed, T0 + timedelta(seconds=330), 30, HOT) == []
    # No persisted state at all: already-hot signals are adopted silently.
    cold = SelfSignalEpisodeGate(GateConfig())
    cold.load(None)
    assert _drive(cold, T0, 40, HOT) == []
    assert _drive(cold, T0 + timedelta(seconds=1200), 30, CALM) == []


# -- service wiring -------------------------------------------------------------


def test_flag_off_builds_no_gate_and_poll_writes_nothing(monkeypatch):
    from app import service as svc_mod

    monkeypatch.setattr(svc_mod.settings, "metacog_enable", True)
    monkeypatch.setattr(svc_mod.settings, "metacog_self_signal_episodes_enabled", False)
    # The outcome rollup rides the same poll; off here so the poll has nothing to do at all.
    monkeypatch.setattr(svc_mod.settings, "self_signal_outcome_association_enabled", False)
    svc = svc_mod.EquilibriumService()
    assert svc._self_signal_gate is None
    assert svc._self_signal_rollup is None
    svc._publish_metacog_trigger = AsyncMock()
    import asyncio

    asyncio.run(svc._self_signal_episode_poll_once())
    svc._publish_metacog_trigger.assert_not_called()


def test_flag_on_by_default():
    from app.settings import Settings

    s = Settings(_env_file=None)
    assert s.metacog_self_signal_episodes_enabled is True


@pytest.mark.asyncio
async def test_service_poll_publishes_with_cooldown_bypass(monkeypatch):
    from app import service as svc_mod

    monkeypatch.setattr(svc_mod.settings, "metacog_enable", True)
    monkeypatch.setattr(svc_mod.settings, "metacog_self_signal_episodes_enabled", True)
    svc = svc_mod.EquilibriumService()
    svc._self_signal_gate.load('{"version": 1, "episodes": {}}')  # warm, not cold
    svc.bus = AsyncMock()
    published = []

    async def _pub(trigger, *, bypass_cooldown=False):
        published.append((trigger.upstream["phase"], bypass_cooldown))
        return True

    svc._publish_metacog_trigger = _pub
    clock = {"now": T0}
    monkeypatch.setattr(svc_mod, "_utcnow", lambda: clock["now"])

    class _Cache:
        def refresh(self, fetch, *, now):
            return 0

        def magnitudes(self, *, now):
            return _mags(now, **{BUS: HOT})

        def history(self, node_id):
            # No readings: no dwell rule, the poll-counted dwell decides (today's behaviour).
            return []

    svc._pe_history_cache = _Cache()
    svc._pe_history_reader = type("R", (), {"fetch_since": lambda self, since: []})()
    for i in range(11):
        clock["now"] = T0 + timedelta(seconds=30 * i)
        await svc._self_signal_episode_poll_once()
    assert published == [("open", True)]
    svc.bus.redis.set.assert_awaited()


def test_failed_first_read_keeps_cold_start_cold():
    cold = SelfSignalEpisodeGate(GateConfig())
    cold.load(None)
    obs = observe_signals(None, now=T0, history_error="unreadable")
    assert cold.step(obs, zen_state="zen", pressure=0.1, recall_enabled=False, history_ok=False) == []
    assert cold.cold_start
    assert _drive(cold, T0 + timedelta(seconds=30), 40, HOT) == []  # adopted, no boot burst


def test_non_dict_state_blob_stays_cold():
    g = SelfSignalEpisodeGate(GateConfig())
    g.load("null")
    assert g.cold_start


def test_poll_task_is_once_per_process(monkeypatch):
    import asyncio

    from app import service as svc_mod

    monkeypatch.setattr(svc_mod.settings, "metacog_enable", True)
    monkeypatch.setattr(svc_mod.settings, "metacog_self_signal_episodes_enabled", True)
    svc = svc_mod.EquilibriumService()

    async def _go():
        svc._self_signal_episode_poll_loop = AsyncMock()
        svc._ensure_self_signal_task()
        first = svc._self_signal_task
        svc._ensure_self_signal_task()  # a restarted _run must not add a second
        assert svc._self_signal_task is first
        await first

    asyncio.run(_go())
