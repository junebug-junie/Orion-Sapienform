"""A continuously-reading body signal must stay unusual to win attention.

Spec: docs/superpowers/specs/2026-10-16-attention-continuous-signal-quota-design.md, option (b).
One rule (orion/attention/sustained.py), three readers: the attention contest
(world_first.judge_candidate), self-readings, and the metacog episode gate. Each test pins one
clause; the last ones pin that the three readers agree on the same history at the same time.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

import orion.attention.self_readings as sr
import orion.attention.world_first as wf
from orion.attention.pe_history_cache import NodePeHistoryCache
from orion.attention.sustained import (
    EVENT_ORIENTING_WINDOW_SEC,
    SUSTAIN_DWELL_SEC,
    SUSTAIN_WINDOW_SEC,
    hot_dwell_sec,
    is_sustained,
)
from orion.metacog import self_signal_episode as sse
from orion.substrate.prediction_error_magnitude import compute_prediction_error_magnitude

_fx_spec = importlib.util.spec_from_file_location(
    "self_readings_history_sustained", Path(__file__).parent / "fixtures" / "self_readings_history.py"
)
fx = importlib.util.module_from_spec(_fx_spec)
_fx_spec.loader.exec_module(fx)  # type: ignore[union-attr]

NOW = fx.NOW
BUS = "node:substrate.bus_synaptic"
BIO = "node:substrate.biometrics"
EXE = "node:substrate.execution"
PER = "node:substrate.perception"
CAB = "node:substrate.cabinet"


def _hist(rows, node, now=NOW):
    return [(ts, v) for n, ts, v in rows if n == node and ts <= now]


def _candidate(rows, node, *, now=NOW, sustain=True, with_history=True):
    hist = _hist(rows, node, now)
    observed_at, value = hist[-1]
    mag = compute_prediction_error_magnitude(value=value, observed_at=observed_at, history=hist, now=now)
    return wf.node_candidate(
        node_id=node, label=node, magnitude=mag, observed_at=observed_at, now=now,
        history_values=[v for _, v in hist],
        pe_history=hist if with_history else None, sustain=sustain,
    )


def _sustained_bus_rows(hot_since_sec: float, *, gap: tuple[float, float] | None = None, now=NOW):
    """Calm week; bus_synaptic at its live 60 s cadence for the last 30 minutes, at its week's top
    from ``hot_since_sec`` ago to now, except inside ``gap`` (seconds ago, calm)."""
    cut = now - timedelta(minutes=30)
    rows = [r for r in fx.calm_rows(now) if not (r[0] == BUS and r[1] > cut)]
    for s in range(1800, -1, -60):
        sec = s + 20 if s else 20
        hot = sec <= hot_since_sec and not (gap and gap[1] <= sec <= gap[0])
        rows.append((BUS, now - timedelta(seconds=sec), 0.2 if hot else 0.02))
    return rows


# --- the pure rule ---------------------------------------------------------------------------


def test_window_and_dwell_reuse_the_orienting_window_not_a_new_knob() -> None:
    assert SUSTAIN_DWELL_SEC is EVENT_ORIENTING_WINDOW_SEC
    assert wf.EVENT_ORIENTING_WINDOW_SEC is EVENT_ORIENTING_WINDOW_SEC
    assert wf.SUSTAIN_DWELL_SEC is SUSTAIN_DWELL_SEC
    assert SUSTAIN_WINDOW_SEC == 2 * EVENT_ORIENTING_WINDOW_SEC == 600.0
    # The episode gate's open dwell/window ARE the contest's: one object, cannot drift.
    assert sse.DEFAULT_OPEN_DWELL_SEC is wf.SUSTAIN_DWELL_SEC
    assert sse.DEFAULT_OPEN_WINDOW_SEC is wf.SUSTAIN_WINDOW_SEC


def test_hot_dwell_counts_each_reading_until_the_next_inside_the_window() -> None:
    now = 10_000.0
    # hot at -500, calm at -400, hot at -100 (still hot at now)
    rd = [(now - 500, True), (now - 400, False), (now - 100, True)]
    assert hot_dwell_sec(rd, now=now) == pytest.approx(100 + 100)
    # the reading before the window speaks into it until the next one
    assert hot_dwell_sec([(now - 900, True), (now - 300, False)], now=now) == pytest.approx(300)
    # a hold cap stops a lone old reading from covering the whole window
    assert hot_dwell_sec([(now - 600, True)], now=now, max_hold_sec=60) == pytest.approx(60)
    # unjudgeable readings are not hot; a future-stamped reading is taken at now
    assert hot_dwell_sec([(now - 200, None), (now + 5, True)], now=now) == 0.0
    assert hot_dwell_sec([], now=now) == 0.0


def test_is_sustained_is_strictly_more_than_the_dwell_and_hot_now() -> None:
    assert not is_sustained(300.0, hot_now=True)  # MORE than 300 s, like the episode gate
    assert is_sustained(300.5, hot_now=True)
    assert not is_sustained(600.0, hot_now=False)
    assert not is_sustained(None, hot_now=True)


# --- the contest -----------------------------------------------------------------------------


def test_continuous_set_comes_from_glossary_semantics() -> None:
    from orion.field.channel_glossary import load_glossary

    nodes = {e.node for e in load_glossary()["entries"] if e.channel == "prediction_error" and e.node}
    got = {n for n in nodes if wf.node_prediction_error_continuous(n)}
    # per_tick body nodes only: perception is the world, event-written and trigger nodes have
    # their own rules, a placeholder is never a measurement.
    assert got == {BIO, BUS}


def test_a_one_reading_spike_does_not_win() -> None:
    cand = _candidate(fx.bus_spike_rows(), BUS)
    assert cand.sustain_window_sec == SUSTAIN_WINDOW_SEC
    assert cand.hot_dwell_sec is not None and cand.hot_dwell_sec < 60
    v = wf.judge_candidate(cand)
    assert v.band in ("high", "unusual")  # high for itself now ...
    assert not v.eligible and v.sustained is False  # ... but a flicker
    assert "not sustained" in v.reason
    trace = v.trace()
    assert trace["hot_dwell_sec"] == cand.hot_dwell_sec and trace["sustained"] is False
    assert trace["sustain_window_sec"] == SUSTAIN_WINDOW_SEC
    assert wf.rank_candidates([cand]).no_winner


def test_hot_for_more_than_300s_of_600s_wins() -> None:
    cand = _candidate(_sustained_bus_rows(400), BUS)
    assert cand.hot_dwell_sec is not None and cand.hot_dwell_sec > 300
    v = wf.judge_candidate(cand)
    assert v.eligible and v.sustained is True and "sustained" in v.reason
    assert wf.rank_candidates([cand]).winner.candidate.source_id == BUS


def test_hot_for_exactly_the_dwell_or_less_does_not_win() -> None:
    v = wf.judge_candidate(_candidate(_sustained_bus_rows(260), BUS))
    assert v.candidate.hot_dwell_sec <= 300 and not v.eligible


def test_a_calm_gap_resets_the_dwell() -> None:
    # hot from 900 s ago, calm 380..80 s ago, hot again for the last ~80 s: only ~260 s hot inside
    # the last 600 s, so not sustained, though it is hot now and was hot for minutes before.
    gapped = _candidate(_sustained_bus_rows(900, gap=(380, 80)), BUS)
    assert gapped.hot_dwell_sec <= 300
    assert not wf.judge_candidate(gapped).eligible
    # The same run without the gap wins.
    assert wf.judge_candidate(_candidate(_sustained_bus_rows(900), BUS)).eligible


def test_not_hot_now_never_wins_even_after_a_long_run() -> None:
    rows = _sustained_bus_rows(900, gap=(30, 0))  # the newest reading is calm
    v = wf.judge_candidate(_candidate(rows, BUS))
    assert not v.eligible and v.sustained is None  # calm now: the dwell is never consulted


@pytest.mark.parametrize("node", [PER, EXE, CAB])
def test_perception_and_event_driven_sources_are_unaffected(node) -> None:
    rows = fx.busy_body_rows()
    with_hist = _candidate(rows, node)
    without = _candidate(rows, node, with_history=False)
    assert with_hist.sustain_window_sec is None and with_hist.hot_dwell_sec is None
    a, b = wf.judge_candidate(with_hist), wf.judge_candidate(without)
    assert (a.eligible, a.reason, a.band, a.rank_score) == (b.eligible, b.reason, b.band, b.rank_score)


def test_missing_history_is_todays_verdict_not_silence() -> None:
    cand = _candidate(fx.bus_spike_rows(), BUS, with_history=False)
    assert cand.sustain_window_sec is None and cand.hot_dwell_sec is None
    assert wf.judge_candidate(cand).eligible  # the old one-reading verdict


def test_kill_switch_restores_todays_verdict() -> None:
    off = _candidate(fx.bus_spike_rows(), BUS, sustain=False)
    assert off.sustain_window_sec is None and off.hot_dwell_sec is None
    v = wf.judge_candidate(off)
    assert v.eligible and v.sustained is None
    assert v.reason == "body unusual for itself (unusual)" or v.reason == "body unusual for itself (high)"


def test_dwell_failure_degrades_to_todays_verdict(monkeypatch) -> None:
    def boom(*a, **k):
        raise RuntimeError("bad history")

    monkeypatch.setattr(wf, "sustained_hot_dwell", boom)
    cand = _candidate(fx.bus_spike_rows(), BUS)
    assert cand.sustain_window_sec is None and wf.judge_candidate(cand).eligible


def _bus_node(observed_at):
    return SimpleNamespace(
        node_id=BUS, label="bus_synaptic", node_kind="concept",
        metadata={"prediction_error": 0.2, "dynamic_pressure": 0.5},
        signals=SimpleNamespace(confidence=0.8),
        temporal=SimpleNamespace(observed_at=observed_at),
    )


def test_broadcast_threads_history_and_the_kill_switch() -> None:
    import orion.substrate.attention_broadcast as ab

    rows = fx.bus_spike_rows()
    cand = _candidate(rows, BUS)
    kw = dict(
        nodes=[_bus_node(cand.observed_at)], now=NOW,
        magnitude_by_node_id={BUS: cand.unusualness}, world_first=True, external_candidates=[],
        pe_history_by_node_id={BUS: _hist(rows, BUS)},
    )
    on = ab.build_substrate_attention_frame(**kw)
    off = ab.build_substrate_attention_frame(**kw, sustain=False)
    assert on.debug["world_first"]["no_winner"] is True
    traced = next(c for c in on.debug["world_first"]["candidates"] if c["source_id"] == BUS)
    assert traced["sustained"] is False and traced["hot_dwell_sec"] is not None
    assert off.debug["world_first"]["winner"] == BUS


# --- one verdict: contest, self-readings and the episode gate agree ---------------------------


def _load_gate():
    path = Path(__file__).resolve().parents[1] / "services" / "orion-equilibrium-service" / "app" / "self_signal_episode_gate.py"
    spec = importlib.util.spec_from_file_location("_sustained_test_gate", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _cache(rows, now) -> NodePeHistoryCache:
    cache = NodePeHistoryCache()
    cache.refresh(lambda since: [r for r in rows if since <= r[1] <= now], now=now)
    return cache


@pytest.mark.parametrize("sustain", [True, False])
def test_contest_self_readings_and_episode_gate_agree_on_the_same_history(sustain) -> None:
    """Walk a bus_synaptic run poll by poll (30 s, the episode gate's cadence): at every poll the
    contest's verdict, the self-reading's ``unusual`` and the episode gate's open agree."""
    gate = _load_gate()
    rows = _sustained_bus_rows(600)  # hot for the last ~10 minutes
    start = NOW - timedelta(seconds=660)
    episode = None
    opened_at = None
    first_eligible = None
    for i in range(0, 23):
        t = start + timedelta(seconds=30 * i)
        cache = _cache(rows, t)
        mags = cache.magnitudes(now=t)
        cand = wf.node_candidate(
            node_id=BUS, label=BUS, magnitude=mags[BUS][0], observed_at=mags[BUS][1], now=t,
            rank_percentile=mags[BUS][2], pe_history=cache.history(BUS), sustain=sustain,
        )
        contest = wf.judge_candidate(cand).eligible
        reading = sr.build_self_readings(cache, now=t, nodes=[BUS], sustain=sustain)[0]
        assert reading.unusual == contest, t
        obs = gate.observe_signals(
            {BUS: mags[BUS]}, now=t, pe_history=cache.history, sustain=sustain,
        )
        assert [o.signal_id for o in obs] == [BUS]
        episode, event = sse.step_signal(episode, obs[0])
        if contest and first_eligible is None:
            first_eligible = t
        if event is not None and event.phase == "open":
            opened_at = t
    assert first_eligible is not None
    if sustain:
        # The episode opens on exactly the poll the contest first lets the signal compete.
        assert opened_at == first_eligible
        assert (first_eligible - (NOW - timedelta(seconds=600))).total_seconds() > SUSTAIN_DWELL_SEC
    else:
        # Kill switch: the contest crowns the first hot reading; the episode keeps its own
        # poll-counted dwell (unchanged), so it opens later.
        assert opened_at is not None and opened_at > first_eligible


def test_episode_counts_a_not_yet_sustained_reading_as_a_hot_poll() -> None:
    """The pending episode starts at the onset, not at confirmation: a not-sustained high
    reading is a hot poll for the gate (only the contest drops it)."""
    v = wf.judge_candidate(_candidate(fx.bus_spike_rows(), BUS))
    obs = sse.observation_from_verdict(v, now=NOW.timestamp())
    assert obs.kind == "hot" and obs.hot_dwell_sec == v.hot_dwell_sec
    episode, event = sse.step_signal(None, obs)
    assert episode is not None and not episode.confirmed and event is None


def test_poll_counted_dwell_is_unchanged_for_signals_without_a_reading_dwell() -> None:
    t0 = 1_791_400_000.0
    ep, opened = None, None
    for i in range(14):
        obs = sse.SignalObservation(signal_id=EXE, kind="hot", at=t0 + 30 * i, reason="r", band="high", percentile=0.95)
        ep, ev = sse.step_signal(ep, obs)
        if ev is not None:
            opened = i
    assert opened == 10  # 11 hot polls x 30 s = 330 s > 300 s, as before
