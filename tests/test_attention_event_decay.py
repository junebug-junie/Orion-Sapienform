"""Event-written signals fade (Juniper, 2026-10-10: "let event type signals decay").

Live case: node:substrate.codebase wrote 0.988 at 05:51:03 (one codebase change
event). The value is carried forward until the next event, and against a
mostly-zero week it sits in the top percentile, so it won every field frame
for 18 minutes. One event should orient attention, then fade.

Which sources fade is read from the glossary semantics (sparsity +
absent_means), never from a node list in the runtime. The expected set below
is the test's pin: a new prediction_error entry must be classified on purpose.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import orion.attention.world_first as wf
from orion.attention.world_first import (
    EVENT_FADE_GRACE_SEC,
    EVENT_ORIENTING_WINDOW_SEC,
    event_fade,
    judge_candidate,
    node_candidate,
    node_prediction_error_event_written,
    prediction_error_is_event_written,
    rank_candidates,
)
from orion.schemas.attention_frame import PredictionErrorMagnitudeV1
from orion.substrate.prediction_error_magnitude import compute_prediction_error_magnitude

NOW = datetime(2026, 10, 10, 5, 51, 3, tzinfo=timezone.utc)
CODEBASE = "node:substrate.codebase"
BIOMETRICS = "node:substrate.biometrics"
BUS = "node:substrate.bus_synaptic"

# A codebase-shaped week: written every 15 min, 91% exact zeros.
_WEEK = [NOW - timedelta(minutes=15 * i) for i in range(1, 672)]
_HISTORY = [(t, 0.0 if i % 11 else 0.2) for i, t in enumerate(_WEEK)]


def _codebase_at(now: datetime, *, event_at: datetime = NOW, value: float = 0.988,
                 event_decay: bool = True):
    hist = [*_HISTORY, (event_at, value)]
    mag = compute_prediction_error_magnitude(value=value, observed_at=event_at, history=hist, now=now)
    return node_candidate(
        node_id=CODEBASE, label="codebase", magnitude=mag, observed_at=event_at, now=now,
        history_values=[v for _, v in hist], event_decay=event_decay,
    )


def _level(node_id: str, pct: float, *, age: float = 20.0, value: float = 0.3):
    band = "quiet" if pct < 0.5 else "usual" if pct < 0.9 else "high" if pct < 0.99 else "unusual"
    m = PredictionErrorMagnitudeV1(
        value=value, age_sec=age, percentile_now=pct, n_readings_7d=5000, band=band, trend="flat"
    )
    return node_candidate(node_id=node_id, label=node_id, magnitude=m, observed_at=NOW, now=NOW)


# --- which sources are event-written: the semantic layer decides -----------


def test_event_written_set_comes_from_glossary_semantics() -> None:
    expected = {
        "node:substrate.execution": True,   # designed_sparse, "only written when a batch carried..."
        "node:substrate.chat": True,        # designed_sparse, "only written when chat turns were touched"
        "node:substrate.codebase": True,    # designed_sparse, "only written on a codebase delta event"
        "node:substrate.route": True,       # designed_sparse, "only written when routing runs were touched"
        "node:substrate.harness_closure": True,  # event_gated (a placeholder: never eligible anyway)
        "node:substrate.cabinet": False,    # designed_sparse but rewritten every 30 s tick
        "node:substrate.perception": False,  # "written as 0.0 every tick"
        "node:substrate.biometrics": False,  # per_tick
        "node:substrate.bus_synaptic": False,  # per_tick
    }
    from orion.field.channel_glossary import load_glossary

    pe_nodes = {
        e.node for e in load_glossary()["entries"]
        if e.channel == "prediction_error" and e.node
    }
    # Every node-qualified PE entry is classified on purpose here.
    assert pe_nodes == set(expected)
    got = {n: node_prediction_error_event_written(n) for n in expected}
    assert got == expected


def test_carried_forward_alone_does_not_make_a_signal_event_written() -> None:
    # bus_synaptic / biometrics say "carried forward" too, but are per_tick.
    assert not prediction_error_is_event_written("per_tick", "carried forward in node_vectors")
    assert not prediction_error_is_event_written("designed_sparse", "written as 0.0 every tick")
    assert not prediction_error_is_event_written(
        "designed_sparse", "rewritten every 30 s tick; carried forward in node_vectors between ticks"
    )
    assert prediction_error_is_event_written("designed_sparse", "Only written on an event; carried forward")
    assert prediction_error_is_event_written("event_gated", None)
    assert not prediction_error_is_event_written(None, None)


def test_glossary_failure_reads_not_event_written_and_is_not_cached(monkeypatch) -> None:
    import orion.field.channel_glossary as cg

    wf._EVENT_WRITTEN_CACHE.pop(CODEBASE, None)
    real = cg.resolve_channel_entry

    def boom(*a, **k):
        raise OSError("glossary unreadable")

    monkeypatch.setattr(cg, "resolve_channel_entry", boom)
    assert node_prediction_error_event_written(CODEBASE) is False
    monkeypatch.setattr(cg, "resolve_channel_entry", real)
    assert node_prediction_error_event_written(CODEBASE) is True


# --- the codebase-shaped case -------------------------------------------------


def test_one_event_orients_then_stops_competing_with_the_value_unchanged() -> None:
    onset = _codebase_at(NOW + timedelta(seconds=10))
    assert onset.event_window_sec == EVENT_ORIENTING_WINDOW_SEC
    v0 = judge_candidate(onset)
    assert v0.eligible and v0.band in {"high", "unusual"}

    # 18 minutes later (the live hold) the carried value and its percentile
    # are unchanged, and the 1800 s staleness bound has not fired -- only the
    # event's age has moved.
    late = _codebase_at(NOW + timedelta(minutes=18))
    assert late.unusualness.value == onset.unusualness.value
    assert late.unusualness.percentile_now == pytest.approx(onset.unusualness.percentile_now, abs=1e-4)
    assert late.unusualness.age_sec < wf.INTERNAL_MAX_AGE_SEC
    v1 = judge_candidate(late)
    assert not v1.eligible
    assert "orienting window" in v1.reason
    assert v1.band == v0.band  # the event was unusual; it is just old news now

    # The boundary is the derived window, not something else.
    just_in = judge_candidate(_codebase_at(NOW + timedelta(seconds=EVENT_ORIENTING_WINDOW_SEC - 1)))
    just_out = judge_candidate(_codebase_at(NOW + timedelta(seconds=EVENT_ORIENTING_WINDOW_SEC)))
    assert just_in.eligible and not just_out.eligible


def test_rank_strength_fades_with_event_age() -> None:
    a = judge_candidate(_codebase_at(NOW + timedelta(seconds=5)))
    b = judge_candidate(_codebase_at(NOW + timedelta(seconds=150)))
    c = judge_candidate(_codebase_at(NOW + timedelta(seconds=290)))
    assert a.rank_score > b.rank_score > c.rank_score > 0.0
    # Full strength through the grace, then linear to 0 at the window.
    assert a.event_decay == 1.0
    expected = 1 - (150 - EVENT_FADE_GRACE_SEC) / (EVENT_ORIENTING_WINDOW_SEC - EVENT_FADE_GRACE_SEC)
    assert b.event_decay == pytest.approx(expected)
    assert b.salience == pytest.approx(b.score * expected)
    trace = b.trace()
    assert trace["event_age_sec"] == pytest.approx(150.0)
    assert trace["event_window_sec"] == EVENT_ORIENTING_WINDOW_SEC


def test_fade_curve() -> None:
    assert event_fade(0, 300, 60) == 1.0 and event_fade(60, 300, 60) == 1.0
    assert event_fade(180, 300, 60) == pytest.approx(0.5)
    assert event_fade(300, 300, 60) == 0.0 and event_fade(900, 300, 60) == 0.0
    assert event_fade(500, 300, 900) == 0.0  # grace clamped to the window


def test_a_fresh_event_is_not_penalised_against_a_per_tick_rival() -> None:
    """Inside the grace an execution spike keeps its full rank: a per-tick
    rival's reading is itself up to ~60 s old and never faded."""
    m = PredictionErrorMagnitudeV1(
        value=1.0, age_sec=30, percentile_now=0.965, n_readings_7d=1000, band="high", trend="flat"
    )
    ex = node_candidate(node_id="node:substrate.execution", label="x", magnitude=m, observed_at=NOW, now=NOW,
                        history_values=[0.0] * 900 + [0.5] * 65 + [1.0] * 35)
    rival = _level(BUS, 0.975)
    assert rank_candidates([rival, ex]).winner.candidate.source_id == "node:substrate.execution"


def test_a_fading_event_yields_to_a_fresh_body_alarm() -> None:
    old_event = _codebase_at(NOW + timedelta(seconds=200))
    fresh = _level(BIOMETRICS, 0.95)
    ranking = rank_candidates([old_event, fresh])
    assert ranking.winner.candidate.source_id == BIOMETRICS
    # ...but with nothing else eligible the orienting response still holds.
    assert rank_candidates([old_event]).winner.candidate.source_id == CODEBASE


def test_a_new_event_re_arms_it() -> None:
    second = NOW + timedelta(minutes=17)
    hist = [*_HISTORY, (NOW, 0.988), (second, 0.95)]
    now = second + timedelta(seconds=20)
    mag = compute_prediction_error_magnitude(value=0.95, observed_at=second, history=hist, now=now)
    cand = node_candidate(node_id=CODEBASE, label="c", magnitude=mag, observed_at=second, now=now,
                          history_values=[v for _, v in hist])
    v = judge_candidate(cand)
    assert v.eligible and v.event_age_sec == pytest.approx(20.0)


def test_a_storm_that_keeps_writing_stays_eligible() -> None:
    """A storm re-arms on every write: execution readings of 1.0 every 135 s
    (the replay's synthetic storm cadence) never age past the window."""
    hist = [(NOW - timedelta(minutes=5 * i), 0.0) for i in range(1, 2000)]
    t = NOW
    for k in range(40):
        hist.append((t, 1.0))
        for probe in (5, 70, 134):
            now = t + timedelta(seconds=probe)
            m = compute_prediction_error_magnitude(value=1.0, observed_at=t, history=hist, now=now)
            c = node_candidate(node_id="node:substrate.execution", label="e", magnitude=m,
                               observed_at=t, now=now, history_values=[v for _, v in hist])
            assert judge_candidate(c).eligible, (k, probe)
        t += timedelta(seconds=135)


def test_a_storm_still_wins_against_a_per_tick_rival_after_every_write() -> None:
    """Against a body rival sitting in its own top decile all the time, the
    storm wins the first minute after every write (full strength through the
    grace), then shares the frame as it fades -- both are real alarms. It
    never drops out of eligibility."""
    hist = [(NOW - timedelta(minutes=5 * i), 0.0) for i in range(1, 2000)]
    rival = _level(BIOMETRICS, 0.95)
    t = NOW
    wins = 0
    probes = 0
    for k in range(20):
        hist.append((t, 1.0))
        for probe in range(5, 135, 10):
            now = t + timedelta(seconds=probe)
            m = compute_prediction_error_magnitude(value=1.0, observed_at=t, history=hist, now=now)
            c = node_candidate(node_id="node:substrate.execution", label="e", magnitude=m,
                               observed_at=t, now=now, history_values=[v for _, v in hist])
            r = rank_candidates([c, rival])
            assert any(v.candidate is c for v in r.eligible)
            won = r.winner.candidate is c
            if probe <= EVENT_FADE_GRACE_SEC:
                assert won, (k, probe)
            wins += won
            probes += 1
        t += timedelta(seconds=135)
    assert wins / probes >= 0.5


# --- continuous level signals are unaffected -----------------------------------


@pytest.mark.parametrize("node_id", [BIOMETRICS, BUS, "node:substrate.perception", "node:substrate.cabinet"])
def test_level_and_per_tick_signals_do_not_fade(node_id: str) -> None:
    c = _level(node_id, 0.95, age=1500.0)
    assert c.event_window_sec is None
    v = judge_candidate(c)
    if c.source_kind == "internal":
        assert v.eligible and v.event_decay == 1.0 and v.salience == v.score
    else:  # perception: external, absent when its reading is stale (unchanged rule)
        assert c.absent


def test_world_chat_rate_is_not_event_written() -> None:
    """A windowed count recomputed every tick from turn times: its value is
    current, not carried, so it is never faded."""
    turns = [NOW - timedelta(days=d, minutes=5) for d in range(1, 6)] + [NOW - timedelta(minutes=10)]
    c = wf.chat_candidate(turns, now=NOW)
    assert c.event_window_sec is None


# --- absent stays distinct from calm -------------------------------------------


def test_absent_stays_absent_not_old_news() -> None:
    c = node_candidate(node_id=CODEBASE, label="c", magnitude=None, observed_at=None, now=NOW)
    v = judge_candidate(c)
    assert c.absent and v.band == "absent" and v.reason.startswith("absent:")
    ranking = rank_candidates([c])
    assert ranking.trace()["absent_sources"] == [CODEBASE]


def test_aged_out_event_is_old_news_not_absent() -> None:
    v = judge_candidate(_codebase_at(NOW + timedelta(minutes=10)))
    assert not v.candidate.absent and v.band != "absent"
    ranking = rank_candidates([v.candidate])
    assert ranking.no_winner and ranking.trace()["absent_sources"] == []


# --- kill switch ---------------------------------------------------------------


def test_event_decay_off_restores_the_world_first_hold() -> None:
    late = _codebase_at(NOW + timedelta(minutes=18), event_decay=False)
    assert late.event_window_sec is None
    v = judge_candidate(late)
    assert v.eligible and v.event_decay == 1.0


def test_broadcast_threads_the_kill_switch() -> None:
    from types import SimpleNamespace

    import orion.substrate.attention_broadcast as ab

    now = NOW + timedelta(minutes=18)
    node = SimpleNamespace(
        node_id=CODEBASE, label="codebase", node_kind="concept",
        metadata={"prediction_error": 0.988, "dynamic_pressure": 0.5},
        signals=SimpleNamespace(confidence=0.8),
        temporal=SimpleNamespace(observed_at=NOW),
    )
    mag = _codebase_at(now).unusualness
    on = ab.build_substrate_attention_frame(
        nodes=[node], now=now, magnitude_by_node_id={CODEBASE: mag}, world_first=True,
        external_candidates=[],
    )
    off = ab.build_substrate_attention_frame(
        nodes=[node], now=now, magnitude_by_node_id={CODEBASE: mag}, world_first=True,
        external_candidates=[], event_decay=False,
    )
    assert on.debug["world_first"]["no_winner"] is True
    assert off.debug["world_first"]["winner"] == CODEBASE


def test_field_frame_reports_the_faded_salience() -> None:
    """The field contest's target salience (what goal provenance and the frame's
    overall_salience read) is the faded strength, not the onset band."""
    from pathlib import Path

    from orion.attention.field_attention.builder import build_attention_frame
    from orion.attention.field_attention.policy import load_attention_policy
    from orion.schemas.field_state import FieldStateV1

    repo = Path(__file__).resolve().parents[1]
    policy = load_attention_policy(repo / "config" / "attention" / "field_attention_policy.v1.yaml")
    now = NOW + timedelta(seconds=200)
    cand = _codebase_at(now)
    field = FieldStateV1(generated_at=now, tick_id="t_decay", node_vectors={})
    frame = build_attention_frame(
        field=field, policy=policy, prediction_error_baselines={}, previous_frame=None,
        now=now, world_first_candidates=[cand],
    )
    top = frame.dominant_targets[0]
    v = judge_candidate(cand)
    assert top.target_id == CODEBASE and v.event_decay < 1.0
    assert top.salience_score == pytest.approx(v.score * v.event_decay)
    assert frame.overall_salience == pytest.approx(v.score * v.event_decay)
