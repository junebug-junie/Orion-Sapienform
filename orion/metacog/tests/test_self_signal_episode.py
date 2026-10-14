"""self_signal_episode lifecycle (spec 2026-10-07 section D).

Observations are built from REAL attention-seam verdicts (world_first.
node_candidate + judge_candidate) wherever the classification matters, so these
tests pin that the episode reads the seam's one eligibility rule."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from orion.attention.world_first import EVENT_ORIENTING_WINDOW_SEC, judge_candidate, node_candidate
from orion.metacog.self_signal_episode import (
    DEFAULT_OPEN_DWELL_SEC,
    MAX_EVIDENCE_REFS,
    SelfSignalEpisodesStateV1,
    SignalObservation,
    episode_reason,
    episode_upstream,
    mark_open_published,
    needs_open_publish,
    observation_from_verdict,
    open_budget_ok,
    record_open,
    state_from_dict,
    state_to_dict,
    step_signal,
    trigger_wants_prose,
)
from orion.schemas.attention_frame import PredictionErrorMagnitudeV1

T0 = 1_791_400_000.0
SIG = "node:substrate.bus_synaptic"


def _obs(t: float, kind: str, *, pct: float | None = 0.95, band: str = "high", value: float = 0.1, sig: str = SIG):
    if kind == "unknown":
        pct, band = None, "absent"
    elif kind in ("calm", "faded"):
        pct, band = 0.3 if pct == 0.95 else pct, "quiet"
    return SignalObservation(
        signal_id=sig, kind=kind, at=t, reason=kind, label="bus", value=value,
        percentile=pct, band=band, observed_at=t - 5.0,
    )


def _run(seq, ep=None, **kw):
    """seq: list of (t, kind[, pct, value]). Returns (episode, events)."""
    events = []
    for item in seq:
        t, kind = item[0], item[1]
        extra = {}
        if len(item) > 2:
            extra["pct"] = item[2]
            extra["band"] = "unusual" if item[2] >= 0.99 else "high"
        if len(item) > 3:
            extra["value"] = item[3]
        ep, ev = step_signal(ep, _obs(t, kind, **extra), **kw)
        if ev is not None:
            events.append(ev)
            if ev.phase == "open":
                ep = mark_open_published(ep)
    return ep, events


def _hot_run(start: float, n: int, step: float = 30.0):
    return [(start + i * step, "hot") for i in range(n)]


# -- the seam is the one definition ------------------------------------------


def _mag(pct: float | None, band: str, *, value: float = 0.2, age: float = 5.0, n: int = 5000):
    return PredictionErrorMagnitudeV1(
        value=value, age_sec=age, percentile_now=pct, n_readings_7d=n, band=band,
        p50_7d=0.03, p90_7d=0.06,
    )


def _verdict(node: str, mag, *, now: datetime, absent: str | None = None):
    cand = node_candidate(
        node_id=node, label=node, magnitude=mag,
        observed_at=None if mag is None else now - timedelta(seconds=mag.age_sec),
        now=now, absent_reason=absent,
    )
    return judge_candidate(cand)


def test_dwell_is_the_seams_orienting_window():
    assert DEFAULT_OPEN_DWELL_SEC == EVENT_ORIENTING_WINDOW_SEC


@pytest.mark.parametrize(
    "node,mag,absent,kind",
    [
        ("node:substrate.biometrics", _mag(0.95, "high"), None, "hot"),
        ("node:substrate.biometrics", _mag(0.995, "unusual"), None, "hot"),
        ("node:substrate.biometrics", _mag(0.5, "usual"), None, "calm"),
        ("node:substrate.biometrics", _mag(0.0, "quiet"), None, "calm"),
        # absent / no history / stale: silence is not calm
        ("node:substrate.biometrics", _mag(0.99, "unusual"), "db down", "unknown"),
        ("node:substrate.biometrics", None, None, "unknown"),
        ("node:substrate.biometrics", _mag(0.99, "insufficient_history", n=10), None, "unknown"),
        ("node:substrate.biometrics", _mag(0.99, "unusual", age=4000.0), None, "unknown"),
        # placeholder is never a measurement
        ("node:substrate.harness_closure", _mag(0.99, "unusual"), None, "unknown"),
        # event-written: news for 300 s, then faded (carried value, not calm)
        ("node:substrate.execution", _mag(0.97, "high", age=60.0), None, "hot"),
        ("node:substrate.execution", _mag(0.97, "high", age=400.0), None, "faded"),
    ],
)
def test_observation_kind_comes_from_the_attention_verdict(node, mag, absent, kind):
    now = datetime(2026, 10, 14, tzinfo=timezone.utc)
    obs = observation_from_verdict(_verdict(node, mag, now=now, absent=absent), now=now.timestamp())
    assert obs.kind == kind, obs.reason
    if kind == "unknown":
        assert obs.percentile is None


def test_perception_is_external_and_never_an_internal_episode_signal():
    from orion.attention.world_first import source_kind_for_node

    assert source_kind_for_node("node:substrate.perception") == "external"


# -- lifecycle ---------------------------------------------------------------


def test_opens_only_after_dwell_then_closes_after_calm_gap():
    # 10 hot polls = 300 s: not MORE than the dwell -> pending, no row.
    ep, ev = _run(_hot_run(T0, 10))
    assert ev == [] and ep is not None and not ep.confirmed
    # 11th hot poll -> open, onset is the first hot poll.
    ep, ev = _run([(T0 + 300, "hot")], ep)
    assert [e.phase for e in ev] == ["open"]
    assert ev[0].episode.opened_at == T0 and ev[0].episode.confirmed_at == T0 + 300
    # calm for < gap: still open; >= gap: close with duration from onset.
    ep, ev = _run([(T0 + 330 + 30 * i, "calm") for i in range(20)], ep)
    assert ev == [] and ep is not None
    ep, ev = _run([(T0 + 330 + 600, "calm")], ep)
    assert [e.phase for e in ev] == ["close"] and ep is None
    assert ev[0].close_reason == "calm"
    assert ev[0].duration_sec == pytest.approx(930.0)


def test_single_hot_blip_never_opens():
    ep, ev = _run([(T0, "hot")] + [(T0 + 30 * i, "calm") for i in range(1, 40)])
    assert ev == [] and ep is None


def test_flicker_shorter_than_calm_gap_does_not_close():
    seq = _hot_run(T0, 11)
    t = T0 + 330
    for _ in range(6):  # 5 min calm, 1 hot, repeated
        seq += [(t + 30 * i, "calm") for i in range(10)]
        t += 300
        seq.append((t, "hot"))
        t += 30
    ep, ev = _run(seq)
    assert [e.phase for e in ev] == ["open"] and ep is not None


def test_pending_run_that_calms_writes_nothing():
    seq = _hot_run(T0, 8) + [(T0 + 240 + 30 * i, "calm") for i in range(1, 25)]
    ep, ev = _run(seq)
    assert ev == [] and ep is None


def test_sparse_hot_polls_outside_window_do_not_accumulate():
    # one hot poll every 120 s: 5 in any 600 s window = 150 s < dwell.
    seq = []
    for i in range(40):
        seq.append((T0 + 120 * i, "hot"))
        seq += [(T0 + 120 * i + 30 * j, "calm") for j in (1, 2, 3)]
    ep, ev = _run(seq)
    assert ev == []


def test_absent_never_opens():
    ep, ev = _run([(T0 + 30 * i, "unknown") for i in range(100)])
    assert ev == [] and ep is None


def test_absent_does_not_close_or_count_as_calm():
    ep, _ = _run(_hot_run(T0, 11))
    # 20 min of absent: not a recovery, not even the start of a calm gap.
    ep, ev = _run([(T0 + 330 + 30 * i, "unknown") for i in range(40)], ep)
    assert ev == [] and ep is not None and ep.calm_since is None
    # calm, interrupted by absence, then calm again: the gap restarts.
    t = T0 + 330 + 1200
    ep, ev = _run([(t + 30 * i, "calm") for i in range(15)], ep)  # 420 s calm
    ep, ev2 = _run([(t + 450, "unknown")], ep)
    ep, ev3 = _run([(t + 480 + 30 * i, "calm") for i in range(15)], ep)  # 420 s calm
    assert ev == ev2 == ev3 == [] and ep is not None


def test_long_absence_closes_as_went_silent_not_calm():
    ep, _ = _run(_hot_run(T0, 11))
    ep, ev = _run([(T0 + 330 + 30 * i, "unknown") for i in range(121)], ep)
    assert [e.close_reason for e in ev] == ["went_silent"] and ep is None


def test_faded_event_closes_as_faded():
    ep, _ = _run(_hot_run(T0, 11))
    ep, ev = _run([(T0 + 330 + 30 * i, "faded") for i in range(21)], ep)
    assert [e.close_reason for e in ev] == ["faded"]


def test_peak_tracks_highest_percentile_then_raw():
    seq = _hot_run(T0, 5) + [(T0 + 150, "hot", 0.995, 0.2), (T0 + 180, "hot", 0.995, 0.3)] + _hot_run(T0 + 210, 5)
    ep, ev = _run(seq)
    assert ev and ev[0].episode.peak_percentile == 0.995 and ev[0].episode.peak_value == 0.3
    assert ev[0].episode.peak_band == "unusual"


# -- one prose per episode ----------------------------------------------------


def test_one_open_per_episode_and_retry_until_published():
    ep = None
    opens = 0
    for t, k in _hot_run(T0, 40):
        ep, ev = step_signal(ep, _obs(t, k))
        if ev is not None and ev.phase == "open":
            opens += 1
    assert opens == 1
    # never published (bus down): the episode still owes its open row ...
    assert needs_open_publish(ep)
    # ... and once published it is owed nothing more.
    assert not needs_open_publish(mark_open_published(ep))


def test_close_row_carries_no_prose_flag_and_orch_skips_it():
    ep, ev = _run(_hot_run(T0, 11) + [(T0 + 330 + 30 * i, "calm") for i in range(21)])
    open_up, close_up = episode_upstream(ev[0]), episode_upstream(ev[1])
    assert open_up["prose"] is True and close_up["prose"] is False
    assert trigger_wants_prose("self_signal_episode", open_up)
    assert not trigger_wants_prose("self_signal_episode", close_up)
    assert trigger_wants_prose("transport", {"phase": "close"})  # other kinds unchanged


def test_rows_carry_the_contract_fields():
    ep, ev = _run(_hot_run(T0, 11) + [(T0 + 330 + 30 * i, "calm") for i in range(21)])
    up = episode_upstream(ev[1])
    for key in (
        "signal_id", "phase", "band_at_open", "percentile_at_open", "raw_at_open",
        "peak_band", "peak_percentile", "peak_raw", "duration_sec", "close_reason",
        "evidence_refs", "source_table", "episode_id", "opened_at",
    ):
        assert key in up, key
    assert up["evidence_refs"][0].startswith("substrate_node_prediction_error_history:node:substrate.bus_synaptic@")
    assert episode_reason(ev[0]).startswith("self_signal_episode:open:node:substrate.bus_synaptic:band=high")
    assert ":closed=calm" in episode_reason(ev[1])


# -- caps ---------------------------------------------------------------------


def test_evidence_refs_capped_and_keep_the_onset():
    ep, _ = _run(_hot_run(T0, 600))
    assert len(ep.evidence_refs) == MAX_EVIDENCE_REFS
    assert ep.evidence_refs[0].endswith(datetime.fromtimestamp(T0 - 5.0, tz=timezone.utc).isoformat())
    assert len(ep.recent_hot) <= 21  # trailing open window only


def test_open_budget():
    s = SelfSignalEpisodesStateV1()
    for i in range(3):
        assert open_budget_ok(s, T0 + i, max_opens_per_hour=3)
        s = record_open(s, T0 + i)
    assert not open_budget_ok(s, T0 + 10, max_opens_per_hour=3)
    assert open_budget_ok(s, T0 + 3601, max_opens_per_hour=3)


# -- restarts -----------------------------------------------------------------


def test_restart_with_persisted_state_does_not_reopen():
    ep, ev = _run(_hot_run(T0, 11))
    assert [e.phase for e in ev] == ["open"]
    state = SelfSignalEpisodesStateV1(episodes={SIG: ep})
    restored = state_from_dict(json.loads(json.dumps(state_to_dict(state))))
    assert restored.episodes[SIG] == ep
    ep2, ev2 = _run(_hot_run(T0 + 330, 30), restored.episodes[SIG])
    assert ev2 == []


def test_cold_start_adopts_already_hot_signal_silently():
    ep, ev = step_signal(None, _obs(T0, "hot"), cold_start=True)
    assert ev is None and ep.silent
    ep, ev = _run(_hot_run(T0 + 30, 30) + [(T0 + 930 + 30 * i, "calm") for i in range(25)], ep)
    assert ev == [] and ep is None  # no open, no close: we never saw it start


def test_state_from_dict_never_raises():
    assert state_from_dict(None).episodes == {}
    assert state_from_dict({"episodes": {"x": {"bad": 1}}, "recent_opens": ["nan?"]}).episodes == {}


def test_self_signal_cut_points_are_the_seams():
    from orion.metacog.evidence_map import SELF_SIGNAL_PCT_CRITICAL, SELF_SIGNAL_PCT_DEGRADED
    from orion.substrate.prediction_error_magnitude import DEFAULT_BAND_CUTS

    assert (SELF_SIGNAL_PCT_DEGRADED, SELF_SIGNAL_PCT_CRITICAL) == DEFAULT_BAND_CUTS[1:]


def test_went_silent_close_is_not_rendered_as_a_recovery():
    from orion.metacog.evidence_map import map_trigger

    ep, ev = _run(_hot_run(T0, 11) + [(T0 + 330 + 30 * i, "unknown") for i in range(121)])
    m = map_trigger("self_signal_episode", episode_reason(ev[-1]), episode_upstream(ev[-1]))
    assert ev[-1].close_reason == "went_silent"
    assert m.severity != "nominal"
    assert "NOT a recovery" in m.evidence[0] and "returned" not in m.evidence[0]


def test_calm_and_unknown_interleaving_keeps_episode_open():
    """Documented behaviour: a flapping read (unknown at least every 10 min)
    neither completes a calm gap nor a silent close."""
    ep, _ = _run(_hot_run(T0, 11))
    t = T0 + 330
    for _ in range(20):
        ep, ev = _run([(t + 30 * i, "calm") for i in range(10)] + [(t + 300, "unknown")], ep)
        assert ev == []
        t += 330
    assert ep is not None


def test_state_from_dict_drops_corrupt_numeric_fields():
    ep, _ = _run(_hot_run(T0, 11))
    d = state_to_dict(SelfSignalEpisodesStateV1(episodes={SIG: ep}))
    d["episodes"][SIG]["calm_since"] = "garbage"
    assert state_from_dict(d).episodes == {}
